"""
DSPy layer: natural language -> structured intent -> Python merges into YAML template.

Mental model for juniors
------------------------
1) ``discover_repo_artifacts`` (optimization.py) scans the repo and builds lists like
   ``["cnn.py", "mlp.py"]``, ``["FedAvg", "FedAvg.py"]``. Those become the ``known_*``
   inputs to the LM so it can only *choose* names we actually have (we still don't
   regex-parse the user's English for those choices).

2) ``IntentExtraction`` is a DSPy *Signature*: it tells the framework what the LM
   receives (sentence + known lists) and what it must return (``intent_json``). The
   long ``desc=`` strings are *prompt engineering* baked into the class.

3) ``dspy.Predict(IntentExtraction)`` is the *student*: one LM call produces
   ``intent_json``. Optionally ``BootstrapFewShot`` wraps it and "trains" it on
   ``_bootstrap_examples`` (many extra LM calls at compile time).

4) ``extract_intent`` parses JSON into ``UserIntent``. ``_intent_str`` tolerates
   common LM mistakes (e.g. ``loss`` instead of ``loss_name``) without scraping the
   original user text for keywords.

5) The pipeline (pipeline.py) then calls ``dspy_config_output`` to copy
   ``config.maximum.yml`` and overwrite only specific keys. The LM never emits the
   full YAML file.

See also: DSPY_IMPLEMENTATION.md in this folder.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional
from urllib.request import urlopen

from .lm_secrets import force_ollama_only, get_groq_api_key, get_groq_model_spec
from .ollama_settings import get_ollama_base_url, get_ollama_model
from .schemas import UserIntent

try:
	import dspy

	HAS_DSPY = True

	# DSPy Signature = contract for one LM step: inputs (fields below) + outputs (intent_json).
	class IntentExtraction(dspy.Signature):
		user_request = dspy.InputField(desc="Natural-language request")
		known_models = dspy.InputField(desc="Known model names")
		known_trainers = dspy.InputField(desc="Known trainer names")
		known_dataloaders = dspy.InputField(desc="Known dataloader names")
		known_aggregators = dspy.InputField(desc="Known aggregator names")
		known_losses = dspy.InputField(desc="Known loss names")
		known_optimizers = dspy.InputField(desc="Known optimizer names")
		known_devices = dspy.InputField(desc="Known device names")

		intent_json = dspy.OutputField(
			desc=(
				"A single line of valid JSON only: one object that json.loads can parse completely. "
				"No text before or after. "
				"If the user asks to use or switch dataloader / data loader, you MUST set data_loader_name to one exact "
				"string from known_dataloaders (e.g. RnaLoader.py). Do not put that only in task. "
				"Same idea for model → model_name, trainer → trainer_name, aggregator → aggregator_name, loss → loss_name. "
				"Keys: task (optional short string like federated_training), model_name, trainer_name, aggregator_name, "
				"loss_name, data_loader_name — each optional but use the right key for what changed. "
				"If the user says focal loss or focal_loss.py, set loss_name to the exact plugin string focal_loss.py from known_losses. "
				"Never omit loss_name when the user named a loss. Values must match known_* lists exactly."
			)
		)

except Exception:
	HAS_DSPY = False


def _intent_str(payload: Dict[str, Any], *keys: str) -> str:
	"""First non-empty string among JSON keys (LM sometimes uses wrong key names)."""
	for k in keys:
		v = payload.get(k)
		if v is None:
			continue
		s = str(v).strip()
		if s:
			return s
	return ""


def _skip_bootstrap_from_env():
	v = os.environ.get("FC_DSPY_SKIP_BOOTSTRAP", "").strip().lower()
	return v in ("1", "true", "yes")


def ollama_up(base_url: str):
	try:
		with urlopen(f"{base_url}/api/tags", timeout=2):
			return True
	except Exception:
		return False


class DSPyModules:
	"""Holds the configured LM and the compiled Predict (or BootstrapFewShot) module."""

	def __init__(self, artifacts: Dict[str, Any], skip_bootstrap: Optional[bool] = None):
		self.artifacts = artifacts or {}
		self.use_dspy = HAS_DSPY
		self.predictor = None
		self._predictor_init_error: Optional[BaseException] = None
		self._backend: Optional[str] = None
		if skip_bootstrap is not None:
			self._skip_bootstrap = bool(skip_bootstrap)
		else:
			self._skip_bootstrap = _skip_bootstrap_from_env()

		self._ollama_base_url = get_ollama_base_url()
		self._ollama_model = get_ollama_model()
		self.ollama_available = ollama_up(self._ollama_base_url)

		# Prefer Groq when a key exists; Ollama is fallback for local dev or when forced.
		groq_key = get_groq_api_key()
		if self.use_dspy and groq_key and not force_ollama_only():
			try:
				dspy.configure(lm=dspy.LM(get_groq_model_spec(), api_key=groq_key))
				student = dspy.Predict(IntentExtraction)
				self.predictor = self._wrap_student(student)
				self._backend = "groq"
			except BaseException as exc:
				self._predictor_init_error = exc
				self.predictor = None

		if self.use_dspy and self.predictor is None and self.ollama_available:
			try:
				dspy.configure(
					lm=dspy.LM(
						f"ollama/{self._ollama_model}",
						api_base=self._ollama_base_url,
					)
				)
				student = dspy.Predict(IntentExtraction)
				self.predictor = self._wrap_student(student)
				self._backend = "ollama"
			except BaseException as exc:
				self._predictor_init_error = exc
				self.predictor = None

	def extract_intent(self, request):
		"""Run the LM once (or use cached demos) and return a ``UserIntent`` dataclass."""
		if not self.use_dspy:
			raise RuntimeError(
				"DSPy is not installed in this environment. Install `dspy-ai` in the project venv."
			)
		if self._backend != "groq" and not self.ollama_available:
			raise RuntimeError(
				f"Ollama is not reachable at {self._ollama_base_url} (expected GET /api/tags). "
				"Start Ollama or set GROQ_API_KEY (env or tools/local_secrets.py) to use Groq instead."
			)
		if not self.predictor:
			hint = ""
			if self._predictor_init_error:
				hint = f" Predictor compile failed: {self._predictor_init_error!r}"
			raise RuntimeError(
				"DSPy intent predictor did not initialize (Groq/Ollama compilation failed)." + hint
			)

		try:
			# All ``known_*`` strings go into the prompt so the model aligns with repo reality.
			out = self.predictor(
				user_request=request,
				known_models=str(self.artifacts.get("models", [])),
				known_trainers=str(self.artifacts.get("trainers", [])),
				known_dataloaders=str(self.artifacts.get("dataloaders", [])),
				known_aggregators=str(self.artifacts.get("aggregators", [])),
				known_losses=str(self.artifacts.get("losses", [])),
				known_optimizers=str(self.artifacts.get("optimizers", [])),
				known_devices=str(self.artifacts.get("devices", [])),
			)

			payload = json.loads(out.intent_json)
			task = payload.get("task") or "federated_training"
			model_name = _intent_str(payload, "model_name")
			trainer_name = _intent_str(payload, "trainer_name")
			aggregator_name = _intent_str(payload, "aggregator_name", "federated_model")
			loss_name = _intent_str(payload, "loss_name", "loss")
			data_loader_name = _intent_str(payload, "data_loader_name", "data_loader")
			return UserIntent(
				task=task,
				model_name=model_name,
				trainer_name=trainer_name,
				aggregator_name=aggregator_name,
				loss_name=loss_name,
				data_loader_name=data_loader_name,
				raw_request=request,
			)
		except Exception as exc:
			lm_desc = (
				get_groq_model_spec() if self._backend == "groq" else f"ollama/{self._ollama_model}"
			)
			raise RuntimeError(
				f"LM inference failed for intent extraction (backend={self._backend!r} lm={lm_desc!r}): {exc}"
			) from exc

	def _wrap_student(self, student):
		# Bootstrap = many LM calls during compile; skip for fast runs and Groq TPM limits.
		if self._skip_bootstrap:
			return student
		return dspy.BootstrapFewShot(metric=self._intent_metric).compile(
			student=student, trainset=self._bootstrap_examples()
		)

	def _bootstrap_examples(self):
		# Few-shot pairs: same shape as production inputs; gold ``intent_json`` teaches phrasing.
		km = str(self.artifacts.get("models", ["cnn.py", "mlp.py"]))
		kt = str(self.artifacts.get("trainers", ["FedMMb.py"]))
		kd = str(self.artifacts.get("dataloaders", ["ImageLoader.py", "RnaLoader.py"]))
		ka = str(self.artifacts.get("aggregators", ["FedAvg", "FedAvg.py"]))
		kl = str(self.artifacts.get("losses", ["CrossEntropyLoss", "focal_loss.py"]))
		ko = str(self.artifacts.get("optimizers", ["SGD", "Adam", "AdamW", "RMSprop"]))
		kdev = str(self.artifacts.get("devices", []))

		inp = (
			"user_request",
			"known_models",
			"known_trainers",
			"known_dataloaders",
			"known_aggregators",
			"known_losses",
			"known_optimizers",
			"known_devices",
		)

		def ex(text, intent_dict):
			return dspy.Example(
				user_request=text,
				known_models=km,
				known_trainers=kt,
				known_dataloaders=kd,
				known_aggregators=ka,
				known_losses=kl,
				known_optimizers=ko,
				known_devices=kdev,
				intent_json=json.dumps(intent_dict),
			).with_inputs(*inp)

		return [
			ex("Use the CNN model with the FedMMb trainer", {"model_name": "cnn.py", "trainer_name": "FedMMb.py"}),
			ex("Switch to cnn.py and use FedMMb", {"model_name": "cnn.py", "trainer_name": "FedMMb.py"}),
			ex("Switch to cnn.py", {"model_name": "cnn.py"}),
			ex("Please switch to CNN", {"model_name": "cnn.py"}),
			ex("Switch to the MLP model", {"model_name": "mlp.py"}),
			ex("Prefer cnn.py and keep current trainer", {"model_name": "cnn.py"}),
			ex("Set the trainer to FedMMb for training", {"trainer_name": "FedMMb.py"}),
			ex("Use mlp.py with FedMMb", {"model_name": "mlp.py", "trainer_name": "FedMMb.py"}),
			ex("Use built-in FedAvg for aggregation", {"aggregator_name": "FedAvg"}),
			ex("Use the FedAvg plugin from aggregators", {"aggregator_name": "FedAvg.py"}),
			ex("Switch model to mlp.py and use FedAvg averaging", {"model_name": "mlp.py", "aggregator_name": "FedAvg"}),
			ex("Use cross entropy loss", {"loss_name": "CrossEntropyLoss"}),
			ex("Switch to focal loss", {"loss_name": "focal_loss.py"}),
			ex("Use focal_loss.py for training", {"loss_name": "focal_loss.py"}),
			ex("Use the image loader", {"data_loader_name": "ImageLoader.py"}),
			ex("Switch to RnaLoader for tabular data", {"data_loader_name": "RnaLoader.py"}),
			ex("Use built-in ImageLoader without .py suffix", {"data_loader_name": "ImageLoader"}),
			ex(
				"Use model mlp.py, trainer FedMMb.py, federated aggregator FedAvg, loss focal_loss.py, and data loader ImageLoader.py",
				{
					"model_name": "mlp.py",
					"trainer_name": "FedMMb.py",
					"aggregator_name": "FedAvg",
					"loss_name": "focal_loss.py",
					"data_loader_name": "ImageLoader.py",
				},
			),
		]

	@staticmethod
	def _intent_metric(gold, pred, trace=None):
		# Score for BootstrapFewShot: fraction of intent keys matching between gold and prediction.
		try:
			gold_intent = json.loads(getattr(gold, "intent_json", "{}"))
			pred_intent = json.loads(getattr(pred, "intent_json", "{}"))
			keys = ("model_name", "trainer_name", "aggregator_name", "loss_name", "data_loader_name")
			matches = 0
			for k in keys:
				if gold_intent.get(k) == pred_intent.get(k):
					matches += 1
			return matches / len(keys)
		except Exception:
			return 0.0
