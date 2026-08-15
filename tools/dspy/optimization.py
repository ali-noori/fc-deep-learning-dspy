"""
Config template + artifact discovery + merge (no DSPy / no LM).

``discover_repo_artifacts`` feeds ``tools/dspy/modules.py`` with allowed names.
``dspy_config_output`` is the *only* place we map ``UserIntent`` fields onto YAML
paths (model, trainer, federated_model, loss.name, data_loader).

See DSPY_IMPLEMENTATION.md for the full pipeline diagram.
"""
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict

import yaml

from .schemas import AgentPlan

DEFAULT_TEMPLATE_FILENAME = "config.maximum.yml"
GENERATED_CONFIG_DIR_REL = Path("tools") / "output" / "config" / "generated"
GENERATED_CONFIG_FILENAME = "config.generated.dspy.yml"


def discover_repo_artifacts(repo_root: str):
	"""Return dict of lists: models, trainers, dataloaders, aggregators, losses, … for LM context."""
	root = Path(repo_root)

	def py_files_in_folder(path: Path):
		p = Path(path)
		if not p.exists() or not p.is_dir():
			return []
		out = []
		try:
			for entry in p.iterdir():
				try:
					if not entry.is_file():
						continue
					if entry.suffix != ".py":
						continue
					if entry.name == "__init__.py" or entry.name.startswith("."):
						continue
					out.append(entry.name)
				except OSError:
					continue
			out.sort(key=str.casefold)
			return out
		except Exception:
			return []

	# Built-in names not represented as files under plugins/
	_aggregator_builtins = ["FedAvg"]
	_loss_builtins = ["CrossEntropyLoss"]
	_dataloader_builtins = ["ImageLoader", "ImageLoader.py"]
	artifacts = {
		"models": py_files_in_folder(root / "plugins" / "models") + py_files_in_folder(root / "models" / "pytorch"),
		"trainers": py_files_in_folder(root / "plugins" / "trainers"),
		"dataloaders": _dataloader_builtins + py_files_in_folder(root / "plugins" / "dataloaders"),
		"aggregators": _aggregator_builtins + py_files_in_folder(root / "plugins" / "aggregators"),
		"losses": _loss_builtins + py_files_in_folder(root / "plugins" / "loss"),
		"optimizers": ["SGD", "Adam", "AdamW", "RMSprop"],
		"devices": ["cpu", "gpu"],
	}

	for key, items in artifacts.items():
		if isinstance(items, list):
			artifacts[key] = list(dict.fromkeys(items))

	return artifacts


def load_default_template_config(repo_root: str, template_filename: str = DEFAULT_TEMPLATE_FILENAME) -> Dict[str, Any]:
	cfg_path = Path(repo_root) / template_filename
	if not cfg_path.exists():
		return {"fc_deep": {}}

	with cfg_path.open("r", encoding="utf-8") as f:
		loaded = yaml.safe_load(f) or {}

	if "fc_deep" not in loaded:
		loaded = {"fc_deep": loaded}

	return loaded


def default_generated_output_dir(repo_root: str) -> Path:
	return Path(repo_root) / GENERATED_CONFIG_DIR_REL


def dspy_config_output(intent, base_config, artifacts):
	"""Deep-copy ``base_config`` and overwrite only non-empty fields on ``intent``.

	``artifacts`` is unused today but kept so callers can extend merge logic later
	(e.g. validate names against discovered lists).
	"""
	_ = artifacts
	cfg = deepcopy(base_config)
	fc = cfg.setdefault("fc_deep", {})
	model = fc.setdefault("model", {})
	trainer = fc.setdefault("trainer", {})
	fed = fc.setdefault("fed_hyper_params", {})

	if getattr(intent, "model_name", None):
		model["name"] = intent.model_name

	if getattr(intent, "trainer_name", None):
		trainer["name"] = intent.trainer_name

	# YAML key is federated_model; plugins use e.g. FedAvg.py, built-in is FedAvg
	if getattr(intent, "aggregator_name", None):
		fed["federated_model"] = intent.aggregator_name

	if getattr(intent, "loss_name", None):
		loss = trainer.setdefault("loss", {})
		loss["name"] = intent.loss_name

	if getattr(intent, "data_loader_name", None):
		trainer["data_loader"] = intent.data_loader_name

	return AgentPlan(config_data=cfg)


def write_outputs(result, output_dir):
	out = Path(output_dir)
	out.mkdir(parents=True, exist_ok=True)

	config_path = out / GENERATED_CONFIG_FILENAME
	config_path.write_text(result.config_yaml, encoding="utf-8")

	files = {"config": str(config_path)}
	result.output_files.update(files)
	return files
