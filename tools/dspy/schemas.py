"""Plain data containers passed between DSPy extraction and YAML merge (no logic)."""
from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class UserIntent:
	"""Structured output of ``json.loads(intent_json)`` after light normalization in modules.py."""
	task: str
	model_name: str
	trainer_name: str
	# maps to fc_deep.fed_hyper_params.federated_model (built-in FedAvg vs FedAvg.py, etc.)
	aggregator_name: str
	# maps to fc_deep.trainer.loss.name (e.g. CrossEntropyLoss, focal_loss.py)
	loss_name: str
	# maps to fc_deep.trainer.data_loader (e.g. ImageLoader.py, RnaLoader.py)
	data_loader_name: str
	raw_request: str


@dataclass
class AgentPlan:
	config_data: Dict[str, Any]


@dataclass
class GenerationResult:
	intent: UserIntent
	plan: AgentPlan
	config_yaml: str
	output_files: Dict[str, str] = field(default_factory=dict)
