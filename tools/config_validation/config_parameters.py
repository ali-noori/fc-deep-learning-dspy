"""
Required vs optional fc_deep configuration parameters per execution mode.

Derived from runtime config access in:
  - CustomStates/ConfigState.py
  - utils/pytorch/states.py (Initialization / LocalUpdate / GlobalAggregation /
    WriteResults / Centralized / Simulation)
  - utils/utils.py (design_architecture)
  - utils/pytorch/DeepModel.py, plugins (conditional kwargs)

There is no separate schema validator yet; "required" means the current
implementation raises KeyError/AttributeError or fails mode selection if the
parameter is absent when that path runs. "optional" means the code uses .get()
with a default, skips when absent, or only reads the key when present.

Paths are dotted under fc_deep (e.g. "local_dataset.train").
List indices use "[]" (e.g. "trainer.metrics[].name").
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Shared notes (apply across modes unless a mode list overrides)
# ---------------------------------------------------------------------------
#
# Mode discriminators (mutually exclusive):
#   - Federated: omit simulation and centralized (both null/absent).
#   - Simulation: simulation must be present; omit centralized.
#   - Centralized: centralized must be present; omit simulation
#     (if both set, Initialization prefers simulation).
#
# local_dataset.test / central_test:
#   get_dataloader() indexes input_files["test"] and input_files["central_test"].
#   Those keys exist only if present under local_dataset, so both keys must appear
#   (central_test may be null). At least one of test or central_test must be
#   non-null or the app reports "no test data".
#
# logic:
#   Entire section optional → defaults mode="file", dir=".".
#   If logic is present, logic.mode and logic.dir are required.
#
# result.model:
#   write_dnn_models() only saves when result.model is non-null, but callers always
#   index output_files["model"], which exists only if the key is under result.
#   So the key must be present (value may be null to skip saving).
#
# model shapes (mutually exclusive):
#   Shape A: model.name present → remaining keys are constructor kwargs.
#   Shape B: model is a list of {type, param?} (no name).
# ---------------------------------------------------------------------------


# =============================================================================
# Federated
# =============================================================================

FEDERATED_REQUIRED = [
    # local_dataset (finalize_config + Initialization)
    "local_dataset",
    "local_dataset.train",
    "local_dataset.test",  # key must exist; see shared notes with central_test
    "local_dataset.central_test",  # key must exist (may be null)
    "local_dataset.detail",  # may be {}; KeyError if section key missing

    # result (finalize_config + WriteResults)
    "result",
    "result.pred",
    "result.target",
    "result.model",  # key required for output_files["model"]; null skips save

    # fed_hyper_params (load_schemas / aggregator / rounds)
    "fed_hyper_params",
    "fed_hyper_params.max_iter",
    "fed_hyper_params.federated_model",  # coordinator get_aggregator()
    "fed_hyper_params.global_updates",  # load_schemas getattr(GlobalUpdates, ...)

    # trainer
    "trainer",
    "trainer.name",
    "trainer.local_updates",  # load_schemas getattr(LocalUpdates, ...)
    "trainer.data_loader",
    "trainer.optimizer",
    "trainer.optimizer.name",
    "trainer.loss",
    "trainer.loss.name",
    "trainer.metrics",  # iterated; empty list skips metrics but key required
    "trainer.metrics[].name",
    "trainer.metrics[].package",

    # train_config (setattr + BasicTrainer attribute reads)
    "train_config",
    "train_config.device",  # build_client_model assigns train_config["device"]
    "train_config.batch_size",
    "train_config.test_batch_size",
    "train_config.epochs",  # BasicTrainer.fit → self.epochs
    "train_config.verbose",  # BasicTrainer.per_epoch_validation → self.verbose

    # model Shape A (named). See FEDERATED_CONDITIONAL for Shape B / kwargs.
    "model",
]

FEDERATED_OPTIONAL = [
    "debug",  # only if "debug" in config

    "local_dataset.init_model",  # coordinator .get(); warm-start when non-null

    # logic section optional as a whole; if omitted, mode=file, dir=.
    "logic",
    "logic.mode",  # required only when logic is present
    "logic.dir",  # required only when logic is present

    "fed_hyper_params.n_classes",  # passed as aggregator kwargs; FedAvg ignores
    "fed_hyper_params.param",  # extra aggregator kwargs; may be absent

    "use_smpc",  # config.get("use_smpc", False)

    "trainer.param",  # trainer.get("param", {})
    "trainer.optimizer.param",  # .get("param", {}); class-specific kwargs inside
    "trainer.loss.param",  # .get("param", {})
    "trainer.metrics[].param",  # .get("param", {})

    "train_config.lr",  # setattr only; optimizer uses trainer.optimizer.param
    "train_config.batch_count",  # used by FedMMb; unused by BasicTrainer

    # Shape A constructor kwargs (required for built-in CNN/cnn.py/mlp.py)
    "model.name",
    "model.n_classes",
    "model.in_features",
]

# Conditional / mutually exclusive (not blindly required for every federated config)
FEDERATED_CONDITIONAL = {
    # logic
    "logic.mode": "Required if logic is present; else defaults apply.",
    "logic.dir": "Required if logic is present; else defaults apply.",
    # test data
    "local_dataset.test|local_dataset.central_test": (
        "At least one must be non-null; both keys must exist for get_dataloader()."
    ),
    # RnaLoader detail kwargs
    "local_dataset.detail.sep": (
        "Required when trainer.data_loader is RnaLoader / RnaLoader.py."
    ),
    "local_dataset.detail.label": (
        "Required when trainer.data_loader is RnaLoader / RnaLoader.py."
    ),
    # FedMMb
    "train_config.batch_count": (
        "Required when trainer.name is FedMMb.py (CustomTrainer uses self.batch_count)."
    ),
    "train_config.epochs": (
        "Required for BasicTrainer; ignored by FedMMb.py (uses batch_count)."
    ),
    # Metric kwargs (torchmetrics classification examples)
    "trainer.metrics[].param.task": (
        "Often required by torchmetrics classification metrics (e.g. Accuracy)."
    ),
    "trainer.metrics[].param.num_classes": (
        "Often required for multiclass metrics; align with model / labels."
    ),
    # Model shapes
    "model.name": (
        "Shape A: if present, design_architecture uses named/plugin Model; "
        "remaining model.* keys are constructor kwargs."
    ),
    "model.n_classes": (
        "Required constructor kwarg for built-in CNN / plugins cnn.py / mlp.py "
        "when using Shape A."
    ),
    "model.in_features": (
        "Required constructor kwarg for CNN/cnn.py/mlp.py when using Shape A "
        "(channels or flat size)."
    ),
    "model[]": (
        "Shape B: use a list of layers instead of model.name "
        "(mutually exclusive with Shape A)."
    ),
    "model[].type": "Required per layer when using Shape B (torch.nn class name).",
    "model[].param": "Optional per layer when using Shape B; may use in_features/in_channels 'None'.",
    # Forbidden for this mode
    "simulation": "Must be absent/null; presence selects Simulation instead of Federated.",
    "centralized": "Must be absent/null; presence selects Centralized after init.",
}

FEDERATED_FORBIDDEN = [
    "simulation",
    "centralized",
]


# =============================================================================
# Simulation
# =============================================================================

SIMULATION_REQUIRED = [
    # mode discriminant
    "simulation",
    "simulation.clients_dir",  # correct_clients_dir(); comma-separated client folders

    # local_dataset
    "local_dataset",
    "local_dataset.train",
    "local_dataset.test",
    "local_dataset.central_test",  # key must exist (may be null); used if non-null
    "local_dataset.detail",

    # result
    "result",
    "result.pred",
    "result.target",
    "result.model",  # Simulation indexes output_files["model"] before save check

    # fed_hyper_params (Simulation uses max_iter as communication rounds)
    "fed_hyper_params",
    "fed_hyper_params.max_iter",
    "fed_hyper_params.federated_model",
    "fed_hyper_params.global_updates",

    # trainer
    "trainer",
    "trainer.name",
    "trainer.local_updates",
    "trainer.data_loader",
    "trainer.optimizer",
    "trainer.optimizer.name",
    "trainer.loss",
    "trainer.loss.name",
    "trainer.metrics",
    "trainer.metrics[].name",
    "trainer.metrics[].package",

    # train_config
    "train_config",
    "train_config.device",
    "train_config.batch_size",
    "train_config.test_batch_size",
    "train_config.epochs",
    "train_config.verbose",

    "model",
]

SIMULATION_OPTIONAL = [
    "debug",

    # Allowed under local_dataset / finalize_config, but Simulation never load_model()
    "local_dataset.init_model",

    "logic",
    "logic.mode",
    "logic.dir",

    "fed_hyper_params.n_classes",
    "fed_hyper_params.param",

    # Allowed in templates; Simulation never applies config use_smpc (smpc_used stays False)
    "use_smpc",

    "trainer.param",
    "trainer.optimizer.param",
    "trainer.loss.param",
    "trainer.metrics[].param",

    "train_config.lr",
    "train_config.batch_count",

    "model.name",
    "model.n_classes",
    "model.in_features",
]

SIMULATION_CONDITIONAL = {
    "logic.mode": "Required if logic is present; else defaults apply.",
    "logic.dir": "Required if logic is present; else defaults apply.",
    "local_dataset.test|local_dataset.central_test": (
        "At least one must be non-null; both keys must exist for get_dataloader()."
    ),
    "local_dataset.detail.sep": (
        "Required when trainer.data_loader is RnaLoader / RnaLoader.py."
    ),
    "local_dataset.detail.label": (
        "Required when trainer.data_loader is RnaLoader / RnaLoader.py."
    ),
    "local_dataset.init_model": (
        "Optional structurally; unused on Simulation path (no warm-start)."
    ),
    "use_smpc": (
        "Optional structurally; Simulation does not copy config use_smpc into store."
    ),
    "fed_hyper_params.max_iter": (
        "Communication rounds (same as Federated), not Centralized epoch override."
    ),
    "train_config.batch_count": (
        "Required when trainer.name is FedMMb.py."
    ),
    "train_config.epochs": (
        "Required for BasicTrainer; ignored by FedMMb.py."
    ),
    "trainer.metrics[].param.task": (
        "Often required by torchmetrics classification metrics."
    ),
    "trainer.metrics[].param.num_classes": (
        "Often required for multiclass metrics."
    ),
    "model.name": "Shape A named/plugin model (mutually exclusive with Shape B list).",
    "model.n_classes": "Required for CNN/cnn.py/mlp.py under Shape A.",
    "model.in_features": "Required for CNN/cnn.py/mlp.py under Shape A.",
    "model[]": "Shape B layer list (mutually exclusive with model.name).",
    "model[].type": "Required per layer for Shape B.",
    "model[].param": "Optional per layer for Shape B.",
    "centralized": "Must be absent/null; if both set, simulation still wins.",
}

SIMULATION_FORBIDDEN = [
    "centralized",
]


# =============================================================================
# Centralized
# =============================================================================

CENTRALIZED_REQUIRED = [
    # mode discriminant
    "centralized",  # any non-null value

    # local_dataset (full Initialization runs before Centralized)
    "local_dataset",
    "local_dataset.train",
    "local_dataset.test",
    "local_dataset.central_test",
    "local_dataset.detail",

    # result
    "result",
    "result.pred",
    "result.target",
    "result.model",

    # fed_hyper_params — max_iter required by Centralized.run (epochs override);
    # federated_model / global_updates still required because Initialization builds
    # aggregator + load_schemas before branching to Centralized.
    "fed_hyper_params",
    "fed_hyper_params.max_iter",
    "fed_hyper_params.federated_model",
    "fed_hyper_params.global_updates",

    # trainer — local_updates still required by load_schemas at init
    "trainer",
    "trainer.name",
    "trainer.local_updates",
    "trainer.data_loader",
    "trainer.optimizer",
    "trainer.optimizer.name",
    "trainer.loss",
    "trainer.loss.name",
    "trainer.metrics",
    "trainer.metrics[].name",
    "trainer.metrics[].package",

    # train_config
    "train_config",
    "train_config.device",
    "train_config.batch_size",
    "train_config.test_batch_size",
    # epochs still needed at BasicTrainer construction; Centralized then overwrites
    # model.epochs from fed_hyper_params.max_iter
    "train_config.epochs",
    "train_config.verbose",

    "model",
]

CENTRALIZED_OPTIONAL = [
    "debug",

    "local_dataset.init_model",  # coordinator load_model during Initialization (used)

    "logic",
    "logic.mode",
    "logic.dir",

    "fed_hyper_params.n_classes",  # init kwargs only; Centralized.run never aggregates
    "fed_hyper_params.param",

    "use_smpc",  # stored at init; Centralized.run never uses SMPC paths

    "trainer.param",
    "trainer.optimizer.param",
    "trainer.loss.param",
    "trainer.metrics[].param",

    "train_config.lr",
    "train_config.batch_count",

    "model.name",
    "model.n_classes",
    "model.in_features",
]

CENTRALIZED_CONDITIONAL = {
    "logic.mode": "Required if logic is present; else defaults apply.",
    "logic.dir": "Required if logic is present; else defaults apply.",
    "local_dataset.test|local_dataset.central_test": (
        "At least one must be non-null; both keys must exist for get_dataloader()."
    ),
    "local_dataset.detail.sep": (
        "Required when trainer.data_loader is RnaLoader / RnaLoader.py."
    ),
    "local_dataset.detail.label": (
        "Required when trainer.data_loader is RnaLoader / RnaLoader.py."
    ),
    "fed_hyper_params.max_iter": (
        "In Centralized, overwrites trainer epochs for BasicTrainer "
        "(not communication rounds). FedMMb still uses train_config.batch_count."
    ),
    "train_config.epochs": (
        "Required for BasicTrainer construction; value then overridden by "
        "fed_hyper_params.max_iter in Centralized.run."
    ),
    "train_config.batch_count": (
        "Required when trainer.name is FedMMb.py; then controls length instead of max_iter/epochs."
    ),
    "fed_hyper_params.federated_model": (
        "Still required at Initialization (coordinator builds aggregator) even though "
        "Centralized.run never calls aggregate()."
    ),
    "fed_hyper_params.global_updates": (
        "Still required by load_schemas() at Initialization; unused by Centralized.run."
    ),
    "trainer.local_updates": (
        "Still required by load_schemas() at Initialization; Centralized does not pack network updates."
    ),
    "use_smpc": (
        "Optional; stored during Initialization but unused by Centralized.run."
    ),
    "trainer.metrics[].param.task": (
        "Often required by torchmetrics classification metrics."
    ),
    "trainer.metrics[].param.num_classes": (
        "Often required for multiclass metrics."
    ),
    "model.name": "Shape A named/plugin model (mutually exclusive with Shape B list).",
    "model.n_classes": "Required for CNN/cnn.py/mlp.py under Shape A.",
    "model.in_features": "Required for CNN/cnn.py/mlp.py under Shape A.",
    "model[]": "Shape B layer list (mutually exclusive with model.name).",
    "model[].type": "Required per layer for Shape B.",
    "model[].param": "Optional per layer for Shape B.",
    "simulation": "Must be absent/null; presence selects Simulation instead of Centralized.",
}

CENTRALIZED_FORBIDDEN = [
    "simulation",
]


# =============================================================================
# Combined lookup (for later skeleton validation imports)
# =============================================================================

CONFIG_PARAMETERS = {
    "federated": {
        "required": FEDERATED_REQUIRED,
        "optional": FEDERATED_OPTIONAL,
        "conditional": FEDERATED_CONDITIONAL,
        "forbidden": FEDERATED_FORBIDDEN,
    },
    "simulation": {
        "required": SIMULATION_REQUIRED,
        "optional": SIMULATION_OPTIONAL,
        "conditional": SIMULATION_CONDITIONAL,
        "forbidden": SIMULATION_FORBIDDEN,
    },
    "centralized": {
        "required": CENTRALIZED_REQUIRED,
        "optional": CENTRALIZED_OPTIONAL,
        "conditional": CENTRALIZED_CONDITIONAL,
        "forbidden": CENTRALIZED_FORBIDDEN,
    },
}

EXECUTION_MODES = ("federated", "simulation", "centralized")


def get_mode_parameters(mode: str) -> dict:
    """Return required/optional/conditional/forbidden dict for one execution mode."""
    key = mode.strip().lower()
    if key not in CONFIG_PARAMETERS:
        raise ValueError(
            f"Unknown execution mode {mode!r}. Expected one of: {', '.join(EXECUTION_MODES)}."
        )
    return CONFIG_PARAMETERS[key]
