# Configuration incompatibilities and dependencies

Authoritative reference for **semantic validity** of `fc_deep` configuration in this repository. Rules are derived from runtime code paths, not from README prose alone.

**Source-of-truth files (primary):**

| Area | Files |
|------|--------|
| Config load / paths / splits | `CustomStates/ConfigState.py` |
| Execution branches | `utils/pytorch/states.py` (`Initialization`, `LocalUpdate`, `GlobalAggregation`, `Centralized`, `Simulation`, `WriteResults`) |
| State registration | `states.py` |
| Update packet schemas | `utils/pytorch/utils.py` (`LocalUpdates`, `GlobalUpdates`) |
| Aggregation protocol | `utils/pytorch/optimizer.py`, `plugins/aggregators/FedAvg.py` |
| Local/client protocol | `utils/pytorch/ClientModels.py`, `utils/pytorch/DeepModel.py` |
| Module resolution | `utils/utils.py` (`get_custom_module`, `get_dataloader`, `get_trainer`, `get_aggregator`, `get_loss_func`, `unpack`, `interpret_global_updates`) |
| Built-in dataloaders | `utils/pytorch/DataLoader.py` (`ImageLoader`) |
| Plugin dataloaders | `plugins/dataloaders/ImageLoader.py`, `plugins/dataloaders/RnaLoader.py` |
| Plugin losses | `plugins/loss/focal_loss.py` |
| Plugin trainers | `plugins/trainers/FedMMb.py` |
| Plugin models | `plugins/models/cnn.py`, `plugins/models/mlp.py` |
| Built-in model | `models/pytorch/models.py` (`CNN`) |
| Mode templates | `tools/execution_mode_dspy/valid_config_file/*.template.yml` |

**Enforcement legend:**

| Tag | Meaning |
|-----|---------|
| **Explicit** | Code raises, asserts, logs ERROR, or returns early with a clear failure |
| **Implicit** | Code runs but produces wrong results, silent misbehavior, or hangs |
| **Structural** | YAML shape / enum / file-layout requirement enforced before training logic |

---

# Part 1 — Required dependencies and conditional requirements

Each rule lists: **trigger → required fields → optional fields → default → consequence if violated → code origin**.

---

## 1.1 Top-level config envelope

| Rule ID | Trigger | Required | Optional | Default | If violated | Enforcement | Code |
|---------|---------|----------|----------|---------|-------------|-------------|------|
| **ENV-01** | App starts | Root YAML contains app block `fc_deep:` | — | — | `KeyError` reading config | **Explicit** | `ConfigState.read_config()` → `bios.read(...)[self.app_name]` |
| **ENV-02** | App starts | `mnt/input/config.yml` exists on container input | — | — | `FileNotFoundError` | **Explicit** | `ConfigState.config_file` |
| **ENV-03** | Plugin fields (`*.py` names) | Referenced `.py` files present on `mnt/input/` (generic dir) | — | — | `module not found` ERROR state | **Explicit** | `get_custom_module()` returns `None` → `get_dataloader()` / `build_client_model()` |

---

## 1.2 Execution mode: federated vs simulation vs centralized

| Rule ID | Trigger | Required | Optional | Default | If violated | Enforcement | Code |
|---------|---------|----------|----------|---------|-------------|-------------|------|
| **MODE-01** | Standard multi-container federated run | **Do not** set `fc_deep.simulation` | `fc_deep.centralized` must be absent/false | Federated `Local Update` ↔ `Global Aggregation` loop | Simulation path taken instead | **Explicit** (branch) | `Initialization.run()` checks `simulation` **first** (`states.py` B1) |
| **MODE-02** | Simulation run | `fc_deep.simulation.clients_dir` non-empty comma-separated folder names | All other `fc_deep` sections per template | — | Wrong/missing client paths → `File Not Found` / empty loaders | **Implicit** → **Explicit** at load | `Simulation.correct_clients_dir()`, `inject_root_path_to_clients_dir()` |
| **MODE-03** | Centralized run | `fc_deep.centralized` truthy | `simulation` must be absent/null | — | Simulation wins; centralized never reached | **Explicit** (branch) | `Initialization.run()` |
| **MODE-04** | Any mode | `simulation` + `centralized` both set | — | Simulation precedence | Centralized ignored | **Implicit** | `Initialization.run()` order |
| **MODE-05** | Simulation | Single coordinator process only | — | — | N/A in FC (by design) | **Structural** | `states.py` — `Federated Simulation` registered `Role.COORDINATOR` only |
| **MODE-06** | Centralized | Coordinator-only training state | — | — | Participants never enter centralized loop | **Structural** | `CentralizedTraining` — `Role.COORDINATOR` |
| **MODE-07** | Federated | ≥2 FeatureCloud client mounts with matching fold layout | Coordinator among clients | — | Aggregation / path errors | **Structural** | `Run_App.md`, `GlobalAggregation.gather_local_models()` |

**`max_iter` meaning depends on mode (same key, different semantics):**

| Mode | `fed_hyper_params.max_iter` means | Code |
|------|-----------------------------------|------|
| Federated | Max **communication rounds** | `FedOptimizer.post_aggregate()` / aggregator `iteration` |
| Simulation | Max **communication rounds** | `Simulation.run()` `for c_round in range(self.max_iter)` |
| Centralized | **Epochs per CV fold** (not FL rounds) | `Centralized.run()` sets `client_model.model.epochs = max_iter` |

---

## 1.3 `local_dataset` — filenames, `detail`, and loader kwargs

| Rule ID | Trigger | Required | Optional | Default | If violated | Enforcement | Code |
|---------|---------|----------|----------|---------|-------------|-------------|------|
| **DATA-01** | Always | `local_dataset.train` string filename | — | — | No sample shape / empty paths | **Explicit** | `Initialization.get_dataloader(input_files['train'][0])` |
| **DATA-02** | Always | `local_dataset.test` string filename **or** `central_test` set | `central_test` | `null` | `"no test data"` ERROR | **Explicit** | `get_dataloader()` lines 143–145 |
| **DATA-03** | `trainer.data_loader` = `RnaLoader` / `RnaLoader.py` | `local_dataset.detail.sep` | — | — | `KeyError: 'sep'` | **Explicit** | `RnaLoader.__init__` → `kwargs['sep']` |
| **DATA-04** | `RnaLoader` | `local_dataset.detail.label` (column name) | — | — | `KeyError: 'label'` | **Explicit** | `RnaLoader.__init__` |
| **DATA-05** | `ImageLoader` / `ImageLoader.py` | `local_dataset.detail` may be `{}` | extra kwargs ignored | `{}` | Usually OK | — | `get_dataloader()` passes `**detail` |
| **DATA-06** | `logic.mode: directory` | Physical files `{split}/{train}`, `{split}/{test}` exist per split | `central_test`, `init_model` | — | `File Not Found`, loader returns `None` | **Explicit** (prints) / **Implicit** (None loader) | `ImageLoader.file_exits()`, `RnaLoader.file_exits()` |
| **DATA-07** | `central_test` non-null | Coordinator can resolve `input_files['central_test'][0]` on disk | — | — | Central eval skipped or fails | **Implicit** | `Initialization.run()` coordinator branch |
| **DATA-08** | `init_model` set | File exists on coordinator input | — | — | Load failure | **Explicit** | `client_model.load_model()` |
| **DATA-09** | Any | Only use **string filename keys** under `local_dataset` for real files | `detail` must be **dict** | — | Nonsense synthetic paths for non-file keys | **Implicit** | `finalize_config()` iterates **all** `local_dataset` keys |

**Do not add spurious keys under `local_dataset`** (e.g. `metadata: "x"`): `finalize_config()` builds `input_files['metadata'] = ["{split}/x", ...]` even though nothing consumes it.

---

## 1.4 `logic` — split discovery

| Rule ID | Trigger | Required | Optional | Default | If violated | Enforcement | Code |
|---------|---------|----------|----------|---------|-------------|-------------|------|
| **LOGIC-01** | `logic.mode: directory` | Subdirectories under `mnt/input/{logic.dir}/` | — | `mode: file`, `dir: .` if `logic` omitted | Zero splits → empty training | **Implicit** | `ConfigState.finalize_config()` `os.scandir` |
| **LOGIC-02** | `logic.mode: file` | Single split root = `input_dir` | — | — | Multi-fold layout ignored | **Structural** | `finalize_config()` `splits = [input_dir]` |
| **LOGIC-03** | Federated + `directory` | **Same number of fold subdirs on every client mount** | — | — | Aggregation shape mismatch | **Explicit** (assert) / crash | `FedAvg.aggregate()` assumes `len(params[0])` consistent |
| **LOGIC-04** | Simulation | Fold dirs at root (`mnt/input/data/0`, …) **and** per-client trees (`c1/data/0`, …) | — | — | Missing `mnt/input/data` or client paths | **Explicit** | `Simulation` layout (`Run_App.md`) |

---

## 1.5 `fed_hyper_params` — aggregator constructor kwargs

| Rule ID | Trigger | Required | Optional | Default | If violated | Enforcement | Code |
|---------|---------|----------|----------|---------|-------------|-------------|------|
| **FED-01** | Coordinator federated/simulation | `fed_hyper_params.max_iter` (int ≥ 1) | — | — | TypeError / wrong loop bounds | **Explicit** | Passed to `FedOptimizer(**param)` |
| **FED-02** | Coordinator | `fed_hyper_params.federated_model` resolvable name | — | — | Aggregator `None` / crash | **Explicit** | `get_aggregator()` |
| **FED-03** | Coordinator | `fed_hyper_params.global_updates` exact `GlobalUpdates` enum name | — | — | `AttributeError` | **Explicit** | `getattr(GlobalUpdates, name)` in `load_schemas()` |
| **FED-04** | Round-limited federated training | `global_updates` must include **STOPPING** (e.g. `WEIGHTS_STOPPING`) | — | — | Training may not terminate on `max_iter` | **Implicit** | `all_converged()` — `bool(stoppage)` false when stoppage omitted |
| **FED-05** | Built-in `FedAvg` | `fed_hyper_params.param` dict (may be `{}`) | custom aggregator kwargs | `{}` | — | — | Merged into aggregator `**kwargs` |
| **FED-06** | `n_classes` | Should match label space | — | — | Wrong metrics / head size | **Implicit** | Model + torchmetrics |

**Built-in vs plugin aggregator resolution:**

| `federated_model` value | Resolves to | `max_iter` enforcement (current code) |
|-------------------------|-------------|--------------------------------------|
| `FedAvg` | `utils/pytorch/optimizer.py` → class `FedAvg` | **Yes** — `aggregate()` increments `iteration`; flat `stoppage` |
| `FedAvg.py` | `mnt/input/FedAvg.py` → `CustomAggregator` | **Yes** (if plugin matches repo version) — same pattern as built-in after local fixes |
| Unknown / missing file | `None` | Crash at coordinator init |

---

## 1.6 `trainer.local_updates` — client upload schema

| Rule ID | Trigger | Required | Optional | Default | If violated | Enforcement | Code |
|---------|---------|----------|----------|---------|-------------|-------------|------|
| **LOC-01** | Always | `trainer.local_updates` exact `LocalUpdates` enum name | — | — | `AttributeError` | **Explicit** | `getattr(LocalUpdates, name)` |
| **LOC-02** | `FedAvg` / `FedAvg.py` weight averaging | `WEIGHTS_N_SAMPLES` (or variant including sample counts) | — | — | Broken averaging / wrong tuple unpack | **Explicit** / crash | `FedAvg.aggregate()` expects `(weights, n_samples)` per fold |
| **LOC-03** | `WEIGHTS` only (no `N_SAMPLES`) | — | Incompatible with `FedAvg` | — | Missing `n_samples` in client packet | **Explicit** | Same |
| **LOC-04** | Any `GRADIENTS*` local update | Aggregator + trainer must supply real gradients | — | — | `get_gradients()` is `pass` → ineffective updates | **Implicit** | `DeepModel.get_gradients()` |
| **LOC-05** | `*_CROSS_VALIDATION` local flags | **No extra CV segment is sent** by current client code | — | — | CV flag is effectively ignored on upload | **Implicit** | `ClientModels.get_local_updates()` never appends CV data |

---

## 1.7 `trainer` — trainer, optimizer, loss, metrics

| Rule ID | Trigger | Required | Optional | Default | If violated | Enforcement | Code |
|---------|---------|----------|----------|---------|-------------|-------------|------|
| **TRN-01** | Always | `trainer.name` resolvable (`BasicTrainer` or `*.py` → `CustomTrainer`) | `trainer.param` | `{}` | Trainer `None` | **Explicit** | `get_trainer()` |
| **TRN-02** | `trainer.name: FedMMb.py` | `train_config.batch_count` ≥ 1 meaningful | `train_config.epochs` ignored | — | Too-short or too-long local training vs intent | **Implicit** | `FedMMb.fit()` breaks on `batch_count` only |
| **TRN-03** | `BasicTrainer` | `train_config.epochs` controls local epochs per round | `batch_count` unused | — | — | — | `BasicTrainer.fit()` |
| **TRN-04** | Always | `trainer.optimizer.name` resolvable (`SGD`, `Adam`, … or plugin) | `optimizer.param` | `{}` | Crash at init | **Explicit** | `get_optimizer()` |
| **TRN-05** | Always | `trainer.optimizer.param` must include params required by optimizer (e.g. `lr` for `SGD`) | — | — | `TypeError` missing `lr` | **Explicit** | `torch.optim` constructors |
| **TRN-06** | Always | `trainer.loss.name` resolvable | `trainer.loss.param` | `{}` | Loss `None` | **Explicit** | `get_loss_func()` |
| **TRN-07** | `focal_loss.py` plugin | `trainer.loss.param.weight` — **sequence** of class weights (list/tuple) | `gamma` (default `2`) | `gamma=2`, `weight=None` | `TypeError: data must be a sequence (got NoneType)` | **Explicit** | `plugins/loss/focal_loss.py` line 12 |
| **TRN-08** | `focal_loss.py` | Do **not** use `alpha` key (not supported by plugin) | use `weight` instead | — | Unexpected kwargs or ignored keys | **Implicit** | Plugin `__init__(gamma=2, weight=None)` only |
| **TRN-09** | Classification metrics | ≥1 `trainer.metrics[]` entry with `name`, `package` | `param` per metric | — | Empty metrics list may break `Metrics` init | **Explicit** / edge | `get_custom_modules()` loop |
| **TRN-10** | `torchmetrics.classification` metrics | `param.task`, `param.num_classes` consistent with problem | — | — | torchmetrics runtime error | **Explicit** | Metric constructor |

---

## 1.8 `train_config`

| Rule ID | Trigger | Required | Optional | Default | If violated | Enforcement | Code |
|---------|---------|----------|----------|---------|-------------|-------------|------|
| **TCFG-01** | Always | Keys consumed via `setattr` on trainer: `verbose`, `batch_size`, `test_batch_size`, `epochs`, `lr`, `batch_count`, `device` | — | — | `AttributeError` if missing on trainer path | **Explicit** | `Trainer.__init__` |
| **TCFG-02** | `device: gpu` | CUDA available | — | Falls back CPU if unavailable | Slower / unexpected device | **Implicit** | `utils.set_device()` |
| **TCFG-03** | Dataloader batching | `batch_size`, `test_batch_size` passed to `dl.load(...)` | — | — | OOM or degenerate batches | **Explicit** (CUDA OOM) | `load_clients_data()` |

---

## 1.9 `model` — architecture and shape hints

| Rule ID | Trigger | Required | Optional | Default | If violated | Enforcement | Code |
|---------|---------|----------|----------|---------|-------------|-------------|------|
| **MOD-01** | `model.name` set (plugin or built-in) | Plugin: `mnt/input/{name}` with class `Model`; Built-in: name in `models.pytorch.models` (e.g. `CNN`) | `n_classes`, `in_features`, … per model | — | `None` architecture | **Explicit** | `design_architecture()` |
| **MOD-02** | `plugins/models/cnn.py` or built-in `CNN` | `n_classes`, `in_features` (used as `Conv2d` in_channels) | — | — | Shape mismatch on forward | **Explicit** | `cnn.py`, `models.py` |
| **MOD-03** | `plugins/models/mlp.py` | `n_classes`, `in_features` = flat feature size | — | — | Shape mismatch | **Explicit** | `mlp.py` |
| **MOD-04** | Layer-list `model` (no `name`) | List of `{type, param}` layers | `in_features`/`in_channels: None` auto-fill | — | Wrong graph | **Explicit** | `design_architecture()` without `name` |
| **MOD-05** | Image data | `in_features` = input channels (often `1` for MNIST) | — | — | Conv2d channel error | **Explicit** | CNN plugins |
| **MOD-06** | RNA tabular (`RnaLoader`) | Typically `mlp.py` not CNN | matching `in_features` | — | Conv on 1D tabular fails | **Explicit** | Architecture vs loader output shape |

---

## 1.10 `result` outputs

| Rule ID | Trigger | Required | Optional | Default | If violated | Enforcement | Code |
|---------|---------|----------|----------|---------|-------------|-------------|------|
| **RES-01** | WriteResults | `result.pred`, `result.target` string basenames | `result.model` | — | Missing outputs | **Implicit** | `finalize_config()` output paths |
| **RES-02** | `result.model` omitted | Model checkpoint writing skipped | — | — | No `model.pt` | **Implicit** | `write_dnn_models()` checks `result.get('model')` |

---

## 1.11 `use_smpc`

| Rule ID | Trigger | Required | Optional | Default | If violated | Enforcement | Code |
|---------|---------|----------|----------|---------|-------------|-------------|------|
| **SMPC-01** | `use_smpc: true` | Aggregator with working `aggregate_smpc` + client SMPC send path | — | `false` | Runtime failure / wrong weights | **Explicit** / bug | `send_data_for_aggregation(use_smpc=...)`, `FedAvg.aggregate_smpc` assigns `self.weights` (property conflict) |

---

## 1.12 Plugin file contract (all plugin types)

When a field uses `Something.py`, the file must be on **`mnt/input/`** (FeatureCloud generic dir) and expose:

| Plugin kind | Class name required | Resolved by |
|-------------|---------------------|-------------|
| Aggregator | `CustomAggregator` | `get_aggregator()` |
| Trainer | `CustomTrainer` | `get_trainer()` |
| Dataloader | `CustomDataLoader` | `get_dataloader()` |
| Loss | `CustomLoss` | `get_loss_func()` |
| Model | `Model` | `design_architecture()` |
| Optimizer | `CustomOptimizer` | `get_optimizer()` |
| Metric | `CustomMetric` | `get_metrics()` |

**Enforcement:** **Explicit** if file missing (`None`); **Explicit** `AttributeError` if class name wrong.

---

# Part 2 — Compatibility and incompatibility matrix

For each component: **valid pairings**, **invalid pairings**, **why**, **code origin**, **enforcement**.

---

## 2.1 Execution mode matrix

### If `fc_deep.simulation` is set (simulation mode)

| Component | Valid | Invalid | Why | Code | Enforcement |
|-----------|-------|---------|-----|------|-------------|
| Mode switch | `simulation.clients_dir: "c1,c2"` | Missing `clients_dir` | Cannot resolve virtual clients | `Simulation.correct_clients_dir()` | **Explicit** |
| vs `centralized` | Simulation only | `centralized: true` together | Simulation checked first | `Initialization.run()` | **Implicit** |
| vs federated FC | Single-container simulation | Multi-container paths without simulation block | Different state machine | `states.py` | **Structural** |
| `max_iter` | Communication rounds | Treating as epochs | Simulation loop uses rounds | `Simulation.run()` | **Implicit** |
| Data layout | Root `data/{fold}` + `{client}/data/{fold}` | Only root folds, no `c1`/`c2` trees | Simulated clients read injected paths | `inject_root_path_to_clients_dir()` | **Explicit** |

### If `fc_deep.centralized` is set (centralized mode)

| Component | Valid | Invalid | Why | Code | Enforcement |
|-----------|-------|---------|-----|------|-------------|
| Mode switch | `centralized: true` | `simulation` also set | Simulation wins | `Initialization.run()` | **Implicit** |
| Aggregation | Local training only | Expecting `Global Aggregation` rounds | `Centralized.run()` trains all folds sequentially | `Centralized` class | **Structural** |
| `max_iter` | Epochs per fold | Communication rounds | Overwrites `model.epochs` | `Centralized.run()` line 554 | **Implicit** |
| `federated_model` / `global_updates` | Present in YAML but unused in loop | — | No federated aggregation in centralized path | `Centralized.run()` | **Implicit** |

### If neither `simulation` nor `centralized` (default federated)

| Component | Valid | Invalid | Why | Code | Enforcement |
|-----------|-------|---------|-----|------|-------------|
| FC deployment | ≥2 clients, same fold count | 1 client | No peer to aggregate | FedAvg design | **Structural** |
| Coordinator | One client is coordinator | No coordinator | No aggregation state | `GlobalAggregation` coordinator-only | **Structural** |
| `max_iter` | Communication rounds | Epochs interpretation | Federated aggregator stopping | `post_aggregate()` | **Implicit** |

---

## 2.2 `fed_hyper_params.federated_model` × `global_updates`

### Built-in `FedAvg` (string `FedAvg`, no `.py`)

| `global_updates` | Valid? | Why | Code | Enforcement |
|------------------|--------|-----|------|-------------|
| `WEIGHTS` | Partial | Weights OK; **no stoppage** in packet | `get_global_updates()` | **Implicit** — may not stop |
| `WEIGHTS_STOPPING` | **Yes (recommended)** | Weights + stoppage; `max_iter` works | `FedAvg.post_aggregate()` | **Explicit** stoppage |
| `WEIGHTS_CONFIG*` | **No** | `FedAvg.config` is always `None` | `optimizer.py` properties | **Explicit** assert / bad unpack |
| `GRADIENTS*` | **No** | `FedAvg.gradients` is `None` | same | **Explicit** |
| `*_CROSS_VALIDATION` | **No / incomplete** | `get_global_updates()` never appends CV segment | `FedOptimizer.get_global_updates()` | **Implicit** — schema mismatch |

### Plugin `FedAvg.py` (`CustomAggregator`)

| Combination | Valid? | Why | Code | Enforcement |
|-------------|--------|-----|------|-------------|
| `WEIGHTS_STOPPING` + `WEIGHTS_N_SAMPLES` | **Yes** (current repo plugin) | Plugin mirrors built-in stopping after fixes | `plugins/aggregators/FedAvg.py` | **Explicit** when correct |
| `GRADIENTS*` | **No** | `gradients` property `None` | plugin file | **Explicit** |
| `*_CONFIG_*` | **No** | `config` property `None` | plugin file | **Explicit** |

---

## 2.3 `trainer.local_updates` × `fed_hyper_params.federated_model`

| `local_updates` | `FedAvg` / `FedAvg.py` | Gradient-based aggregator (if added) | Why | Code | Enforcement |
|-----------------|------------------------|--------------------------------------|-----|------|-------------|
| `WEIGHTS_N_SAMPLES` | **Valid** | N/A | Matches `(weights, n_samples)` | `FedAvg.aggregate()` | — |
| `WEIGHTS` | **Invalid** | — | No sample counts | same | **Explicit** |
| `GRADIENTS` / `GRADIENTS_N_SAMPLES` | **Invalid** | Needs working `get_gradients()` | `get_gradients` is `pass` | `DeepModel.py` | **Implicit** |
| `*_CROSS_VALIDATION` | Ignored on send | — | CV bit not implemented client-side | `ClientModels.get_local_updates()` | **Implicit** |

---

## 2.4 Dataset file format × `trainer.data_loader`

| Dataset files (`train` / `test` extension) | Compatible loader | Incompatible loader | Why | Code | Enforcement |
|---------------------------------------------|-------------------|---------------------|-----|------|-------------|
| `.npz` / `.npy` (image-like arrays) | `ImageLoader`, `ImageLoader.py` | `RnaLoader.py` | RnaLoader rejects non csv/tsv | `read_file()` format checks | **Explicit** (prints, returns false) |
| `.csv` / `.tsv` | `RnaLoader.py` + `detail.sep` + `detail.label` | `ImageLoader` | ImageLoader rejects non npz/npy | `ImageLoader.file_exits()` | **Explicit** |
| `.npz` keys `data`, `targets` | ImageLoader npz path | — | Expected NPZ schema | `ImageLoader.read_file()` | **Explicit** if keys missing |

### `local_dataset.detail` × dataloader

| Loader | `detail: {}` | `detail: {sep, label}` |
|--------|--------------|------------------------|
| `ImageLoader` | **Valid** | Extra keys passed but unused |
| `RnaLoader.py` | **Invalid** | **Required** |

---

## 2.5 Model × loss × output activation

| Model | Loss | Valid? | Why | Code | Enforcement |
|-------|------|--------|-----|------|-------------|
| `plugins/models/mlp.py` (raw logits) | `CrossEntropyLoss` | **Valid** | Standard multiclass | `mlp.py` forward | — |
| `plugins/models/cnn.py` (`log_softmax` output) | `CrossEntropyLoss` | **Invalid / wrong math** | CE expects raw logits; double softmax effect | `cnn.py` line 21 | **Implicit** |
| Built-in `CNN` (`log_softmax`) | `CrossEntropyLoss` | **Invalid / wrong math** | Same | `models/pytorch/models.py` | **Implicit** |
| `cnn.py` / `CNN` | `NLLLoss` | **Valid** | Matches `log_softmax` | pairing convention | — |
| `focal_loss.py` | expects `[N,C]` logits; applies internal `log_softmax` | `CrossEntropyLoss` alternative | Plugin design | `focal_loss.py` | — |
| Any classifier | `MSELoss` | Only if regression targets | `predict()` uses `max(1)` — classification assumption | `DeepModel.predict()` | **Implicit** |

**Task assumption (all default trainers):** integer class labels, multiclass `predict()` via `pred.max(1)[1]` — **classification only** (`DeepModel.predict()`, `BasicTrainer`).

---

## 2.6 Model × dataloader (input shape)

| Data source | Compatible models | Incompatible models | Why | Code |
|-------------|-------------------|---------------------|-----|------|
| `ImageLoader` (2D/3D image tensors) | `cnn.py`, built-in `CNN` | `mlp.py` without flatten redesign | Conv expects image topology | `FromNumpyDataset` + CNN |
| `RnaLoader` (1D feature vectors) | `mlp.py` | `cnn.py`, `CNN` | Conv2d on flat RNA features | shape mismatch at forward |
| Auto layer-list | Sample-driven `in_features` / `in_channels` | Wrong manual dims | Auto-fill only when `None` string | `get_param_value_from_data()` |

---

## 2.7 Trainer × `train_config`

| Trainer | Honors `epochs` | Honors `batch_count` | Invalid assumption |
|---------|-----------------|----------------------|--------------------|
| `BasicTrainer` | **Yes** | No (ignored) | Using `batch_count` to limit BasicTrainer |
| `FedMMb.py` | **No** | **Yes** | Using `epochs` to control FedMMb local training |

---

## 2.8 Class count consistency (`n_classes`)

These should match for a valid classification config:

| Field | Must align with |
|-------|-----------------|
| `fed_hyper_params.n_classes` | Label space |
| `model.n_classes` | Output head size |
| `trainer.metrics[].param.num_classes` | torchmetrics task |
| `focal_loss.param.weight` length (if used) | Number of classes |

**Enforcement:** mix of **Explicit** (shape errors) and **Implicit** (misleading accuracy).

---

## 2.9 `central_test` × aggregator × evaluation

| `central_test` | `federated_model` | Global agg eval metrics | Stoppage / side effects |
|----------------|-------------------|-------------------------|-------------------------|
| `null` | `FedAvg` | `evaluate_aggregated_models` returns `[]`; built-in FedAvg no longer uses `len(metrics)` for stoppage | OK with current built-in/plugin FedAvg |
| set | `FedAvg` | Coordinator `test_loader` runs central metrics | Central CSV written in `WriteResults` |
| `null` | Custom aggregator using **old** `len(metrics)` stoppage pattern | Empty metrics → empty stoppage | **Invalid** — training may not stop (**Implicit**) |

---

## 2.10 `use_smpc` compatibility

| Setting | Aggregator | Valid? | Why | Code |
|---------|------------|--------|-----|------|
| `use_smpc: true` | Built-in `FedAvg` | **Broken / risky** | `aggregate_smpc` assigns read-only `weights` property | `optimizer.py` |
| `use_smpc: true` | `FedAvg.py` plugin | Same risk | `self.weights = ...` in plugin | `plugins/aggregators/FedAvg.py` |
| `use_smpc: false` | Any weight-based FedAvg | **Valid** | Standard path | — |

---

## 2.11 Enum string exactness (`global_updates`, `local_updates`)

| Value in YAML | Requirement |
|---------------|-------------|
| Any `GlobalUpdates` / `LocalUpdates` name | Must match enum identifier **exactly** (case-sensitive) |
| Typo e.g. `WEIGHT_STOPPING` | `AttributeError` at `load_schemas()` |

**Full lists:** `utils/pytorch/utils.py` classes `GlobalUpdates`, `LocalUpdates`.

**Enforcement:** **Explicit**.

---

## 2.12 Plugin dataloader caveat (`ImageLoader.py` plugin vs built-in)

| Loader | Multi-fold `load(path)` per fold | Issue |
|--------|----------------------------------|-------|
| Built-in `ImageLoader` (`utils/pytorch/DataLoader.py`) | Updates `self.path = path` | **Valid** per-fold loading |
| Plugin `ImageLoader.py` | Does **not** assign `path` to `self.path` in `load()` | Re-reads wrong file for folds ≠ first | **Implicit** bug |

**Code:** `plugins/dataloaders/ImageLoader.py` `load()` vs built-in fix at `DataLoader.py` line 29.

---

## 2.13 FeatureCloud deployment × path conventions (config executor)

Not app-runtime YAML validation, but affects whether staged config reaches containers:

| Setting | Valid FeatureCloud CLI value | Invalid example | Why |
|---------|------------------------------|-----------------|-----|
| `--generic-dir` | `tools/config` (relative to mounted `data/`) | `data/tools/config` | Double `data/` prefix → empty generic dir |
| `--client-dirs` | `sample_data/c1,sample_data/c2` | `data/sample_data/c1` | Resolves to `data/data/...` |

**Enforcement:** **Explicit** `FileNotFoundError: mnt/input/config.yml` in clients.

---

## 2.14 `global_updates` packet protocol (cross-cutting)

Whatever flags are enabled must match **both sides**:

```
Coordinator: FedOptimizer.get_global_updates()  → broadcast list
Clients:     interpret_global_updates() + unpack()
Aggregator:  properties weights / gradients / config / stoppage
```

| Mismatch | Result | Enforcement |
|----------|--------|-------------|
| Schema expects 2 elements, list has 1 | Assertion error | **Explicit** — `interpret_global_updates()` |
| Stoppage nested list `[[True]]` vs flat `[True]` | `all(stoppage)` wrong | **Implicit** — early or failed convergence |
| Empty stoppage `[]` | `all_converged` false; `if stoppage` false in LocalUpdate | **Implicit** — non-terminating federated loop |

**Code:** `utils/utils.py` (`unpack`, `interpret_global_updates`), `utils/pytorch/states.py` (`all_converged`, `LocalUpdate.run`).

---

# Quick reference — invalid combination table

| A | B | Issue | § |
|---|----|-------|---|
| `simulation` set | `centralized` set | Simulation branch wins | 2.1 |
| `global_updates` with **gradients** | built-in **`FedAvg`** | `gradients` not provided | 2.2 |
| `global_updates` with **config** | built-in **`FedAvg`** | `config` is `None` | 2.2 |
| `local_updates: WEIGHTS` | **`FedAvg`** | Missing `n_samples` | 2.3 |
| `local_updates: GRADIENTS*` | **`FedAvg`** + `DeepModel` | Gradients not implemented | 2.3 |
| `WEIGHTS` only (no STOPPING) | need **`max_iter`** stop | Stoppage absent | 1.5 FED-04 |
| **`RnaLoader`** | **`detail: {}`** | **KeyError** | 1.3 DATA-03/04 |
| **`RnaLoader`** | **`.npz` files** | Format rejected | 2.4 |
| **`ImageLoader`** | **`.csv` files** | Format rejected | 2.4 |
| **`focal_loss.py`** | **`param: {}`** or `alpha:` only | **`weight` sequence required** | 1.7 TRN-07 |
| **`cnn.py` / `CNN`** | **`CrossEntropyLoss`** | log_softmax + CE mismatch | 2.5 |
| **`use_smpc: true`** | **`FedAvg` SMPC path** | Property assignment bug | 2.10 |
| **`init_model`** coordinator-only | other clients random init | Inconsistent starting weights | 1.3 DATA-08 |
| Different **#folds** per client | **`directory`** mode | Aggregation breaks | 1.4 LOGIC-03 |
| **`FedMMb.py`** | rely on **`epochs`** | Uses **`batch_count`** only | 2.7 |
| Plugin **`ImageLoader.py`** | multiple CV folds | Wrong path on fold >0 | 2.12 |
| `*_CROSS_VALIDATION` enums | current client upload | CV data never sent | 1.6 LOC-05 |

---

# Scenario recipes (valid minimal pairings)

| Goal | Mode | Aggregator | `global_updates` | `local_updates` | Loader | Loss | Model |
|------|------|------------|------------------|-----------------|--------|------|-------|
| Standard MNIST FL | Federated | `FedAvg` | `WEIGHTS_STOPPING` | `WEIGHTS_N_SAMPLES` | `ImageLoader` | `CrossEntropyLoss` | `mlp.py` or logits-based model |
| MNIST FL + plugin CNN | Federated | `FedAvg` | `WEIGHTS_STOPPING` | `WEIGHTS_N_SAMPLES` | `ImageLoader` | `NLLLoss` | `cnn.py` |
| RNA tabular FL | Federated | `FedAvg` | `WEIGHTS_STOPPING` | `WEIGHTS_N_SAMPLES` | `RnaLoader.py` + `detail` | `CrossEntropyLoss` | `mlp.py` |
| Imbalanced classes | Federated | `FedAvg` | `WEIGHTS_STOPPING` | `WEIGHTS_N_SAMPLES` | matching loader | `focal_loss.py` + `weight: [...]` | logits model |
| Local CV training | Centralized | (unused) | (present but unused) | `WEIGHTS_N_SAMPLES` | `ImageLoader` | `CrossEntropyLoss` | `mlp.py` |
| Debug FL in one box | Simulation | `FedAvg` | `WEIGHTS_STOPPING` | `WEIGHTS_N_SAMPLES` | `ImageLoader` | `CrossEntropyLoss` | `mlp.py` |

---

# Maintenance note for config-generation agents

When extending this app or DSPy pipelines:

1. Treat **`GlobalUpdates` / `LocalUpdates` / `get_global_updates()` / `unpack()` / `FedAvg.aggregate()`** as one protocol — YAML flags must match what both coordinator and clients send.
2. Distinguish **built-in names** (`FedAvg`, `BasicTrainer`, `ImageLoader`, `CNN`) from **plugin filenames** (`FedAvg.py`, `cnn.py`, …) — resolution path differs (`utils/utils.py` `get_custom_module`).
3. Validate **file extensions** against **dataloader** before validating model architecture.
4. Validate **loss params** against the **actual plugin constructor**, not generic examples (e.g. `focal_loss` uses `weight`, not `alpha`).
5. Remember **`max_iter` semantics change** with execution mode (rounds vs epochs).
6. For FeatureCloud staging, generic-dir paths are **relative to `data/` mount**, not repo root (see §2.13).

---

*Document version: expanded from original `CONFIG_INCOMPATIBILITIES.md` after full-repo code review. Plugin `FedAvg.py` stopping behavior reflects current file in `plugins/aggregators/FedAvg.py` (iteration increment + flat stoppage).*
