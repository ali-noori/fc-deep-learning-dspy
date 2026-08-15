# Configuration Parameter Reference

User-facing lookup for every important field under `fc_deep` in `config.yml`.

- **This document:** what each parameter means, which values are valid, and what changes when you pick a value.
- **Not this document:** incompatible combinations → see [`CONFIG_COMPATIBILITY_GUIDE.md`](CONFIG_COMPATIBILITY_GUIDE.md) and [`CONFIG_INCOMPATIBILITIES.md`](CONFIG_INCOMPATIBILITIES.md).

Values and behavior below come from runtime code (`CustomStates/ConfigState.py`, `utils/pytorch/states.py`, `utils/utils.py`, trainers/loaders/models/plugins), not from parameter names alone.

---

## How to use this reference

1. Find the YAML path (for example `trainer.name`).
2. Read **Allowed values**, **What each value changes**, and **Dependencies**.
3. Check **Execution-mode relevance** so you know whether the field matters in your run.

### Mode selection (affects many fields)

Checked in `Initialization.run()`:

1. If `fc_deep.simulation` is present (not `null`) → **simulation**
2. Else if `fc_deep.centralized` is present (not `null`) → **centralized**
3. Else → **federated**

### Built-in vs plugin naming (shared rule)

Many fields accept either:

| Form | Meaning | Class the file must define |
|------|---------|----------------------------|
| Plain name (`BasicTrainer`, `FedAvg`, `CNN`, `CrossEntropyLoss`) | Built-in attribute on a Python module | — |
| `Something.py` | File under container `mnt/input/` (generic dir) | See table below |

| Component | Required class inside `.py` |
|-----------|----------------------------|
| Trainer | `CustomTrainer` |
| Aggregator | `CustomAggregator` |
| Data loader | `CustomDataLoader` |
| Loss | `CustomLoss` |
| Optimizer | `CustomOptimizer` |
| Model | `Model` |
| Metric | `CustomMetric` (if not imported from a package) |

---

## Quick index

| Path | Section |
|------|---------|
| `debug` | [§1](#1-debug) |
| `logic.mode` / `logic.dir` | [§2](#2-logic) |
| `local_dataset.*` | [§3](#3-local_dataset) |
| `simulation.clients_dir` | [§4](#4-simulation) |
| `centralized` | [§5](#5-centralized) |
| `result.*` | [§6](#6-result) |
| `fed_hyper_params.*` | [§7](#7-fed_hyper_params) |
| `use_smpc` | [§8](#8-use_smpc) |
| `trainer.*` | [§9](#9-trainer) |
| `train_config.*` | [§10](#10-train_config) |
| `model.*` | [§11](#11-model) |

---

## 1. `debug`

**Path:** `fc_deep.debug`

**What it does:** Turns on extra FeatureCloud/app debug logging during config load.

**Allowed values:** `true` | `false` (boolean)

| Value | Effect |
|-------|--------|
| `true` | Stores shared `debug=True` and logs that debug mode is on |
| `false` / omitted | No debug flag (or stores `false` if key present) |

**Default:** omitted (treated as off)

**Required:** Optional

**Dependencies:** None

**Execution-mode relevance:** Common to all modes

```yaml
fc_deep:
  debug: true
```

---

## 2. `logic`

Controls how training **splits / folds** are discovered under the container input directory.

### 2.1 `logic.mode`

**Path:** `fc_deep.logic.mode`

**What it does:** Chooses whether each subdirectory under `logic.dir` is a CV fold, or the whole input is one split.

**Allowed values (code):**

| Value | Effect |
|-------|--------|
| `"directory"` | Scans `{input}/{logic.dir}/` for **subdirectories**; each becomes a split |
| `"file"` | Uses a **single** split: the input directory itself |

**Default in code:** if the whole `logic` section is missing → `mode: "file"`, `dir: "."`  
**Common sample default:** `"directory"` (`config.minimum.yml`)

**Required:** Optional as a section (defaults apply); if `logic` exists, `mode` and `dir` are expected

**Dependencies:** Works with `logic.dir`; drives how many loaders / models are created per client

**Execution-mode relevance:** Common (all modes use splits)

```yaml
logic:
  mode: "directory"   # e.g. data/1, data/2, ...
  dir: "data"
```

```yaml
logic:
  mode: "file"        # one split = whole mnt/input
  dir: "."            # ignored for split discovery when mode=file
```

### 2.2 `logic.dir`

**Path:** `fc_deep.logic.dir`

**What it does:** Folder name under `mnt/input` that contains fold directories when `mode` is `directory`.

**Allowed values:** Any relative folder name that exists under input (e.g. `"data"`)

**Default:** `"."` if `logic` omitted

**Required:** Expected when `logic` is present; meaningful mainly for `mode: directory`

**Dependencies:** `logic.mode`

**Execution-mode relevance:** Common

---

## 3. `local_dataset`

Filenames and loader options for data under each split.

> **Important:** Every key under `local_dataset` is turned into paths `{split}/{value}` — including `detail`. Keep `detail` as a **dict** of loader kwargs (samples use `{}`), not as a filename.

### 3.1 `local_dataset.train`

**Path:** `fc_deep.local_dataset.train`

**What it does:** Filename of the training file inside each fold directory.

**Allowed values:** Filename string supported by the chosen data loader  
- Image loaders: `.npz`, `.npy`  
- `RnaLoader.py`: `.csv`, `.tsv`

**Default:** None in code (must be provided)

**Required:** Yes

**Dependencies:** `trainer.data_loader`, `logic.*` (where files live)

**Execution-mode relevance:** Common  
In **simulation**, each client folder must also contain the same relative layout.

```yaml
local_dataset:
  train: "train.npz"
```

### 3.2 `local_dataset.test`

**Path:** `fc_deep.local_dataset.test`

**What it does:** Filename of the local test/validation file per fold.

**Allowed values:** Filename string, or a value that loaders treat as missing (e.g. `"None"` path handling in ImageLoader)

**Required:** Key is expected; at least one of `test` or `central_test` must provide usable test data, or init logs `"no test data"`

**Dependencies:** `trainer.data_loader`

**Execution-mode relevance:** Common

### 3.3 `local_dataset.central_test`

**Path:** `fc_deep.local_dataset.central_test`

**What it does:** Optional shared/central evaluation file (coordinator / simulation central eval).

**Allowed values:** `null` | filename string (e.g. `"mnist.npz"`)

| Value | Effect |
|-------|--------|
| `null` / omitted | No central test loader |
| `"some.npz"` | Coordinator (and simulation) can evaluate / write central predictions |

**Default:** `null` in samples

**Required:** Optional

**Dependencies:** `trainer.data_loader`, `train_config.test_batch_size`

**Execution-mode relevance:** Common field; used on coordinator in federated; also used in simulation central-eval path

### 3.4 `local_dataset.detail`

**Path:** `fc_deep.local_dataset.detail`

**What it does:** Extra keyword arguments passed into the data-loader constructor:  
`dl_class(sample_path, **detail)`.

**Allowed values:** Mapping (dict). Contents depend on loader:

| Loader | Typical `detail` | Required keys |
|--------|------------------|---------------|
| Built-in `ImageLoader` | `{}` | Prefer empty — non-empty keys cause `TypeError` (`__init__(path=None)` only) |
| `ImageLoader.py` | `{}` or extra ignored kwargs | None |
| `RnaLoader.py` | `{sep: ",", label: "Group"}` | **`sep`**, **`label`** |

**Default:** samples use `{}`

**Required:** Key expected in normal configs; content optional except RnaLoader

**Dependencies:** `trainer.data_loader`

**Execution-mode relevance:** Common

```yaml
# Image path
local_dataset:
  detail: {}

# RNA path
local_dataset:
  detail:
    sep: ","
    label: "Group"
```

### 3.5 `local_dataset.init_model`

**Path:** `fc_deep.local_dataset.init_model`

**What it does:** Optional filename of a pretrained model the **coordinator** loads before federated rounds.

**Allowed values:** `null` | model filename present under input splits

**Default:** omitted / null

**Required:** Optional

**Dependencies:** Trainer `load_model` path; only applied when coordinator finds the key

**Execution-mode relevance:** Federated coordinator warm-start (not used on the early simulation return path in `Initialization`)

---

## 4. `simulation`

### 4.1 `simulation` (section presence)

**Path:** `fc_deep.simulation`

**What it does:** Presence of this key (even empty) switches the app into **simulation** mode.

**Allowed values:** Mapping with at least `clients_dir`, or any non-`null` value that makes `.get('simulation')` non-None

**Required:** Required for simulation; must be **absent/`null`** for federated and centralized

**Execution-mode relevance:** Simulation only (wins over `centralized` if both set)

### 4.2 `simulation.clients_dir`

**Path:** `fc_deep.simulation.clients_dir`

**What it does:** Comma-separated list of client folder names under the simulation data root. Defines how many simulated clients exist and where their data trees live.

**Allowed values:** String like `"c1,c2"` (no spaces in the sample convention)

| Example | Effect |
|---------|--------|
| `"c1,c2"` | Two clients; app injects paths under each client dir |
| `"c1,c2,c3"` | Three clients |

**Default:** None (must be set for simulation)

**Required:** Yes when running simulation

**Dependencies:** `local_dataset.train` / `test` filenames; each client must have matching fold layout

**Execution-mode relevance:** Simulation only

```yaml
fc_deep:
  simulation:
    clients_dir: "c1,c2"
```

---

## 5. `centralized`

**Path:** `fc_deep.centralized`

**What it does:** Presence of this key (any non-`null` value) selects **centralized** training instead of the federated LocalUpdate ↔ GlobalAggregation loop.

**Allowed values:** Any non-null YAML value (samples use `true`). Even `{}` is treated as “set”.

| Value | Effect |
|-------|--------|
| absent / `null` | Not centralized |
| `true` / `{}` / other non-null | Enter `Centralized` state |

**Default:** absent

**Required:** Required for centralized mode; omit for federated/simulation

**Dependencies:** Must not compete with `simulation` (simulation is checked first)

**Execution-mode relevance:** Centralized only

```yaml
fc_deep:
  centralized: true
  # do not set simulation
```

**Note:** In centralized mode, `fed_hyper_params.max_iter` is copied onto the trainer as **`epochs`** for `BasicTrainer`. Aggregator fields are still present in templates but the federated aggregation loop does not run.

---

## 6. `result`

Output filenames written under each split’s output directory.

### 6.1 `result.pred`

**Path:** `fc_deep.result.pred`

**What it does:** Basename for the prediction CSV written after training.

**Allowed values:** Filename string (e.g. `"y_pred.csv"`)

**Required:** Yes

**Execution-mode relevance:** Common

### 6.2 `result.target`

**Path:** `fc_deep.result.target`

**What it does:** Basename for the ground-truth CSV written alongside predictions.

**Allowed values:** Filename string (e.g. `"y_test.csv"`)

**Required:** Yes

**Execution-mode relevance:** Common

### 6.3 `result.model`

**Path:** `fc_deep.result.model`

**What it does:** Optional basename for saving the trained model after convergence.

| Value | Effect |
|-------|--------|
| omitted / `null` | Model file not saved via this path |
| `"model.pt"` | Calls `store_model` per fold |

**Required:** Optional

**Execution-mode relevance:** Common

```yaml
result:
  pred: "y_pred.csv"
  target: "y_test.csv"
  model: "model.pt"   # optional
```

---

## 7. `fed_hyper_params`

Federated hyperparameters. Still present in centralized/simulation templates; meaning of `max_iter` changes by mode.

### 7.1 `fed_hyper_params.max_iter`

**Path:** `fc_deep.fed_hyper_params.max_iter`

**What it does:** Limits how long the outer training loop runs — but the **unit depends on mode**.

| Mode | Meaning |
|------|---------|
| Federated | Max **communication rounds**; with `*_STOPPING` global updates, stoppage becomes true when `iteration >= max_iter` |
| Simulation | Max **simulated rounds** (`for c_round in range(max_iter)`) |
| Centralized | Assigned to `client_model.model.epochs` → for **`BasicTrainer`**, number of local epochs per fold |

**Allowed values:** Positive integer

**Default in samples:** `10`

**Required:** Yes (read directly; missing → KeyError)

**Dependencies:** Stopping needs a `global_updates` schema that includes STOPPING (e.g. `WEIGHTS_STOPPING`). For centralized + `FedMMb.py`, local length still follows `train_config.batch_count`, not this epoch override for FedMMb’s loop.

**Execution-mode relevance:** Common key; semantics mode-specific

```yaml
fed_hyper_params:
  max_iter: 10   # 10 FL rounds, or 10 centralized epochs for BasicTrainer
```

### 7.2 `fed_hyper_params.n_classes`

**Path:** `fc_deep.fed_hyper_params.n_classes`

**What it does:** Declared number of classes in the hyper-parameter block. The whole `fed_hyper_params` dict is passed into the aggregator constructor, but **built-in `FedAvg` does not read `n_classes`**.

**Allowed values:** Positive integer (task-dependent)

**Default in samples:** `10`

**Required:** Present in templates; **not enforced against** `model.n_classes` at runtime

**Dependencies:** Should match `model.n_classes` and metric `num_classes` for consistency (application does not auto-sync)

**Execution-mode relevance:** Common (documentation / aggregator kwargs); model size comes from `model.n_classes`

### 7.3 `fed_hyper_params.federated_model`

**Path:** `fc_deep.fed_hyper_params.federated_model`

**What it does:** Selects the **aggregator** class used on the coordinator (and in simulation aggregation).

**Allowed values (verified in this repo):**

| Value | Type | Effect |
|-------|------|--------|
| `FedAvg` | Built-in (`utils/pytorch/optimizer.py`) | Sample-weighted average of client weights; unpacks each fold as `(weights, n_samples)` |
| `FedAvg.py` | Plugin (`plugins/aggregators/FedAvg.py`, class `CustomAggregator`) | Same algorithm as a staged plugin file |
| Other `*.py` | Your custom aggregator on `mnt/input` | Must expose `CustomAggregator` |

**Default in samples:** `"FedAvg"`

**Required:** Yes for coordinator aggregation path

**Dependencies:** Must match what clients send (`trainer.local_updates`) and what is broadcast (`global_updates`)

**Execution-mode relevance:** Federated + simulation (built in centralized init path but unused by `Centralized.run`)

```yaml
fed_hyper_params:
  federated_model: "FedAvg"
```

### 7.4 `fed_hyper_params.global_updates`

**Path:** `fc_deep.fed_hyper_params.global_updates`

**What it does:** Enum name controlling which pieces the coordinator **broadcasts** after aggregation.

**Allowed values** (exact names from `GlobalUpdates` in `utils/pytorch/utils.py`):

Flags order: `(WEIGHTS, GRADIENTS, CONFIG, STOPPING, CROSS_VALIDATION)`

| Value | WEIGHTS | GRADIENTS | CONFIG | STOPPING | CROSS_VALIDATION |
|-------|---------|-----------|--------|----------|------------------|
| `WEIGHTS` | ✓ | | | | |
| `GRADIENTS` | | ✓ | | | |
| `WEIGHTS_CONFIG` | ✓ | | ✓ | | |
| `GRADIENTS_CONFIG` | | ✓ | ✓ | | |
| `WEIGHTS_CROSS_VALIDATION` | ✓ | | | | ✓ |
| `GRADIENTS_CROSS_VALIDATION` | | ✓ | | | ✓ |
| `WEIGHTS_CONFIG_CROSS_VALIDATION` | ✓ | | ✓ | | ✓ |
| `GRADIENTS_CONFIG_CROSS_VALIDATION` | | ✓ | ✓ | | ✓ |
| `WEIGHTS_STOPPING` | ✓ | | | ✓ | |
| `GRADIENTS_STOPPING` | | ✓ | | ✓ | |
| `WEIGHTS_CONFIG_STOPPING` | ✓ | | ✓ | ✓ | |
| `GRADIENTS_CONFIG_STOPPING` | | ✓ | ✓ | ✓ | |
| `WEIGHTS_STOPPING_CROSS_VALIDATION` | ✓ | | | ✓ | ✓ |
| `GRADIENTS_STOPPING_CROSS_VALIDATION` | | ✓ | | ✓ | ✓ |
| `WEIGHTS_CONFIG_STOPPING_CROSS_VALIDATION` | ✓ | | ✓ | ✓ | ✓ |
| `GRADIENTS_CONFIG_STOPPING_CROSS_VALIDATION` | | ✓ | ✓ | ✓ | ✓ |

**What each flag changes:**

| Flag | Effect when true |
|------|------------------|
| WEIGHTS | Broadcast aggregated model weights |
| GRADIENTS | Include gradients in broadcast (FedAvg currently exposes `None`) |
| CONFIG | Include aggregator “config” payload (FedAvg: `None`) |
| STOPPING | Include stoppage flags so clients/coordinator can terminate when `max_iter` reached |
| CROSS_VALIDATION | CV-related packaging bit (limited use in current upload path) |

**Default in samples:** `WEIGHTS_STOPPING`

**Required:** Yes (invalid name → `AttributeError` via `getattr`)

**Dependencies:** Pair with `trainer.local_updates` and aggregator capabilities

**Execution-mode relevance:** Federated + simulation (centralized does not run the FL broadcast loop)

```yaml
fed_hyper_params:
  global_updates: WEIGHTS_STOPPING
```

### 7.5 `fed_hyper_params.param`

**Path:** `fc_deep.fed_hyper_params.param`

**What it does:** Extra constructor kwargs for custom aggregators (merged into the hyper-param dict passed to the aggregator).

**Allowed values:** Mapping (usually `{}`)

**Default:** `{}` in samples

**Required:** Optional

**Execution-mode relevance:** Common in templates; useful mainly for custom aggregators

---

## 8. `use_smpc`

**Path:** `fc_deep.use_smpc`

**What it does:** Enables FeatureCloud SMPC packaging for sending/gathering local updates (weights × sample counts on the wire).

**Allowed values:** `true` | `false`

| Value | Effect |
|-------|--------|
| `false` | Normal send/gather |
| `true` | SMPC path in local send / coordinator gather |

**Default:** `false` (`.get('use_smpc', False)`)

**Required:** Optional

**Dependencies:** Current FedAvg `aggregate_smpc` has a known property conflict (`self.weights` vs read-only `@property`) — treat as advanced / risky unless fixed

**Execution-mode relevance:** Federated + simulation communication; stored always; no multi-client aggregation in centralized

```yaml
use_smpc: false
```

---

## 9. `trainer`

### 9.1 `trainer.name`

**Path:** `fc_deep.trainer.name`

**What it does:** Selects the **local training loop** implementation (how one client/fold updates the model each round or run).

**Allowed values (this repository):**

| Value | Type | Effect |
|-------|------|--------|
| `BasicTrainer` | Built-in | Runs `for e in range(self.epochs)` over the full train loader each epoch; sets `n_trained_samples = len(train_loader)` (**batch count**) |
| `FedMMb.py` | Plugin | Trains at most `batch_count` batches; **ignores `epochs`** in its `fit()`; returns a sample count but does **not** store it on `self.n_trained_samples` (stays 0) |
| Other `*.py` | Custom | Must define `CustomTrainer` on `mnt/input` |

> Note: The valid built-in name is **`BasicTrainer`**, not `basic_trainer`.

**Default in samples:** `"BasicTrainer"`

**Required:** Yes

**Dependencies:**

- `BasicTrainer` → uses `train_config.epochs`
- `FedMMb.py` → uses `train_config.batch_count`
- Both use optimizer/loss/model from the same trainer section

**Execution-mode relevance:** Common (all modes build a trainer)

```yaml
trainer:
  name: "BasicTrainer"   # full-epoch local training

trainer:
  name: "FedMMb.py"      # stop after batch_count batches per local step
```

### 9.2 `trainer.param`

**Path:** `fc_deep.trainer.param`

**What it does:** Extra keyword arguments passed into the trainer constructor (`**param`).

**Allowed values:** Mapping (usually `{}`)

**Default:** `{}` via `.get('param', {})`

**Required:** Optional

**Execution-mode relevance:** Common

### 9.3 `trainer.local_updates`

**Path:** `fc_deep.trainer.local_updates`

**What it does:** Enum name controlling what each client **uploads** after local training.

**Allowed values** (`LocalUpdates` in `utils/pytorch/utils.py`):

Flags order: `(WEIGHTS, GRADIENTS, N_SAMPLES, CROSS_VALIDATION)`

| Value | WEIGHTS | GRADIENTS | N_SAMPLES | CROSS_VALIDATION |
|-------|---------|-----------|-----------|------------------|
| `WEIGHTS` | ✓ | | | |
| `GRADIENTS` | | ✓ | | |
| `WEIGHTS_N_SAMPLES` | ✓ | | ✓ | |
| `GRADIENTS_N_SAMPLES` | | ✓ | ✓ | |
| `WEIGHTS_CROSS_VALIDATION` | ✓ | | | ✓ |
| `GRADIENTS_CROSS_VALIDATION` | | ✓ | | ✓ |
| `WEIGHTS_N_SAMPLES_CROSS_VALIDATION` | ✓ | | ✓ | ✓ |
| `GRADIENTS_N_SAMPLES_CROSS_VALIDATION` | | ✓ | ✓ | ✓ |
| `ALL` | ✓ | ✓ | ✓ | ✓ |

**What each flag changes:**

| Flag | Effect when true |
|------|------------------|
| WEIGHTS | Append model weights to the upload list |
| GRADIENTS | Append gradients (`get_gradients()` is currently `pass` → effectively `None`) |
| N_SAMPLES | Append `model.n_trained_samples` (needed by FedAvg’s `(weights, n_samples)` unpack) |
| CROSS_VALIDATION | Unpacked from schema but **not appended** in current `get_local_updates` |

**Default in samples:** `WEIGHTS_N_SAMPLES`

**Required:** Yes (invalid name → crash)

**Dependencies:** Must match aggregator expectations (FedAvg needs sample counts with weights)

**Execution-mode relevance:** Federated + simulation uploads; less meaningful in pure centralized (no multi-client aggregate)

```yaml
trainer:
  local_updates: WEIGHTS_N_SAMPLES
```

### 9.4 `trainer.data_loader`

**Path:** `fc_deep.trainer.data_loader`

**What it does:** Selects how files are read into PyTorch `DataLoader`s that yield `(data, target)` batches.

**Allowed values (this repository):**

| Value | Type | Formats | Effect |
|-------|------|---------|--------|
| `ImageLoader` | Built-in | `.npz`, `.npy` | Loads arrays; normalizes; expects image-like tensors. Constructor accepts only `path` |
| `ImageLoader.py` | Plugin | `.npz`, `.npy` | Similar; accepts `**kwargs`. Plugin `load(path=...)` does **not** update `self.path` (multi-fold risk) |
| `RnaLoader.py` | Plugin | `.csv`, `.tsv` | Tabular features + label column; needs `detail.sep` and `detail.label` |
| Other `*.py` | Custom | Your formats | Must define `CustomDataLoader` with `sample_data_loader` and `load` |

**Default in samples:** often `"ImageLoader"` or `"ImageLoader.py"`

**Required:** Yes

**Dependencies:** `local_dataset.train` / `test` / `detail`; model input shape (`in_features`)

**Execution-mode relevance:** Common

```yaml
trainer:
  data_loader: "ImageLoader"

trainer:
  data_loader: "RnaLoader.py"
# with local_dataset.detail.sep / label
```

### 9.5 `trainer.optimizer.name`

**Path:** `fc_deep.trainer.optimizer.name`

**What it does:** Selects the PyTorch optimizer class used for local steps.

**Allowed values:**

1. Any attribute of `torch.optim` that is an optimizer class. In the current environment these names resolve:

   `ASGD`, `Adadelta`, `Adafactor`, `Adagrad`, `Adam`, `AdamW`, `Adamax`, `LBFGS`, `Muon`, `NAdam`, `Optimizer`, `RAdam`, `RMSprop`, `Rprop`, `SGD`, `SparseAdam`

2. Or a custom `*.py` exposing `CustomOptimizer` on `mnt/input`

| Common choice | Effect |
|---------------|--------|
| `SGD` | Stochastic gradient descent; typically needs `param.lr` |
| `Adam` / `AdamW` | Adaptive methods; still need their constructor args (e.g. `lr`) |

**Default in samples:** `"SGD"`

**Required:** Yes

**Dependencies:** `trainer.optimizer.param` (constructor kwargs). Model parameters are injected automatically as `params=`.

**Execution-mode relevance:** Common

```yaml
trainer:
  optimizer:
    name: "SGD"
    param:
      lr: 0.1
```

### 9.6 `trainer.optimizer.param`

**Path:** `fc_deep.trainer.optimizer.param`

**What it does:** Keyword arguments for the optimizer constructor (except `params`, which the app adds).

**Allowed values:** Mapping accepted by the chosen optimizer (e.g. `lr`, `momentum`, `weight_decay`)

**Default:** `{}` via `.get`, but many optimizers **require** `lr`

**Required:** Optional key; practically required fields depend on optimizer

**Dependencies:** `trainer.optimizer.name`

**Execution-mode relevance:** Common

> `train_config.lr` is **also** copied onto the trainer as an attribute, but the optimizer is built from **`trainer.optimizer.param`**, not from `train_config.lr`.

### 9.7 `trainer.loss.name`

**Path:** `fc_deep.trainer.loss.name`

**What it does:** Selects the loss module used in `train_on_batch` / evaluation: `loss_func(pred, target)`.

**Allowed values:**

1. Any `torch.nn` loss class name (current env, names ending with `Loss`):

   `AdaptiveLogSoftmaxWithLoss`, `BCELoss`, `BCEWithLogitsLoss`, `CTCLoss`, `CosineEmbeddingLoss`, `CrossEntropyLoss`, `GaussianNLLLoss`, `HingeEmbeddingLoss`, `HuberLoss`, `KLDivLoss`, `L1Loss`, `LinearCrossEntropyLoss`, `MSELoss`, `MarginRankingLoss`, `MultiLabelMarginLoss`, `MultiLabelSoftMarginLoss`, `MultiMarginLoss`, `NLLLoss`, `PoissonNLLLoss`, `SmoothL1Loss`, `SoftMarginLoss`, `TripletMarginLoss`, `TripletMarginWithDistanceLoss`

2. Plugin: `focal_loss.py` (class `CustomLoss`) — applies its own `log_softmax`, then NLL; **requires** `param.weight` sequence

| Common value | Expects model output | Typical pairing in this app |
|--------------|----------------------|-----------------------------|
| `NLLLoss` | Log-probabilities | `CNN` / `cnn.py` (`log_softmax`) |
| `CrossEntropyLoss` | Raw logits | `mlp.py` |
| `MSELoss` | Same shape as targets | Regression-style setups |
| `BCEWithLogitsLoss` | Logits | Multi-label / binary setups |
| `focal_loss.py` | Logits | `mlp.py` + `param.weight: [...]` |

**Default in samples:** often `"CrossEntropyLoss"` or `"NLLLoss"` depending on model

**Required:** Yes

**Dependencies:** Model forward output; `trainer.loss.param`

**Execution-mode relevance:** Common

```yaml
# With CNN / cnn.py
trainer:
  loss:
    name: "NLLLoss"
    param: {}

# With mlp.py
trainer:
  loss:
    name: "CrossEntropyLoss"
    param: {}
```

### 9.8 `trainer.loss.param`

**Path:** `fc_deep.trainer.loss.param`

**What it does:** Keyword arguments for the loss constructor.

**Allowed values:** Mapping accepted by the loss class  
Example for focal loss: `{weight: [1.0, ...], gamma: 2}`

**Default:** `{}`

**Required:** Optional for most built-ins; **required content** for `focal_loss.py` (`weight`)

**Dependencies:** `trainer.loss.name`

**Execution-mode relevance:** Common

### 9.9 `trainer.metrics` (list)

**Path:** `fc_deep.trainer.metrics`

**What it does:** List of metric definitions used for logging/evaluation (not for the training step loss).

Each list item:

| Sub-key | Path | Required | Meaning |
|---------|------|----------|---------|
| `name` | `trainer.metrics[].name` | Yes | Metric class name on the package (e.g. `Accuracy`) |
| `package` | `trainer.metrics[].package` | Yes | Import path (e.g. `torchmetrics.classification`) |
| `param` | `trainer.metrics[].param` | Optional | Passed to metric constructor |

**Allowed values for `name`:** Any class available on the imported `package` (or a custom metric module pattern). Common torchmetrics examples used in samples: `Accuracy`, `F1Score`, `AUROC`.

**What changing them does:** Changes which scores appear in logs / history; does **not** change the loss used for backprop.

**Default in samples:**

```yaml
metrics:
  - name: "Accuracy"
    package: "torchmetrics.classification"
    param:
      task: "multiclass"
      num_classes: 10
```

**Required:** Section expected; list may be empty in theory but samples always define metrics

**Dependencies:** `param.num_classes` / `task` should match the classification setup and `model.n_classes`

**Execution-mode relevance:** Common

---

## 10. `train_config`

All keys under `train_config` are copied onto the trainer object via `setattr` (`DeepModel.Trainer.__init__`). Unknown keys become attributes; known ones drive training.

### 10.1 `train_config.verbose`

**Path:** `fc_deep.train_config.verbose`

**What it does:** Controls whether per-epoch / validation metric logs are printed.

**Allowed values:** `true` | `false`

**Default in samples:** `false`

**Required:** Expected in templates

**Execution-mode relevance:** Common

### 10.2 `train_config.batch_size`

**Path:** `fc_deep.train_config.batch_size`

**What it does:** Batch size for **training** loaders (`dl.load(..., batch_size=...)`).

**Allowed values:** Positive integer

**Default in samples:** `32`

**Required:** Yes in practice

**Dependencies:** Affects number of steps per epoch and (for BasicTrainer) `len(train_loader)` uploaded as “samples”

**Execution-mode relevance:** Common

### 10.3 `train_config.test_batch_size`

**Path:** `fc_deep.train_config.test_batch_size`

**What it does:** Batch size for **test / validation / central_test** loaders.

**Allowed values:** Positive integer

**Default in samples:** `32`

**Required:** Yes in practice

**Execution-mode relevance:** Common

### 10.4 `train_config.epochs`

**Path:** `fc_deep.train_config.epochs`

**What it does:** Local epochs per training call for **`BasicTrainer`**.

| Trainer | Effect of `epochs` |
|---------|-------------------|
| `BasicTrainer` | Outer loop `for e in range(self.epochs)` |
| `FedMMb.py` | **Ignored** by `fit()` |

In **centralized** mode, `fed_hyper_params.max_iter` **overwrites** `client_model.model.epochs` before training.

**Allowed values:** Positive integer

**Default in samples:** `1` (one local epoch per federated round is common)

**Required:** Expected for BasicTrainer

**Dependencies:** `trainer.name`; centralized `max_iter`

**Execution-mode relevance:** Common; meaning interacts with mode

```yaml
train_config:
  epochs: 1   # federated: 1 local epoch each round
```

### 10.5 `train_config.lr`

**Path:** `fc_deep.train_config.lr`

**What it does:** Stored as trainer attribute `self.lr`. **Does not** automatically configure the optimizer; optimizer learning rate comes from `trainer.optimizer.param.lr`.

**Allowed values:** Float

**Default in samples:** `0.1`

**Required:** Optional for BasicTrainer logic; included in templates

**Execution-mode relevance:** Common (attribute only unless a custom trainer reads it)

### 10.6 `train_config.batch_count`

**Path:** `fc_deep.train_config.batch_count`

**What it does:** For **`FedMMb.py`**, stops local training after this many batches (`if i == self.batch_count - 1: break`). Unused by `BasicTrainer.fit()`.

**Allowed values:** Positive integer

**Default in samples:** `1`

**Required:** Needed when using FedMMb

**Dependencies:** `trainer.name: FedMMb.py`

**Execution-mode relevance:** Common key; effective mainly with FedMMb

### 10.7 `train_config.device`

**Path:** `fc_deep.train_config.device`

**What it does:** Selects compute device; normalized by `set_device()`.

**Allowed values:**

| Value | Effect |
|-------|--------|
| `cpu` | Force CPU |
| `gpu` | Use CUDA **if available**, otherwise CPU |

**Default in samples:** `cpu`

**Required:** Yes (read and rewritten in place during init)

**Execution-mode relevance:** Common

```yaml
train_config:
  device: cpu
# or
  device: gpu
```

---

## 11. `model`

Two shapes are supported.

### Shape A — Named model (usual)

```yaml
model:
  name: "CNN"      # or "cnn.py" / "mlp.py"
  n_classes: 10
  in_features: 1
```

### Shape B — Layer list (no `name`)

```yaml
model:
  - type: Flatten
    param: {}
  - type: Linear
    param:
      in_features: "None"   # auto from sample batch
      out_features: 10
```

Each item needs `type` = a `torch.nn` module class name. Optional `param` mapping. Special string `"None"` for `in_features` / `in_channels` triggers inference from a sample batch.

---

### 11.1 `model.name`

**Path:** `fc_deep.model.name`

**What it does:** Selects which neural network class is constructed.

**Allowed values (this repository):**

| Value | Type | Constructor args used from config | Forward output | Effect |
|-------|------|-----------------------------------|----------------|--------|
| `CNN` | Built-in | `n_classes`, `in_features` (channels) | `log_softmax` | Small conv net for image-like tensors |
| `cnn.py` | Plugin `Model` | same | `log_softmax` | Plugin CNN (same idea as built-in) |
| `mlp.py` | Plugin `Model` | `n_classes`, `in_features` (flat size) | **raw logits** | MLP for flat / tabular vectors |
| Other `*.py` | Custom `Model` | whatever your `__init__` accepts | your design | User architecture on `mnt/input` |

If `name` is **absent**, Shape B (layer list) is used instead.

**Default in samples:** `"cnn.py"` or `"CNN"` depending on sample

**Required:** Either `name` **or** a layer-list model section

**Dependencies:** Loss choice should match output (`NLLLoss` vs `CrossEntropyLoss`); `in_features` must match loader tensors

**Execution-mode relevance:** Common

```yaml
model:
  name: "CNN"
  n_classes: 10
  in_features: 1      # channels for CNN

model:
  name: "mlp.py"
  n_classes: 10
  in_features: 784    # flat feature size
```

### 11.2 `model.n_classes`

**Path:** `fc_deep.model.n_classes`

**What it does:** Number of output classes for named CNN/MLP models (final linear layer size).

**Allowed values:** Positive integer matching the label space

**Required:** Yes for named CNN/`cnn.py`/`mlp.py`

**Dependencies:** Should match labels, metrics `num_classes`, and (conventionally) `fed_hyper_params.n_classes`

**Execution-mode relevance:** Common

### 11.3 `model.in_features`

**Path:** `fc_deep.model.in_features`

**What it does:** Input size for the named model.

| Model | Meaning of `in_features` |
|-------|--------------------------|
| `CNN` / `cnn.py` | Number of **input channels** (e.g. `1` for grayscale) |
| `mlp.py` | Flat **feature dimension** (e.g. `784` for 28×28 flattened) |

**Required:** Yes for those named models

**Dependencies:** Data loader tensor shape

**Execution-mode relevance:** Common

---

## 12. Cross-parameter cheat sheet

| If you change… | Also check… |
|----------------|-------------|
| `trainer.name` | `train_config.epochs` vs `batch_count` |
| `trainer.data_loader` | file extensions + `local_dataset.detail` |
| `model.name` | `trainer.loss.name` (log-softmax vs logits) |
| `model.n_classes` | metrics `num_classes`, labels, `fed_hyper_params.n_classes` |
| `fed_hyper_params.federated_model` | `local_updates` + `global_updates` |
| `fed_hyper_params.max_iter` | mode meaning (rounds vs epochs) + STOPPING schema |
| `simulation` / `centralized` | omit the other; simulation wins if both set |
| `logic.mode` | folder layout under `logic.dir` |

---

## 13. Minimal annotated example (federated)

```yaml
fc_deep:
  # omit simulation / centralized → federated

  local_dataset:
    train: "train.npz"
    test: "test.npz"
    central_test: null
    detail: {}

  logic:
    mode: "directory"
    dir: "data"

  result:
    pred: "y_pred.csv"
    target: "y_test.csv"

  fed_hyper_params:
    max_iter: 10                 # communication rounds
    n_classes: 10                # not enforced vs model; keep aligned
    federated_model: "FedAvg"
    global_updates: WEIGHTS_STOPPING
    param: {}

  use_smpc: false

  trainer:
    name: "BasicTrainer"         # epoch-based local training
    param: {}
    local_updates: WEIGHTS_N_SAMPLES
    data_loader: "ImageLoader"
    optimizer:
      name: "SGD"
      param:
        lr: 0.1
    loss:
      name: "NLLLoss"            # matches CNN log_softmax
      param: {}
    metrics:
      - name: "Accuracy"
        package: "torchmetrics.classification"
        param:
          task: "multiclass"
          num_classes: 10

  train_config:
    verbose: false
    batch_size: 32
    test_batch_size: 32
    epochs: 1                    # local epochs per round
    lr: 0.1                      # attribute only; optimizer uses optimizer.param.lr
    batch_count: 1               # used by FedMMb, not BasicTrainer
    device: cpu

  model:
    name: "CNN"
    n_classes: 10
    in_features: 1
```

---

## Related documents

| Document | Purpose |
|----------|---------|
| [`CONFIG_PARAMETER_REFERENCE.md`](CONFIG_PARAMETER_REFERENCE.md) (this file) | What each parameter means and which values do what |
| [`CONFIG_COMPATIBILITY_GUIDE.md`](CONFIG_COMPATIBILITY_GUIDE.md) | Which combinations are safe together |
| [`CONFIG_INCOMPATIBILITIES.md`](CONFIG_INCOMPATIBILITIES.md) | Technical incompatibility rule IDs and code citations |
| `config.minimum.yml` / `config.maximum.yml` | Commented sample configs |
