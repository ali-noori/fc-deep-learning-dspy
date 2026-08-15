# Configuration Compatibility Guide (User-Friendly)

This document is a **practical companion** to [`CONFIG_INCOMPATIBILITIES.md`](CONFIG_INCOMPATIBILITIES.md).

- **Use this file** when you are writing or reviewing a config before training.
- **Use `CONFIG_INCOMPATIBILITIES.md`** when you need the full technical rule IDs and code citations.

Rules below were checked against the runtime implementation (`utils/`, `plugins/`, `CustomStates/`, `states.py`).  
Where the code does **not** hard-block a combination, that is stated explicitly (it may still fail at runtime for shape/math reasons).

---

## 1. How to read this guide

For each component you choose:

1. Check the **execution mode** section first.
2. Check **must-match fields** (class counts, update schemas).
3. Check **component pairing** tables (trainer / loader / loss / model / aggregator).
4. Check **paths & files** before running FeatureCloud.

Legend:

| Label | Meaning |
|-------|---------|
| **Required** | Missing / wrong value → crash or clear error |
| **Recommended** | Code may run, but results are wrong or training may not stop |
| **Not coupled in code** | No `isinstance` / name lock; mix is allowed; failure is shape/math/schema |

---

## 2. Must-match fields across the config

These values are **not auto-synced** by the app. You must set them consistently yourself.

| Field A | Field B | Rule | Why |
|---------|---------|------|-----|
| `fed_hyper_params.n_classes` | `model.n_classes` | **Should be equal** (recommended) | Model head size comes only from `model.n_classes`. `fed_hyper_params.n_classes` is passed into the aggregator kwargs but **FedAvg never reads it**. The app **does not compare** these two fields at runtime. |
| `model.n_classes` | `trainer.metrics[].param.num_classes` | **Should be equal** (for multiclass metrics) | Metrics are built only from YAML `param`. Wrong `num_classes` → torchmetrics error or wrong scores. **Not auto-synced** with the model. |
| `local_dataset.train` / `test` | Files on disk under each fold | **Must exist** | Paths become `{fold}/{filename}` via `logic`. |
| `trainer.local_updates` | Aggregator expectations | **Must be compatible** | See §5. |
| `fed_hyper_params.global_updates` | Aggregator stoppage / packet shape | **Must be compatible** | See §5. |

### Quick checklist for class count

If you train 10-class MNIST:

```yaml
fed_hyper_params:
  n_classes: 10
model:
  n_classes: 10
trainer:
  metrics:
    - name: Accuracy
      param:
        task: multiclass
        num_classes: 10   # match model
```

---

## 3. Execution mode rules

Mode is decided in `Initialization.run()`:

1. If `fc_deep.simulation` is set → **simulation**
2. Else set up training modules…
3. If `fc_deep.centralized` is set → **centralized**
4. Else → **federated**

### 3.1 Federated

| Item | Valid | Invalid / risky |
|------|-------|-----------------|
| Mode flags | Do **not** set `simulation` | `simulation` present → you leave federated |
| Clients | ≥2 FeatureCloud `--client-dirs` with **same fold layout** | Different fold counts → aggregation crash |
| `max_iter` | Communication **rounds** | Treating it as local epochs (that is centralized) |
| Config location | `config.yml` in generic dir | Missing → `FileNotFoundError` |

### 3.2 Simulation

| Item | Valid | Invalid / risky |
|------|-------|-----------------|
| Required | `simulation.clients_dir` like `"c1,c2"` | Missing/empty → bad client paths |
| Layout | Root folds **and** `{client}/data/{fold}/…` | Missing client trees → load failures |
| `centralized` | Prefer absent | If both set, **simulation wins**; centralized ignored |
| `max_iter` | Communication **rounds** | — |

### 3.3 Centralized

| Item | Valid | Invalid / risky |
|------|-------|-----------------|
| Required | `centralized: true` (or truthy) | `simulation` also set → never reaches centralized |
| Role | Coordinator-only training state | Expecting participant containers to train in this state |
| `max_iter` | Used as **`epochs` for BasicTrainer** (per fold) | With **FedMMb.py**, local length still follows `train_config.batch_count`, not epochs |

---

## 4. Built-in vs plugin naming (all components)

Resolution (`get_custom_module`):

| You write | Meaning |
|-----------|---------|
| `CNN`, `BasicTrainer`, `ImageLoader`, `FedAvg`, `CrossEntropyLoss` | Built-in (attribute on a library module) |
| `cnn.py`, `FedMMb.py`, `ImageLoader.py`, `focal_loss.py` | Plugin file on **`mnt/input/`** (basename after config_executor staging) |
| `plugins/models/cnn.py` in builder output | OK for generation; executor rewrites to `cnn.py` and copies the file |

**Important:** After FeatureCloud staging, plugins must exist as files next to `config.yml` under the generic mount. A value like `plugins/models/cnn.py` that is **not** rewritten/copied will fail with “module not found”.

---

## 5. Aggregator ↔ update schemas

### 5.1 Recommended safe pair (current FedAvg)

| Field | Recommended value |
|-------|-------------------|
| `fed_hyper_params.federated_model` | `FedAvg` or staged `FedAvg.py` |
| `fed_hyper_params.global_updates` | `WEIGHTS_STOPPING` |
| `trainer.local_updates` | `WEIGHTS_N_SAMPLES` |

**Why:**

- `FedAvg.aggregate()` unpacks each client fold as `(weights, n_samples)`.
- Without `N_SAMPLES` in `local_updates` (e.g. plain `WEIGHTS`) → unpack / averaging failure.
- Without `STOPPING` in `global_updates` → stoppage flags omitted → training may **not stop** on `max_iter` (`all_converged` needs stoppage).

### 5.2 Aggregator compatibility table

| Aggregator | Compatible `local_updates` | Incompatible / risky `local_updates` | Compatible `global_updates` | Risky `global_updates` |
|------------|----------------------------|--------------------------------------|-----------------------------|------------------------|
| `FedAvg` / `FedAvg.py` | `WEIGHTS_N_SAMPLES`, `WEIGHTS_N_SAMPLES_CROSS_VALIDATION`, `ALL` | `WEIGHTS` (no sample counts) | `WEIGHTS_STOPPING` (+ CV variants with STOPPING) | Schemas needing live `gradients`/`config` (`GRADIENTS*`, `*_CONFIG*`): FedAvg properties are `None` |
| Unknown name / missing `.py` | — | — | — | Init crash (“module not found”) |

**Note:** Cross-validation bits in enum names are largely **ignored on upload** today (`ClientModels.get_local_updates` does not send CV payloads). Prefer non-CV names for clarity.

### 5.3 Sample counts sent to FedAvg (implementation caveats)

Client upload uses `self.model.n_trained_samples` when `local_updates` includes `N_SAMPLES`.

| Trainer | What is uploaded as `n_samples` | Consequence |
|---------|----------------------------------|-------------|
| `BasicTrainer` | `len(train_loader)` after `fit` | This is the **number of batches**, not the number of examples. Averaging is still non-zero and often usable, but not true sample-weighted FL. |
| `FedMMb.py` | Stays **`0`** | `fit()` accumulates a local `n_trained_samples` and **returns** it, but never assigns `self.n_trained_samples`. FedAvg then divides by total sample counts that can be zero / wrong. Prefer **BasicTrainer** with FedAvg unless you patch FedMMb. |

### 5.4 Schemas that look valid but break with FedAvg

| Schema choice | Problem | Why (code) |
|---------------|---------|------------|
| `local_updates: WEIGHTS` | Unpack / averaging failure | FedAvg expects `(weights, n_samples)` per fold |
| `local_updates` / `global_updates` including `GRADIENTS` | Gradients are always `None` | `Trainer.get_gradients()` is `pass` |
| `global_updates` with `*_CONFIG*` | Config payload is `None` | FedAvg `config` property returns `None` |
| Invalid enum string (typo) | Crash at init | `getattr(LocalUpdates/GlobalUpdates, name)` |
| `use_smpc: true` with current FedAvg | Property assignment conflict | `aggregate_smpc` assigns `self.weights` but `weights` is a read-only `@property` |

---

## 6. Trainer compatibility

### 6.1 Trainers available

| Trainer | Type | What controls local training length |
|---------|------|-------------------------------------|
| `BasicTrainer` | Built-in | `train_config.epochs` (full pass over loader each epoch) |
| `FedMMb.py` | Plugin | `train_config.batch_count` (stops after N batches); **`epochs` ignored** |

### 6.2 Trainer × data loader

| Trainer | Compatible data loaders | Incompatible loaders | Reason |
|---------|-------------------------|----------------------|--------|
| `BasicTrainer` | `ImageLoader`, `ImageLoader.py`, `RnaLoader.py` (if files/`detail` OK) | None by name | Code only needs batches as `(data, target)` tensors |
| `FedMMb.py` | Same | None by name | Same batch API (`train_on_batch`) |

There is **no hard trainer↔loader lock** in code. Failures come from tensor shape vs model, or loader file/`detail` errors.

### 6.3 Trainer × `train_config`

| Trainer | Required / used | Ignored / unused | If wrong |
|---------|-----------------|------------------|----------|
| `BasicTrainer` | `epochs` ≥ 1 meaningful | `batch_count` unused by `fit` | Too few/many local epochs |
| `FedMMb.py` | `batch_count` ≥ 1 | `epochs` unused by `fit` | Local step stops early/late vs intent |
| Both | `optimizer.param.lr` (for SGD) | — | `TypeError` missing `lr` |

### 6.4 Trainer × loss / model

**Not name-coupled.** Any resolvable loss/model can be plugged in. See §7–§8 for shape/math constraints.

---

## 7. Data loader compatibility

### 7.1 Loaders

| Loader | Type | File formats | `local_dataset.detail` |
|--------|------|--------------|------------------------|
| `ImageLoader` | Built-in | `.npz`, `.npy` | Prefer `{}`. Built-in ctor is `__init__(path=None)` → **non-empty detail kwargs can TypeError** |
| `ImageLoader.py` | Plugin | `.npz`, `.npy` | Extra keys accepted (`**kwargs`) and ignored |
| `RnaLoader.py` | Plugin | `.csv`, `.tsv` | **Required:** `sep`, `label` |

### 7.2 Loader × model (practical)

| Data loader | Fits well with | Poor fit | Reason |
|-------------|----------------|----------|--------|
| `ImageLoader` / `ImageLoader.py` | `CNN` / `cnn.py` (`in_features` = channels, e.g. 1) | `mlp.py` unless you flatten outside | CNN expects image tensors `(N,C,H,W)` |
| `RnaLoader.py` | `mlp.py` with `in_features` = feature dimension | `CNN` / `cnn.py` | RNA loader yields **1D feature vectors**, not image maps |

### 7.3 Loader × loss

**Not name-coupled.** Labels must be integer class indices for CE / NLL / focal.

### 7.4 Example: RnaLoader required detail

```yaml
local_dataset:
  train: train.csv
  test: test.csv
  detail:
    sep: ","
    label: my_label_column
trainer:
  data_loader: RnaLoader.py   # or staged basename
```

Missing `sep`/`label` → `KeyError` at loader init.

---

## 8. Model × loss compatibility

### 8.1 Models

| Model | Type | Output of `forward` | `in_features` means |
|-------|------|---------------------|---------------------|
| `CNN` | Built-in | `log_softmax` | Input **channels** |
| `cnn.py` | Plugin | `log_softmax` | Input **channels** |
| `mlp.py` | Plugin | **Raw logits** | Flat input size (e.g. 784) |

### 8.2 Loss choices

| Loss | Type | Expects model output | Extra params |
|------|------|----------------------|--------------|
| `NLLLoss` | Built-in | Log-probabilities | — |
| `CrossEntropyLoss` | Built-in | **Logits** (applies log-softmax internally) | — |
| `BCEWithLogitsLoss` | Built-in | Logits (typically multi-label / special setups) | — |
| `MSELoss` | Built-in | Same shape as targets | — |
| `focal_loss.py` | Plugin | **Logits** (plugin applies `log_softmax` then NLL) | **`param.weight` must be a sequence** (list/tuple of class weights). `gamma` optional (default 2). No `alpha`. |

### 8.3 Model × loss matrix

| Model | Recommended losses | Avoid / risky | Reason (from code) |
|-------|--------------------|---------------|--------------------|
| `CNN` / `cnn.py` | `NLLLoss` | `CrossEntropyLoss`, `focal_loss.py` | Model already returns `F.log_softmax(...)`. CE/focal apply another softmax/log-softmax → wrong training signal. **Not blocked by code.** |
| `mlp.py` | `CrossEntropyLoss`, `focal_loss.py` (with `weight`) | Bare `NLLLoss` | MLP returns logits; NLL expects log-probs. |

### 8.4 Focal loss required shape

```yaml
trainer:
  loss:
    name: focal_loss.py   # after staging: basename on mnt/input
    param:
      weight: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]  # length = n_classes
      gamma: 2
```

`weight: null` / omitted → `TypeError` when building `FloatTensor(weight)`.

---

## 9. Component “shopping lists” (copy-friendly)

### Trainer: `BasicTrainer`

| | |
|--|--|
| Compatible data loaders | `ImageLoader`, `ImageLoader.py`, `RnaLoader.py` (with valid files/`detail`) |
| Incompatible by name | None |
| Compatible losses | Any resolvable loss (watch model×loss math in §8) |
| Needs from `train_config` | `epochs` |
| Notes | With FedAvg + `WEIGHTS_N_SAMPLES`, uploaded count = **batch count** (`len(train_loader)`), not raw sample count |

### Trainer: `FedMMb.py`

| | |
|--|--|
| Compatible data loaders | Same as BasicTrainer (batch API identical) |
| Incompatible by name | None |
| Needs from `train_config` | `batch_count` |
| Notes | With FedAvg + `WEIGHTS_N_SAMPLES`, uploaded `n_samples` stays **0** (implementation gap) |

### Data loader: `ImageLoader` (built-in)

| | |
|--|--|
| Compatible models | `CNN`, `cnn.py` |
| Risky models | `mlp.py` without flattening |
| Files | `.npz` / `.npy` under each fold |
| `detail` | Prefer `{}` |

### Data loader: `ImageLoader.py`

| | |
|--|--|
| Formats | Same as built-in (`.npz` / `.npy`) |
| `detail` | Extra kwargs accepted |
| Important bug vs built-in | Plugin `load(path=...)` does **not** assign `self.path = path` (built-in does). Multi-fold training can keep reading the first sample path. Prefer built-in `ImageLoader` unless you only use one file / patch the plugin. |

### Data loader: `RnaLoader.py`

| | |
|--|--|
| Compatible models | `mlp.py` (match `in_features`) |
| Risky models | `CNN` / `cnn.py` |
| Required `detail` | `sep`, `label` |
| Files | `.csv` / `.tsv` |

### Loss: `CrossEntropyLoss`

| | |
|--|--|
| Best with | `mlp.py` logits |
| Risky with | `CNN` / `cnn.py` (already log-softmax) |

### Loss: `NLLLoss`

| | |
|--|--|
| Best with | `CNN` / `cnn.py` |
| Risky with | `mlp.py` logits |

### Loss: `focal_loss.py`

| | |
|--|--|
| Required | `param.weight` sequence length = `#classes` |
| Best with | logit models (`mlp.py`) |
| Risky with | `CNN` / `cnn.py` (double log-softmax) |

---

## 10. Paths & FeatureCloud mounting (before executor)

| Setting | Valid example | Invalid example | Why |
|---------|---------------|-----------------|-----|
| `--client-dirs` | `sample_data/c1,sample_data/c2` | `D:/.../data/siteA` or `data/sample_data/c1` | Paths are relative to mounted host `data/`; absolute/wrong prefix breaks mounts |
| `--generic-dir` | `tools/config` | `data/tools/config` | Double `data/` → empty generic dir / missing `config.yml` |
| `logic.mode: directory` | fold dirs under `mnt/input/{logic.dir}/` | empty `logic.dir` | Zero splits → empty training |
| Plugin fields | staged basenames on generic dir | `plugins/.../*.py` without copy | `get_custom_module` looks under `mnt/input/` |

Dataset filenames in config (`train.npz`) are **not** prefixed with `data/` — `logic.dir: data` already places folds under `data/{fold}/train.npz`.

---

## 11. Safe starter combinations (verified patterns)

### A. Classic federated MNIST (recommended)

```text
Mode: federated (no simulation / centralized)
Model: CNN or cnn.py
Loader: ImageLoader
Loss: NLLLoss          # because CNN outputs log_softmax
Trainer: BasicTrainer
Aggregator: FedAvg
local_updates: WEIGHTS_N_SAMPLES
global_updates: WEIGHTS_STOPPING
n_classes: 10 everywhere (model + metrics)
Files: train.npz / test.npz
```

### B. MLP + CrossEntropy (tabular / flat features)

```text
Model: mlp.py
Loader: ImageLoader only if inputs are already flat vectors of size in_features
     or RnaLoader.py for csv/tsv with detail.sep/label
Loss: CrossEntropyLoss
Trainer: BasicTrainer
```

### C. Simulation

```text
simulation.clients_dir: "c1,c2"
Same FL hyper-params as federated
Data layout: root folds + per-client fold trees
```

### D. Centralized

```text
centralized: true
no simulation block
fed_hyper_params.max_iter = epochs for BasicTrainer
(aggregation schemas are unused in this mode, but the block is still read for max_iter)
```

---

## 12. Plugin file naming contract

When you point a field at a `.py` plugin, the file must define a **fixed class name**:

| Component | Config field example | Required class inside the `.py` file |
|-----------|----------------------|--------------------------------------|
| Trainer | `trainer.name: FedMMb.py` | `CustomTrainer` |
| Data loader | `trainer.data_loader: RnaLoader.py` | `CustomDataLoader` |
| Loss | `trainer.loss.name: focal_loss.py` | `CustomLoss` |
| Aggregator | `fed_hyper_params.federated_model: FedAvg.py` | `CustomAggregator` |
| Model | `model.name: cnn.py` | `Model` |

Wrong class name → module loads but the expected class is missing → “module not found” / init failure.

---

## 13. “Looks valid in YAML” but fails later — short list

| Config looks OK | Fails because |
|-----------------|---------------|
| `CrossEntropyLoss` + `CNN`/`cnn.py` | Double log-softmax math |
| `focal_loss.py` without `weight` list | `FloatTensor(None)` TypeError |
| `local_updates: WEIGHTS` + `FedAvg` | Missing `n_samples` in packet |
| `global_updates` without `STOPPING` | May never stop on `max_iter` |
| `simulation` + `centralized` both set | Centralized never runs |
| `RnaLoader.py` with `detail: {}` | Missing `sep`/`label` |
| Built-in `ImageLoader` + non-empty `detail` | `__init__(path=None)` rejects extra kwargs |
| `ImageLoader` + `.csv` files | Format not npz/npy |
| `plugins/models/cnn.py` not staged to basename | Module not found on `mnt/input` |
| Absolute Windows paths in `--client-dirs` | FeatureCloud expects paths under mounted `data/` |
| `FedMMb.py` + FedAvg | Runs, but uploaded sample counts stay 0 |
| Mismatched `metrics.num_classes` vs model | torchmetrics / wrong scores |
| `GRADIENTS*` update schemas | `get_gradients()` is unimplemented (`pass`) |
| `ImageLoader.py` for multi-fold CV | Plugin `load()` does not update `self.path`; may reread first file |

---

## 14. Relationship to `CONFIG_INCOMPATIBILITIES.md`

| This guide | That file |
|------------|-----------|
| User-facing “what can I combine?” | Engineer-facing rule IDs (ENV-01, FED-04, …) |
| Focus on practical matrices | Focus on exhaustive enforcement tags |
| Keep both | Do not delete either |

If you find a new failure mode while training, add it here **and** (optionally) as a new ID in `CONFIG_INCOMPATIBILITIES.md`.
