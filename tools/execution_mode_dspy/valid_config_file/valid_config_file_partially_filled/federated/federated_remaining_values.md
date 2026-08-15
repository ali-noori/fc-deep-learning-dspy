# Federated mode — remaining config fields (valid values)

This document explains what you can put in each **empty** field in  
`federated.template.partially.yml`.

The app resolves names through `utils.get_*()` helpers in `utils/utils.py`.  
Most plugin-related fields accept either:

- a **built-in** name (class inside the repo or PyTorch), or  
- a **custom** `.py` file mounted at `mnt/input/<filename>.py` with the expected class name.

---

## Data paths (user-specific — no fixed list)

These depend on your folder layout and chosen dataloader. The app does **not** read paths relative to `config.yml` in git; it builds paths under **`mnt/input/`** inside each client container.

### `local_dataset.train`

- **Valid values:** any **filename** (not a full path) inside each split folder.
- **How it is used:** with `logic.mode: "directory"` and `logic.dir`, the app opens  
  `mnt/input/<dir>/<split>/<train>` for each split (e.g. `0`, `1`).
- **Constraints:** must match the file type your dataloader supports (see `trainer.data_loader` below).

**Examples (MNIST-style):** `train.npz`, `train.npy`

---

### `local_dataset.test`

- Same rules as `local_dataset.train`, but for the test file in each split folder.
- **Examples:** `test.npz`, `test.npy`

---

### `logic.dir`

- **Valid values:** name of the folder under `mnt/input/` that contains split subfolders.
- **How it is used:** the app scans `mnt/input/<dir>/` and treats each subfolder (e.g. `0`, `1`) as one fold.
- **Examples:** `data` (used in sample_data: `c1/data/0/train.npz`, etc.)

**Federated note:** each client container mounts its own data (`sample_data/c1` or `c2`). The same `dir` name should exist on every client mount with the same split structure.

---

## `fed_hyper_params.n_classes`

- **Valid values:** any **positive integer** (number of classes in your labels).
- **Not** a plugin name — discovered from your dataset, not from `plugins/`.
- **Should match:** `model.n_classes` and `trainer.metrics[].param.num_classes` when you fill those.

**Example:** `10` (MNIST)

---

## `fed_hyper_params.federated_model` (aggregator)

Resolved by `utils.get_aggregator()` → built-in `utils.pytorch.optimizer` or custom file with class **`CustomAggregator`**.

### Built-in (PyTorch / repo)

| Value   | Source                          |
|---------|----------------------------------|
| `FedAvg` | `utils/pytorch/optimizer.py`     |

### Custom (plugin `.py` on `mnt/input/`)

| Value        | Repo file (copy or mount to input)   | Required class      |
|--------------|----------------------------------------|---------------------|
| `FedAvg.py`  | `plugins/aggregators/FedAvg.py`        | `CustomAggregator`  |

Any other `<name>.py` you add under `plugins/aggregators/` (or mount into `mnt/input/`) works if it defines `CustomAggregator`.

---

## `trainer.name`

Resolved by `utils.get_trainer()` → built-in `utils.pytorch.DeepModel` or custom file with class **`CustomTrainer`**.

### Built-in

| Value          | Source                         |
|----------------|---------------------------------|
| `BasicTrainer` | `utils/pytorch/DeepModel.py`    |

### Custom (plugin `.py` on `mnt/input/`)

| Value       | Repo file                      | Required class   |
|-------------|--------------------------------|------------------|
| `FedMMb.py` | `plugins/trainers/FedMMb.py`   | `CustomTrainer`  |

---

## `trainer.data_loader`

Resolved by `utils.get_dataloader()` → built-in `utils.pytorch.DataLoader` or custom file with class **`CustomDataLoader`**.

### Built-in

| Value         | Source                          | Typical file formats |
|---------------|----------------------------------|----------------------|
| `ImageLoader` | `utils/pytorch/DataLoader.py`    | `.npz`, `.npy`       |

### Custom (plugin `.py` on `mnt/input/`)

| Value            | Repo file                               | Required class      | Typical file formats |
|------------------|-----------------------------------------|---------------------|----------------------|
| `ImageLoader.py` | `plugins/dataloaders/ImageLoader.py`    | `CustomDataLoader`  | `.npz`, `.npy`       |
| `RnaLoader.py`   | `plugins/dataloaders/RnaLoader.py`      | `CustomDataLoader`  | `.csv`, `.tsv`       |

**Practical rule:** pick the dataloader that matches your `train` / `test` file type.  
For `.npz` MNIST data, use `ImageLoader` or `ImageLoader.py`.

---

## `trainer.optimizer.name`

Already filled as `SGD` in the partial template. Documented here for completeness.

Resolved by `utils.get_optimizer()` → `torch.optim` or custom file with class **`CustomOptimizer`**.

### Built-in (PyTorch)

Common valid names (any class in `torch.optim`):

- `SGD`
- `Adam`
- `AdamW`
- `RMSprop`

### Custom (plugin `.py` on `mnt/input/`)

- Any `<optimizer>.py` that defines **`CustomOptimizer`** (no extra optimizer plugins in this repo by default).

---

## `trainer.loss.name`

Resolved by `utils.get_loss_func()` → `torch.nn` or custom file with class **`CustomLoss`**.

### Built-in (PyTorch `torch.nn`)

Common valid names (any loss class in `torch.nn`), for example:

- `CrossEntropyLoss` (used in sample configs)
- `NLLLoss`
- `MSELoss`
- `BCEWithLogitsLoss`

These are **class names**, not filenames — that is why `CrossEntropyLoss` works without a file in `plugins/loss/`.

### Custom (plugin `.py` on `mnt/input/`)

| Value            | Repo file                    | Required class |
|------------------|------------------------------|----------------|
| `focal_loss.py`  | `plugins/loss/focal_loss.py` | `CustomLoss`   |

---

## `trainer.metrics[].param.num_classes`

- **Valid values:** positive integer — same meaning as `fed_hyper_params.n_classes`.
- Fill this when you fill `n_classes` so the Accuracy metric matches your task.

---

## `model.name`

Resolved by `utils.design_architecture()` → `models.pytorch.models` or custom file with class **`Model`**.

### Built-in

| Value | Source                    | Notes                    |
|-------|---------------------------|--------------------------|
| `CNN` | `models/pytorch/models.py` | Built-in CNN architecture |

### Custom (plugin `.py` on `mnt/input/`)

| Value              | Repo file (examples)              | Required class |
|--------------------|-----------------------------------|----------------|
| `cnn.py`           | `plugins/models/cnn.py`             | `Model`        |
| `mlp.py`           | `plugins/models/mlp.py`             | `Model`        |
| `customized_model_v2.py` | `plugins/models/generated_architecture/...` | `Model` |

Any `<name>.py` mounted at `mnt/input/` with a `Model(n_classes, in_features)` class is valid.

---

## `model.n_classes`

- **Valid values:** positive integer — number of output classes.
- Should match `fed_hyper_params.n_classes` and `trainer.metrics[].param.num_classes`.

**Example:** `10`

---

## `model.in_features`

- **Valid values:** positive integer — input size for the model.
- For **CNN** / `ImageLoader`: usually **number of channels** (MNIST grayscale → `1`).
- For **MLP** / tabular data: number of input features per sample.

**Example:** `1` (MNIST images, 1 channel)

---

## Quick reference (empty fields in partial template)

| Field | Built-in options | Custom plugin options (this repo) |
|-------|------------------|-----------------------------------|
| `local_dataset.train` | — | User filenames (e.g. `train.npz`) |
| `local_dataset.test` | — | User filenames (e.g. `test.npz`) |
| `logic.dir` | — | User folder name (e.g. `data`) |
| `fed_hyper_params.n_classes` | — | Any positive integer |
| `fed_hyper_params.federated_model` | `FedAvg` | `FedAvg.py` |
| `trainer.name` | `BasicTrainer` | `FedMMb.py` |
| `trainer.data_loader` | `ImageLoader` | `ImageLoader.py`, `RnaLoader.py` |
| `trainer.loss.name` | `CrossEntropyLoss`, … | `focal_loss.py` |
| `model.name` | `CNN` | `cnn.py`, `mlp.py`, … |
| `model.n_classes` | — | Positive integer |
| `model.in_features` | — | Positive integer |
| `metrics[].param.num_classes` | — | Same as `n_classes` |

---

## Federated-specific reminder

- Do **not** set `simulation` or `centralized` in this template.
- Run with two client mounts, e.g. `--client-dirs "sample_data/c1,sample_data/c2"` and `--generic-dir "sample_data/mnist/generic"`.
- The same `config.yml` is merged into each client; each client uses **its own** `mnt/input/data/...` files.
