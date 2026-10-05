# Mental model: federated training and testing (implementation-backed)

This document matches how the **FeatureCloud deep-learning app in this repository** behaves, based on the code paths below. It focuses on the **normal federated path** (`states.py` → `utils/pytorch/states.py`: `Initialization` → `LocalUpdate` ↔ `GlobalAggregation` → `WriteResults`), not `Centralized` or `Simulation` unless noted.

---

## 1. What “splits” are (data layout)

- Config is read in `CustomStates/ConfigState.py` → `finalize_config()`.
- If `logic.mode` is **`directory`** and `logic.dir` is e.g. **`data`**, the app scans **subdirectories** of `mnt/input/data/` (each subfolder is one split, e.g. `0`, `1`).
- For each key under `local_dataset` (e.g. `train`, `test`), it builds **one path per split**:  
  `{split_dir}/{filename}` (e.g. `.../data/0/train.npz`, `.../data/1/train.npz`).

So **split index** `0` / `1` here means **cross-validation / multi-fold layout on one client**, not “client id”. Clients are separate FeatureCloud participants (`c1`, `c2` in your tests).

---

## 2. What gets built at startup (`Initialization`)

Relevant code: `utils/pytorch/states.py` → `Initialization.run`, `load_clients_data`, `store_`.

- One **`ClientModels`** instance per client process: a **single** underlying `nn.Module` trainer (`DeepModel` / plugin trainer).
- **`train_loaders`** and **`test_loaders`** are **Python lists**, length = number of splits: one train loader and one test loader **per split folder**.
- `n_splits = len(train_loaders)` is stored.
- Coordinator may attach a **`central_test`** loader if configured; otherwise it stays `None` (your sample uses `central_test: null`).
- Coordinator broadcasts **initial weights** as a single payload: `broadcast_data([client_model.get_weights()])`.

Important nuance: there are **not** two permanent PyTorch modules in memory for two splits. There is **one** module whose weights are **reloaded** for each split inside a round (see §4).

---

## 3. Communication rounds vs `max_iter`

- Round counter for stopping is advanced in **`FedAvg.aggregate()`** (`utils/pytorch/optimizer.py`): `self.iteration += 1` each aggregation.
- **`fed_hyper_params.max_iter`** is passed into the aggregator as **`max_iter`** (via `Initialization.build_client_model` → aggregator `**param`).
- After aggregation, **`FedAvg.post_aggregate()`** sets stoppage flags: stop when `self.iteration >= self.max_iter`. Stoppage is **one boolean per split-model** (`len(self.global_weights)`), as a **flat** list of booleans (not nested lists), so `all(stoppage)` reflects real convergence intent.

State machine glue: `states.py` registers `Local Update` → `Global Aggregation` → `Local Update` … until convergence or terminal.

---

## 4. Local training in one round (`LocalUpdate`)

Relevant code: `LocalUpdate.run`, `preprocess_global_updates`, `local_computation`, `ClientModels.update`.

### 4.1 Incoming global updates

- `preprocess_global_updates()` turns the coordinator’s broadcast into a dict via `utils.unpack` / `utils.cv_first`.
- On **iteration 0**, the coordinator message is expanded to **per-split initial weights** plus per-split stoppage placeholders:  
  `[[w] * n_splits, [False] * n_splits]` so `cv_first` can treat **weights as a list indexed by split**.

### 4.2 Per-split loop (this is the “parallel split tracks” behavior)

`local_computation()` does:

```text
for each split index k:
  take (train_loader[k], test_loader[k], per-split global update, optimizer backup[k])
  optionally evaluate on test_loader[k] BEFORE update
  client_model.update(train_loader[k], updates_for_split_k, backup[k], test_loader[k])
  append local update payload for split k
  optionally evaluate on test_loader[k] AFTER update
```

`ClientModels.update()` always:

1. `set_weights(global_updates['weights'])` — loads **global weights for this split** into the **same** module.
2. `fit(...)` on **that split’s train loader** only.

So within one communication round, the **same physical model** is trained **sequentially** on split `0`, then split `1`, etc., producing **one local update list entry per split** (fed to aggregation).

---

## 5. Global aggregation (`GlobalAggregation` + `FedAvg`)

Relevant code: `GlobalAggregation.aggregate` → `FedAvg.aggregate`.

- Coordinator **`gather_data()`** collects each client’s list of per-split updates.
- `FedAvg.aggregate(params)` expects:
  - `params` = list over **clients**
  - each client entry = list over **splits**
  - each split entry = `(weights, n_samples)` when using `WEIGHTS_N_SAMPLES`.

It maintains **`global_weights` as a list with length = number of splits**:

- index `k` = **global model weights for split `k`**, computed by weighted averaging of **all clients’** local weights for split `k`.

So at the end of training you should think in terms of:

> **`n_splits` distinct global weight vectors**, evolving each round — **not** one single global vector reused across different split datasets.

---

## 6. Coordinator-side evaluation during training (optional)

Relevant code: `GlobalAggregation.evaluate_aggregated_models`.

- If `central_test` loader exists, coordinator loops `for w in global_weights`, sets weights, evaluates on **that same central test loader** for each split index.
- If `central_test` is **null**, this loop produces **no metrics entries** for that path (your runs).

This is separate from per-client CSV outputs at the end.

---

## 7. Final testing and outputs (`WriteResults`)

Relevant code: `WriteResults.run`, `write_local_test_results`, `write_central_test_results`.

### 7.1 Local per-split CSVs (`y_pred.csv` / `y_test.csv`)

`write_local_test_results()`:

```text
for each split index k:
  load global final weights weights[k] into the client model
  run predict on this client's test_loaders[k]
  write CSVs under that split's output paths
```

So **on each client**:

- global weights for split `k` (aggregated across all clients for split `k`) are evaluated on **that client’s** split-`k` test loader.

Across clients, split `k` global weights are **the same aggregated vector** (up to normal FP / transport details), but **each client’s test data for split `k` is different** if you put different NPZ files under each client’s `data/k/`.

### 7.2 Central test (only if configured)

If `central_test` is set and coordinator has `test_loader`, `write_central_test_results` uses **`weights[0]` only** (first split’s weights) for central prediction — that is a separate behavior from per-split local writes.

---

## 8. How this relates to your mental picture

Correct picture for **directory splits + multiple clients**:

1. Each round: each client trains **all splits locally** (sequentially on one shared module), sends **one update per split**.
2. Coordinator aggregates **per split index across clients**, producing **`n_splits` global weight vectors** each round.
3. After the last round: each client writes predictions for each split using the **final** `weights[k]` on its **own** `test_loaders[k]`.

Incorrect picture to avoid:

- “One global model for all splits” — the implementation keeps **parallel global weight vectors indexed by split**, not a single merged dataset model.

---

## 9. Where to look in code (quick index)

| Topic | Primary location |
|--------|-------------------|
| Split directory discovery | `CustomStates/ConfigState.py` → `finalize_config()` |
| Build loaders per split | `utils/pytorch/states.py` → `Initialization.load_clients_data()` |
| Per-split local train loop | `utils/pytorch/states.py` → `LocalUpdate.local_computation()` |
| FedAvg across clients per split | `utils/pytorch/optimizer.py` → `FedAvg.aggregate()` |
| Round stopping (`max_iter`) | `utils/pytorch/optimizer.py` → `FedAvg.post_aggregate()` |
| Final per-client CSV predict | `utils/pytorch/states.py` → `WriteResults.write_local_test_results()` |

---

## 10. Outside this document

- **`tools/report/report_accuracy.py`** scans `data/tests` for result zips; if multiple tests exist, filter by `results_test_<id>_...` when comparing runs. Result folder layouts: `tools/report/report_explanation.md`.
- **`Run_App.md`** lists minimal CLI steps to rebuild and run tests.

This file is descriptive only; it does not change runtime behavior.
