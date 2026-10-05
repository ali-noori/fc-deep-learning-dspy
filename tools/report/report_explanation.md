# Result layout (`logic.mode`)

Local files come from `fc_deep.result` (`y_pred.csv`, `y_test.csv`, `model.pt`).  
Each client also gets a copy of `config.yml` in its output root.

`central_test` is optional. If it is set (not `null`), the coordinator (federated) or first simulated client also writes `central_pred.csv` and `central_target.csv` at **output root** — not inside fold folders. Trees below assume `central_test: null`.

Folds exist only when `logic.mode: directory`. Clients are FeatureCloud / simulation participants (`c1`, `c2`, …), not fold ids.

---

## `logic.mode: directory`

Input uses fold folders, e.g. `data/0/train.npz` + `data/0/test.npz` (and the same for `1`, …).  
Outputs are **one trio per fold**.

### Federated

One FeatureCloud output zip per client. Example: 2 clients × 2 folds.

```text
client 1 output/
  config.yml
  data/0/  y_pred.csv  y_test.csv  model.pt
  data/1/  y_pred.csv  y_test.csv  model.pt

client 2 output/
  config.yml
  data/0/  y_pred.csv  y_test.csv  model.pt
  data/1/  y_pred.csv  y_test.csv  model.pt
```

`y_pred` / `y_test` are that **client’s local test** for that fold.  
`model.pt` for the same fold is the **same global weights** on every client.

### Simulation

Same local CSVs as federated, under each simulated client. Models are written **once** at the shared split paths (not copied into every `c1`/`c2` tree).

```text
output/
  config.yml
  data/0/model.pt
  data/1/model.pt
  c1/data/0/  y_pred.csv  y_test.csv
  c1/data/1/  y_pred.csv  y_test.csv
  c2/data/0/  y_pred.csv  y_test.csv
  c2/data/1/  y_pred.csv  y_test.csv
```

### Centralized

One process, **one** input tree (usually one client’s folder, e.g. only `c1`). No loop over `c1` and `c2`.  
Number of trios = number of fold folders.

```text
output/
  config.yml
  data/0/  y_pred.csv  y_test.csv  model.pt   # trained on that fold only
  data/1/  y_pred.csv  y_test.csv  model.pt
```

---

## `logic.mode: file`

No fold folders. `train.npz` / `test.npz` sit next to `config.yml`. **One** local trio per client process.

### Federated

```text
client 1 output/
  config.yml  y_pred.csv  y_test.csv  model.pt

client 2 output/
  config.yml  y_pred.csv  y_test.csv  model.pt
```

Same global `model.pt` on both clients; CSVs use each client’s own `test.npz`.

### Simulation

```text
output/
  config.yml
  model.pt
  c1/  y_pred.csv  y_test.csv
  c2/  y_pred.csv  y_test.csv
```

### Centralized

Mount **one** client’s files as the whole input.

```text
output/
  config.yml  y_pred.csv  y_test.csv  model.pt
```

---

## Quick count (2 clients, 2 folds, no `central_test`)

| Mode        | directory                         | file                                      |
|-------------|-----------------------------------|-------------------------------------------|
| Federated   | 2 zips × 2 folders × 3 files      | 2 zips × 3 files at output root           |
| Simulation  | CSVs under each `c*/data/*`; 2 models at `output/data/*` | CSVs under `c1`/`c2`; 1 model at output root |
| Centralized | 2 folders (if that input has 2 folds) | 1 trio at output root                     |
