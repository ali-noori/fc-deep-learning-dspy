# Run App (PowerShell)

Guide for running the FeatureCloud deep learning app from this repo. All commands assume you are in the project unless noted.

---

## Execution modes — reference (read this first)

This app supports **three ways to run training**. They share the same model code (`cnn.py`, `ImageLoader.py`) and similar `config.yml` structure, but they differ in **how many Docker containers** FeatureCloud starts and **what the app does inside** them.

### Quick comparison

| | **Federated** | **Simulation** | **Centralized** |
|---|---|---|---|
| **Purpose** | Real multi-party federated learning | Test federated logic locally in one process | Classic local training (no federation) |
| **FeatureCloud containers** | 2 (one per client) | 1 | 1 |
| **`--client-dirs`** | Two paths: `c1`, `c2` | One path: whole folder | One path: whole folder |
| **Config switch** | Default (no extra block) | `simulation.clients_dir: "c1,c2"` | `centralized: true` |
| **Sample data folder** | `sample_data/` | `sample_data_simulation/` | `sample_data_centralized/` |
| **`max_iter` in config** | Number of **communication rounds** | Number of **communication rounds** | **Epochs per CV fold** (not rounds) |

**Rule of thumb:** Use **federated** when you want true multi-container FL. Use **simulation** when you want to debug federated behavior in one container. Use **centralized** when you only want local training on all folds with no aggregation loop.

---

### 1. Federated mode

**What it is:** Two separate FeatureCloud clients (`c1` and `c2`), each with its own data. One acts as **coordinator**. They exchange model weights over the network each round.

**Flow (simplified):**

Two Docker containers (`c1`, `c2`). Each container only sees its own folder: `c1/data/0`, `c1/data/1` or `c2/data/0`, `c2/data/1`. One container is the **coordinator** (it trains locally like the other client, and also runs aggregation).

There are **two CV folds** (0 and 1). Aggregation is **per fold**: fold 0 from c1 is averaged with fold 0 from c2; fold 1 from c1 with fold 1 from c2. Fold 0 and fold 1 never get mixed.

```text
SETUP (once)
  Coordinator builds the model and broadcasts initial global weights
  (one weight set per fold: fold 0 and fold 1)
  Both c1 and c2 receive them over the network

FOR each communication round (1 .. max_iter):

  1. Broadcast
     Coordinator sends current global weights to c1 and c2
     (separate global model for fold 0 and for fold 1)

  2. Local Update — c1 container
     Train on c1/data/0 using global weights for fold 0
     Train on c1/data/1 using global weights for fold 1
     Send both local updates → coordinator

  3. Local Update — c2 container (coordinator also trains here)
     Train on c2/data/0 and c2/data/1 the same way
     Send local updates → coordinator

  4. Global Aggregation — coordinator only
     FedAvg: average fold-0 weights from c1 + c2  →  new global model for fold 0
     FedAvg: average fold-1 weights from c1 + c2  →  new global model for fold 1

  5. Next round uses the new global weights; repeat until max_iter

END FOR

  Write Results: y_pred.csv, y_test.csv, model.pt per fold on each client
```

```text
                    fold 0                    fold 1
  c1 container   c1/data/0                 c1/data/1
                      \                        /
                       \   FedAvg per fold   /
                        \        ▲          /
  c2 (coordinator)   c2/data/0                 c2/data/1
```

**Practical example — when to use:** You are testing the app the way it would run in production FeatureCloud with **two hospitals** (or two sites), each keeping data on its own container.

**CLI pattern:**

```powershell
--client-dirs "sample_data/c1,sample_data/c2"
--generic-dir "sample_data/mnist/generic"
```

**Config:** `sample_data/mnist/generic/config.yml` — **no** `simulation` block, **no** `centralized: true`.

**Logs:** `Local Update`, `Global Aggregation`, communication rounds across containers.

---

### 2. Simulation mode

**What it is:** **One** container mounts the whole tree (`sample_data_simulation/`). The app reads `simulation.clients_dir: "c1,c2"` and **simulates** two clients **inside one Python process** — same federated round loop (local train → aggregate), but **no** second Docker container.

**Flow (simplified):**

**One** Docker container, **one** Python process — but the app **pretends** there are two clients (`c1` and `c2`), like federated mode. The federated **algorithm** is the same (local train → FedAvg → next round); only the **packaging** changes (no network between containers).

**Folder layout matters:** You need **both**:

- `data/0`, `data/1` at the input root — for fold discovery (`logic.dir: data`)
- `c1/data/0`, `c1/data/1`, `c2/data/0`, `c2/data/1` — where simulated clients actually train

Config: `simulation.clients_dir: "c1,c2"` tells the app which subfolders are virtual clients.

**Per-fold aggregation** (same as federated): fold 0 from c1 + fold 0 from c2 → global model for fold 0; fold 1 from c1 + fold 1 from c2 → global model for fold 1.

```text
SETUP (once)
  Mount: sample_data_simulation/
         data/0, data/1          ← fold discovery only
         c1/data/0, c1/data/1    ← Client #1 data
         c2/data/0, c2/data/1    ← Client #2 data
         mnist/generic/config.yml  (simulation.clients_dir: "c1,c2")

  State: Federated Simulation
  Build one model in memory; set initial global weights (like coordinator broadcast)

FOR each communication round (1 .. max_iter):

  Client #1  (logical client — folder c1/)
    • Load current global weights (fold 0 and fold 1)
    • Train on c1/data/0 and c1/data/1 locally
    • Store local updates in memory

  Client #2  (logical client — folder c2/)
    • Same on c2/data/0 and c2/data/1

  Aggregate  (in the same process — no FeatureCloud network)
    • FedAvg: c1 fold 0 + c2 fold 0  →  new global weights for fold 0
    • FedAvg: c1 fold 1 + c2 fold 1  →  new global weights for fold 1

  Next round uses the new global weights; repeat until max_iter

END FOR

  Write Results under output/c1/... and output/c2/... per fold
```

```text
  Federated:     [c1 container] ←──network──→ [c2 container]
  Simulation:    [ one container: c1 logic → c2 logic → FedAvg in memory ]
```

| | Federated | Simulation |
|---|-----------|------------|
| Containers | 2 | 1 |
| Client data | `c1/...` in container 1, `c2/...` in container 2 | `c1/...` and `c2/...` in the **same** mount |
| Weight exchange | FeatureCloud messaging | Python variables in one process |
| `max_iter` | Communication rounds | Communication rounds (same meaning) |

**Practical example — when to use:** You want federated-style rounds and logs (`Communication round`, `Client #1`) **without** managing two containers — useful for debugging data layout and simulation code paths.

**CLI pattern:**

```powershell
--client-dirs "sample_data_simulation"
--generic-dir "sample_data_simulation/mnist/generic"
```

**Config:** `simulation.clients_dir: "c1,c2"` — **do not** set `centralized: true`. Folder must include top-level `data/0`, `data/1` (for fold discovery) **and** `c1/`, `c2/` (for simulated clients).

**Logs:** `Simulation mode`, `Communication round`, `Client #1` / `Client #2`.

**Not the same as:** Passing `--client-dirs "sample_data_simulation/c1,sample_data_simulation/c2"` — that starts **two containers** again (federated-style), not simulation.

---

### 3. Centralized mode

**What it is:** **One** container, **one** process trains on **all CV folds** locally. There is **no** federated loop: no broadcast, no aggregation across clients, no simulated `Client #1` / `#2`.

**Flow (simplified):**

**One** Docker container, **one** process. Data lives under **`data/0/`** and **`data/1/`** at the input root — **not** under `c1/` or `c2/`. Those client folders are only for federated and simulation; centralized does not use them.

There is **no** federated loop: no broadcast, no FedAvg, no second container. You get **two separate trained models** at the end — one for fold 0, one for fold 1. Folds are **not** averaged together.

```text
SETUP (once)
  Mount: sample_data_centralized/
         data/0/train.npz, test.npz
         data/1/train.npz, test.npz
         mnist/generic/config.yml  (centralized: true)

  State: Centralized Training
  Build one CNN and take one snapshot of initial weights

FOR each CV fold (0, then 1):

  Training model #0  (fold 0 — data/0/)
    • Load the same initial weights into the model
    • Train only on data/0/train.npz
    • Run for max_iter epochs  ← max_iter = epochs here, not FL rounds
    • Evaluate on data/0/test.npz
    • Save trained weights → model for fold 0

  Training model #1  (fold 1 — data/1/)
    • Load the same initial weights again (not fold 0’s result)
    • Train only on data/1/train.npz
    • Run for max_iter epochs
    • Evaluate on data/1/test.npz
    • Save trained weights → model for fold 1

  Write Results
    • y_pred.csv, y_test.csv, model.pt under output/data/0/ and output/data/1/
```

```text
  sample_data_centralized/
    data/0  ──train──►  Model 0   (independent)
    data/1  ──train──►  Model 1   (independent)

  c1/ and c2/  →  not used in this mode
```

**Practical example — when to use:** You have **all data in one place** and only need standard centralized training (e.g. cross-validation folds on one machine) — not federated learning.

**CLI pattern:**

```powershell
--client-dirs "sample_data_centralized"
--generic-dir "sample_data_centralized/mnist/generic"
```

**Config:** `centralized: true` — **no** `simulation` block. Data lives under `data/0/`, `data/1/` at the input root (no `c1`/`c2` required).

**Logs:** `Centralized mode`, `Training model #0`, `Training model #1`.

**Important:** Here `fed_hyper_params.max_iter` means **epochs per fold**, not federated communication rounds.

---

### Config priority (do not mix)

The app picks a mode from `config.yml` in this order:

1. If `simulation` is set → **Simulation** (ignores `centralized` for the main path)
2. Else if `centralized: true` → **Centralized**
3. Else → **Federated**

Always use **one folder tree + one config** per test. Do not combine federated client paths with simulation or centralized config files.

---

### Side-by-side command examples (from `data/` folder)

All three assume the same image and controller; only paths and config differ.

**Federated:**

```powershell
featurecloud test start --app-image featurecloud.ai/fc_deep_networks1 --client-dirs "sample_data/c1,sample_data/c2" --generic-dir "sample_data/mnist/generic" --controller-host "http://localhost:8000"
```

**Simulation:**

```powershell
featurecloud test start --app-image featurecloud.ai/fc_deep_networks1 --client-dirs "sample_data_simulation" --generic-dir "sample_data_simulation/mnist/generic" --controller-host "http://localhost:8000"
```

**Centralized:**

```powershell
featurecloud test start --app-image featurecloud.ai/fc_deep_networks1 --client-dirs "sample_data_centralized" --generic-dir "sample_data_centralized/mnist/generic" --controller-host "http://localhost:8000"
```

---

## Prerequisites (both modes)

**1. Activate environment**

```powershell
& "C:\FC\fc-deep-learning-env\Scripts\Activate.ps1"
```

**2. Go to project root**

```powershell
cd "C:\FC\fc-deep-learning-master\fc-deep-learning-master"
```

**3. Build Docker image**

```powershell
featurecloud app build . featurecloud.ai/fc_deep_networks1
```

**4. Start controller** (skip if already running)

```powershell
featurecloud controller start --port=8000
```

---

## Federated mode (multi-container)

Uses **`data/sample_data/`**: two client datasets plus a shared generic folder (config and model code only — not training data under `mnist/`).

**Layout**

```text
data/sample_data/
  c1/                          ← client 1 (mounted as its own container)
    data/0/train.npz, test.npz
    data/1/train.npz, test.npz
  c2/                          ← client 2
    data/0/ ...
    data/1/ ...
  mnist/generic/               ← shared for all clients
    config.yml                 ← federated config (no simulation block)
    cnn.py
    ImageLoader.py
```

**Run test** (from `data` folder)

```powershell
cd "C:\FC\fc-deep-learning-master\fc-deep-learning-master\data"
featurecloud test start --app-image featurecloud.ai/fc_deep_networks1 --client-dirs "sample_data/c1,sample_data/c2" --generic-dir "sample_data/mnist/generic" --controller-host "http://localhost:8000"
```

**What to expect in logs:** `Local Update`, `Global Aggregation`, communication rounds across containers.

---

## Simulation mode (single coordinator process)

Uses **`data/sample_data_simulation/`**: same *style* as federated ( **`c1` and `c2` beside `mnist/generic`** ), but both clients are under one input root so the app can run `Federated Simulation` in one process.

**Layout**

```text
data/sample_data_simulation/
  c1/                          ← simulated client 1
    data/0/train.npz, test.npz
    data/1/train.npz, test.npz
  c2/                          ← simulated client 2
    data/0/ ...
    data/1/ ...
  mnist/generic/               ← config + model (not NPZ training data)
    config.yml                 ← includes simulation.clients_dir: "c1,c2"
    cnn.py
    ImageLoader.py
```

**Run test** (from `data` folder)

```powershell
cd "C:\FC\fc-deep-learning-master\fc-deep-learning-master\data"
featurecloud test start --app-image featurecloud.ai/fc_deep_networks1 --client-dirs "sample_data_simulation" --generic-dir "sample_data_simulation/mnist/generic" --controller-host "http://localhost:8000"
```

**What to expect in logs:** `Simulation mode`, `Communication round`, `Client #1` / `Client #2` (no cross-container `Local Update` / `Global Aggregation`).

---

## Centralized mode (single coordinator, no federation)

Uses **`data/sample_data_centralized/`**: one container trains on **all local CV folds** in a single process. There is **no** federated loop (no `Local Update` / `Global Aggregation`, no simulated multi-client loop).

**Layout**

```text
data/sample_data_centralized/
  data/0/train.npz, test.npz       ← CV fold 0 (discovered via logic.dir)
  data/1/train.npz, test.npz       ← CV fold 1
  mnist/generic/                   ← config + model (not NPZ training data)
    config.yml                     ← includes centralized: true (no simulation block)
    cnn.py
    ImageLoader.py
```

**Run test** (from `data` folder)

```powershell
cd "C:\FC\fc-deep-learning-master\fc-deep-learning-master\data"
featurecloud test start --app-image featurecloud.ai/fc_deep_networks1 --client-dirs "sample_data_centralized" --generic-dir "sample_data_centralized/mnist/generic" --controller-host "http://localhost:8000"
```

**What to expect in logs:** `Centralized mode`, `Training model #0`, `Training model #1`, then result writing. No communication rounds and no separate client containers.

**How centralized differs from the other modes**

| | Federated | Simulation | Centralized |
|---|---|---|---|
| **Containers** | 2 (`c1`, `c2`) | 1 | 1 |
| **`--client-dirs`** | `sample_data/c1,sample_data/c2` | `sample_data_simulation` | `sample_data_centralized` |
| **`--generic-dir`** | `sample_data/mnist/generic` | `sample_data_simulation/mnist/generic` | `sample_data_centralized/mnist/generic` |
| **Config trigger** | (default — no `simulation`, no `centralized`) | `simulation.clients_dir` | `centralized: true` |
| **Training** | Local train → aggregate → broadcast across containers | In-process loop over simulated clients | Train each fold locally in one process |
| **`max_iter` meaning** | Communication rounds | Communication rounds | **Epochs per CV fold** |

Do not mix paths or configs (e.g. centralized `config.yml` with federated `c1`/`c2` client dirs).

---

## Federated vs simulation (commands)

| | Federated | Simulation |
|---|-----------|--------------|
| **Data folder** | `sample_data/c1`, `sample_data/c2` | `sample_data_simulation/c1`, `sample_data_simulation/c2` |
| **Generic folder** | `sample_data/mnist/generic` | `sample_data_simulation/mnist/generic` |
| **`--client-dirs`** | `sample_data/c1,sample_data/c2` | `sample_data_simulation` |
| **`--generic-dir`** | `sample_data/mnist/generic` | `sample_data_simulation/mnist/generic` |
| **Config file** | `sample_data/mnist/generic/config.yml` | `sample_data_simulation/mnist/generic/config.yml` |

Do not mix paths (e.g. federated `c1`/`c2` with simulation `config.yml`); use one folder tree per mode.

---

## After any test run

**Check status**

```powershell
featurecloud test list --controller-host http://localhost:8000 --format json
```

**Check accuracy** (from project root)

```powershell
cd "C:\FC\fc-deep-learning-master\fc-deep-learning-master"
python ali_test.py --last-n 1
```

Results are read from `data/tests/` (prediction/target CSVs inside exported zips or extracted folders).
