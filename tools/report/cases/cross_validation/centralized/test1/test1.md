# Test 1 — Centralized, cross-validation (smoke)

**Goal:** Check DSPy → centralized run on the **small testbed**, not high accuracy.  
**Data:** `data/sample_data_centralized` (`data/0` and `data/1` with `train.npz` / `test.npz`). Same tree as `Run_App.md` centralized. **No** `c1` / `c2`.  
**Model:** existing `plugins/models/cnn.py` — do **not** generate a new plugin.  
**Vs federated tests:** mode `centralized`; `max_iter` here is **epochs per fold**, not FedAvg rounds.

**Not run yet.**

## PowerShell (once)

```powershell
& "C:\UH\Thesis\FC\fc-deep-learning-env-zenbook\Scripts\Activate.ps1"
cd "C:\UH\Thesis\FC\fc-deep-learning-master\fc-deep-learning-master"
python -m tools.execution_mode_dspy.cli --recompile=False
```

Use the Zenbook venv. If FeatureCloud is blocked, see `tools/report/troubleshoot.md`.

## Trace (agent + prompt)

Prompts are one line unless noted.

1. **`execution_mode_dspy`**
   ```text
   centralized
   ```

2. **`dataset_dspy`**
   ```text
   I use one directory: sample_data_centralized. Train file is train.npz, test file is test.npz. The split folder under mnt/input is data.
   ```

3. **`hyper_params_dspy`** (still asked; `FedAvg` is unused in centralized)
   ```text
   max_iter: 2 n_class: 10 federated_model: FedAvg
   ```

4. **Generate a new model?** `No`  
   (Skip `model_plugin_dspy_v2`. No “save reusable” step.)

5. **`model_dspy`**
   ```text
   name: cnn.py n_class: 10 in_features: 1
   ```

### Layer view (existing `cnn.py`)

```text
input  [N, in_features, 28, 28]
   |
   v
Conv2d  in_features -> 10, kernel 5     ->  [N, 10, 24, 24]
MaxPool2d(2) + ReLU                    ->  [N, 10, 12, 12]
   |
   v
Conv2d  10 -> 20, kernel 5              ->  [N, 20,  8,  8]
Dropout2d
MaxPool2d(2) + ReLU                    ->  [N, 20,  4,  4]
   |
   v
Flatten                                ->  [N, 320]
Linear  320 -> 50 + ReLU
Dropout
Linear  50 -> n_classes                 ->  [N, 10]
log_softmax
```

6. **`trainer_dspy`**
   ```text
   name: BasicTrainer data_loader: ImageLoader loss.name: CrossEntropyLoss n_class: 10
   ```

7. **`config_builder`** (no prompt) writes `tools/config_builder/output/centralized/centralized_config.yml`.

8. **`config_executor`** — laptop FeatureCloud (same idea as federated tests):

   ```powershell
   python -m tools.config_executor.cli --execution-mode centralized --keep-staging
   ```

   Type `yes`. Client dir should be `sample_data_centralized`.

## Planned config

- `centralized: true` (no `simulation` block)
- `logic.mode: directory`, `dir: data`
- `max_iter: 2` (epochs **per fold**), `n_classes: 10`
- `BasicTrainer` / `ImageLoader` / `CrossEntropyLoss`
- `model.name` resolves to `cnn.py`, `in_features: 1`
- `epochs: 1` (template default unless you edit YAML), `batch_size: 32`, `device: cpu`

## Optional: Docker `STANDALONE=1` (this branch)

After the YAML exists, you can test **without** the FeatureCloud controller (server-style). Put `config.yml`, `cnn.py`, data, and any plugins the YAML names into one `input/` folder, then:

```bash
docker build -t featurecloud.ai/fc_deep_networks .
docker run --rm \
  -e STANDALONE=1 \
  -v "$PWD/data/scenarios/<your_centralized_input>:/mnt/input" \
  -v "$PWD/data/scenarios/<your_centralized_output>:/mnt/output" \
  featurecloud.ai/fc_deep_networks
```

Standalone **forces** centralized even if YAML had federated/simulation.

## Result

Not executed yet. Target: pipeline works on the **small** centralized testbed; accuracy may stay near chance.
