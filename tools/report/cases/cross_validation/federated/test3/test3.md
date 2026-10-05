# Test 3 — Federated, cross-validation (pretrained EfficientNet-B0)

**Goal:** Same data as test 1 / test 2, but **do not generate** a CNN. Use the existing plugin `plugins/models/pre-trained_models/cnn_architectures/efficientnet_b0.py`.  
**Data:** `data/sample_data_cutomized/cross_validation_sample`.  
**Vs test 1 / 2:** step 4 is **No** (skip `model_plugin_dspy_v2`). ImageNet backbone, frozen; only a 1×1 adapter and the last classifier head train.  
**Not run yet.** This is **not** faster than test 1 / 2 on CPU: each image is resized to **224×224**. Keep `max_iter` small.

## PowerShell (once)

```powershell
& "C:\UH\Thesis\FC\fc-deep-learning-env-zenbook\Scripts\Activate.ps1"
cd "C:\UH\Thesis\FC\fc-deep-learning-master\fc-deep-learning-master"
python -m tools.execution_mode_dspy.cli --recompile=False
```

Use the Zenbook venv. If FeatureCloud is blocked, see `tools/report/troubleshoot.md`.  
First run may **download** EfficientNet-B0 ImageNet weights (needs network inside the app container).

## Trace (agent + prompt)

Prompts are one line unless noted.

1. **`execution_mode_dspy`**
   ```text
   federated
   ```

2. **`dataset_dspy`**
   ```text
   I use two clients: sample_data_cutomized/cross_validation_sample/c1 and sample_data_cutomized/cross_validation_sample/c2. Train file is train.npz, test file is test.npz. The split folder under mnt/input is data.
   ```

3. **`hyper_params_dspy`**
   ```text
   max_iter: 5 n_class: 10 federated_model: FedAvg
   ```

4. **Generate a new model?** `No`  
   (Do **not** run `model_plugin_dspy_v2`. There is no “save reusable” step.)

5. **`model_dspy`**
   ```text
   name: efficientnet_b0.py n_class: 10 in_features: 1
   ```

   `in_features: 1` is MNIST grayscale. The plugin maps 1 channel → 3 with a 1×1 conv.

### Layer view (plugin, not generated)

```text
input  [N, 1, 28, 28]     (in_features=1)
   |
   v
Conv2d 1 -> 3, kernel 1   (to_rgb; skipped if in_features=3)
   |
   v
Upsample to 224 x 224
   |
   v
EfficientNet-B0 backbone  (ImageNet weights, frozen)
   |
   v
classifier last Linear -> n_classes   [N, 10]
(only this head and to_rgb are trained)
```

6. **`trainer_dspy`**
   ```text
   name: BasicTrainer data_loader: ImageLoader loss.name: CrossEntropyLoss n_class: 10
   ```

7. **`config_builder`** (no prompt) writes the federated YAML.

8. **Epochs** (not an agent). Leave template **`epochs: 1`** for this case (do not copy test 2’s `epochs: 3` unless you accept a much longer CPU run).

9. **`config_executor`** → `featurecloud test start` — type `yes`.

## Planned config

- `logic.mode: directory`, `dir: data`
- `max_iter: 5`, `FedAvg`, `n_classes: 10`
- `BasicTrainer` / `ImageLoader` / `CrossEntropyLoss`
- `model.name`: path that resolves to `efficientnet_b0.py` (repo-relative under `plugins/models/pre-trained_models/cnn_architectures/`)
- `in_features: 1`
- `epochs: 1`, `batch_size: 32`, `device: cpu`

## Result

Not executed yet. This case checks the **pretrained plugin path**, not a short wall-clock vs test 1.
