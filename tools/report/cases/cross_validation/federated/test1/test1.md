# Test 1 — Federated, cross-validation (smoke)

**Goal:** Check the full DSPy → FeatureCloud path, not high accuracy.  
**Data:** `data/sample_data_cutomized/cross_validation_sample` (`c1` / `c2`, folds `data/0` and `data/1`).

## PowerShell (once)

```powershell
& "C:\UH\Thesis\FC\fc-deep-learning-env-zenbook\Scripts\Activate.ps1"
cd "C:\UH\Thesis\FC\fc-deep-learning-master\fc-deep-learning-master"
python -m tools.execution_mode_dspy.cli --recompile=False
```

Use the Zenbook venv, not `C:\Installation\python`. If FeatureCloud is blocked by Windows, see `tools/report/troubleshoot.md`.

## Trace (agent + prompt)

Prompts are one line unless noted (PowerShell).

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

4. **Generate a new model?** `Yes`

5. **`model_plugin_dspy_v2`** — architecture, then a **blank line**
   ```text
   Create a simple CNN like plugins/models/cnn.py for 28x28 images. in_features channels, n_classes classes. Conv2d: in_features->10 kernel=5 no padding. MaxPool2d(2) ReLU. Conv2d: 10->20 kernel=5 no padding. Dropout2d. MaxPool2d(2) ReLU. Flatten to 320. Linear 320->50 ReLU. Dropout. Linear 50->n_classes. log_softmax output.
   ```

### Layer view (step 5)

28×28 image, `in_features` channels → `n_classes` outputs:

```text
input  [N, in_features, 28, 28]
   |
   v
Conv2d  in_features -> 10, kernel 5, no padding     ->  [N, 10, 24, 24]
MaxPool2d(2) + ReLU                                ->  [N, 10, 12, 12]
   |
   v
Conv2d  10 -> 20, kernel 5, no padding              ->  [N, 20,  8,  8]
Dropout2d
MaxPool2d(2) + ReLU                                ->  [N, 20,  4,  4]
   |
   v
Flatten                                            ->  [N, 320]
Linear  320 -> 50 + ReLU
Dropout
Linear  50 -> n_classes                             ->  [N, 10]
log_softmax                                        ->  log probabilities
```

6. **Save reusable?** `Yes` — **name:** `c_f_test1`

7. **`model_dspy`**
   ```text
   name: c_f_test1.py n_class: 10 in_features: 1
   ```

8. **`trainer_dspy`**
   ```text
   name: BasicTrainer data_loader: ImageLoader loss.name: CrossEntropyLoss n_class: 10
   ```

9. **`config_builder`** (no prompt) writes the federated YAML.

10. **`config_executor`** → `featurecloud test start` — type `yes`.

## What ran

Staged config: `data/tools/config/config.yml`.

- `logic.mode: directory`, `dir: data`
- `max_iter: 5`, `FedAvg`, `n_classes: 10`
- `BasicTrainer` / `ImageLoader` / `CrossEntropyLoss`
- `model.name: c_f_test1.py`, `in_features: 1`
- `epochs: 1`, `batch_size: 32`, `device: cpu`

Plugin `c_f_test1.py` was copied next to that config.

## Result

The run finished. **Accuracy ≈ 10%** (about chance for 10 classes). That matches 5 FedAvg rounds and `epochs: 1`. The pipeline worked; this is not a strong MNIST score.
