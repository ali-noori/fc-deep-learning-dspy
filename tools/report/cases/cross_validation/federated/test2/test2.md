# Test 2 — Federated, cross-validation (more training)

**Goal:** Same data and same-size CNN as test 1, but enough local/global steps to aim for about **30%** accuracy on a laptop. Not a large network.  
**Data:** same as test 1 — `data/sample_data_cutomized/cross_validation_sample`.  
**Vs test 1:** `max_iter: 10`, `epochs: 3`, **no `log_softmax`** (raw scores + `CrossEntropyLoss`).  
**Not run yet.** After the YAML is built, set `train_config.epochs: 3` yourself (agents do not set epochs).

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
   federated
   ```

2. **`dataset_dspy`**
   ```text
   I use two clients: sample_data_cutomized/cross_validation_sample/c1 and sample_data_cutomized/cross_validation_sample/c2. Train file is train.npz, test file is test.npz. The split folder under mnt/input is data.
   ```

3. **`hyper_params_dspy`**
   ```text
   max_iter: 10 n_class: 10 federated_model: FedAvg
   ```

4. **Generate a new model?** `Yes`

5. **`model_plugin_dspy_v2`** — architecture, then a **blank line**
   ```text
   Create a simple CNN like plugins/models/cnn.py for 28x28 images, in_features channels, n_classes classes. Conv2d in_features->10 kernel=5 no padding. MaxPool2d(2) ReLU. Conv2d 10->20 kernel=5 no padding. Dropout2d. MaxPool2d(2) ReLU. Flatten to 320. Linear 320->50 ReLU. Dropout. Linear 50->n_classes. Return the Linear scores only. Do not use softmax or log_softmax.
   ```

### Layer view (step 5)

Same sizes as test 1 until the last step: **no log_softmax** (logits for `CrossEntropyLoss`).

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
(stop here; raw scores, not log_softmax)
```

6. **Save reusable?** `Yes` — **name:** `c_f_test2`

7. **`model_dspy`**
   ```text
   name: c_f_test2.py n_class: 10 in_features: 1
   ```

8. **`trainer_dspy`**
   ```text
   name: BasicTrainer data_loader: ImageLoader loss.name: CrossEntropyLoss n_class: 10
   ```

9. **`config_builder`** (no prompt) writes the federated YAML.

10. **Set epochs** (not an agent). In `tools/config_builder/output/federated/federated_config.yml` or `data/tools/config/config.yml`:

    ```yaml
    train_config:
      epochs: 3
      batch_size: 32
    ```

11. **`config_executor`** → `featurecloud test start` — type `yes`.

## Planned config

- `logic.mode: directory`, `dir: data`
- `max_iter: 10`, `FedAvg`, `n_classes: 10`
- `BasicTrainer` / `ImageLoader` / `CrossEntropyLoss`
- `model.name: c_f_test2.py`, `in_features: 1`
- `epochs: 3`, `batch_size: 32`, `device: cpu`

## Result

Not executed yet. Target: about **30%** accuracy (still a modest laptop run, not a full MNIST score).
