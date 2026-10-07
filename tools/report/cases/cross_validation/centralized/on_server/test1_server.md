# Test 1 (server) — Centralized CV, EfficientNet-B0, `STANDALONE=1`

**Goal:** Full DSPy agent chain in **centralized** mode, then train **without** the FeatureCloud controller.  
**Host:** `mario` — repo `~/fc-deep-learning-dspy`, env `~/fc-deep-learning-env`. Branch **`standalone-local`**.  
**Data:** `data/sample_data_cutomized/cross_validation_sample/c1` only (`data/0` and `data/1`).
**Model:** existing `plugins/models/pre-trained_models/cnn_architectures/efficientnet_b0.py` (+ `.pth` next to it). **No** new plugin from `model_plugin_dspy_v2`.  
**`max_iter`:** 2 (= 1 epochs **per fold**)

FeatureCloud `test start` is **not** used on this server (controller/port 8000). Training is **`docker run -e STANDALONE=1`**. Rebuild the image from this branch if `docker run` still shows supervisord spawn loops.

---

## 1. Env

```bash
cd ~
source ~/fc-deep-learning-env/bin/activate
cd ~/fc-deep-learning-dspy
```

Need `tools/keys_secrets.py` (Git-ignored; copy from the laptop if missing), the **c1** npz files, `efficientnet_b0.py`, and `efficientnet_b0_rwightman-7f5810bc.pth`.

---

## 2. Agents

```bash
python -m tools.execution_mode_dspy.cli --recompile=False
```

| Step | Input |
|------|--------|
| Execution mode | `centralized` |
| Dataset | `data_dir: sample_data_cutomized/cross_validation_sample/c1` then `train_dataset_file_name: train.npz` `test_dataset_file_name: test.npz` `logic_dir: data` |
| Hyper-params | `max_iter: 2 n_class: 10 federated_model: FedAvg` |
| Generate a new model? | `No` (skips `model_plugin_dspy_v2`) |
| Model | `name: efficientnet_b0.py n_class: 10 in_features: 1` |
| Trainer | `name: BasicTrainer data_loader: ImageLoader loss.name: CrossEntropyLoss n_class: 10` |

Builder writes `tools/config_builder/output/centralized/centralized_config.yml`.  
When the executor asks `yes/no`, type **`no`**. (If it waits for zips: Ctrl+C.)

**Why `no`:** `yes` would run `featurecloud test start` (controller on port 8000, app containers). On this server that path is blocked or broken (no working FeatureCloud controller; `Error: Not Found` / port 8000 used by something else). The YAML is already built. Training is the Docker `STANDALONE=1` step below, which is the supervisor workaround for that restriction.

**For your information**

**If FeatureCloud worked (normal laptop workflow):** type **`yes`** instead (or run the executor yourself). The controller must be **Up** and `--data-dir` must be this repo’s `data/`. Then the usual command is:

```bash
python -m tools.config_executor.cli --execution-mode centralized --keep-staging
# confirm: yes
```

That starts (defaults):

```bash
python -m FeatureCloud.api.cli test start \
  --app-image featurecloud.ai/fc_deep_networks1 \
  --client-dirs sample_data_cutomized/cross_validation_sample/c1 \
  --generic-dir tools/config \
  --controller-host http://localhost:8000
```

Centralized = **one** app container. Results land as zips under `data/tests/`. You would **not** need the Docker `STANDALONE=1` section. On this server we skip that and use section 3.

---

## 3. Docker input

```bash
cd ~/fc-deep-learning-dspy
SCEN=data/scenarios/centralized_cv_test1
mkdir -p "$SCEN/input/data/0" "$SCEN/input/data/1" "$SCEN/output"
mkdir -p "$SCEN/input/plugins/models/pre-trained_models/cnn_architectures"
mkdir -p "$SCEN/input/plugins/dataloaders"

cp tools/config_builder/output/centralized/centralized_config.yml "$SCEN/input/config.yml"
cp plugins/models/pre-trained_models/cnn_architectures/efficientnet_b0.py \
   plugins/models/pre-trained_models/cnn_architectures/efficientnet_b0_rwightman-7f5810bc.pth \
   "$SCEN/input/plugins/models/pre-trained_models/cnn_architectures/"
cp plugins/dataloaders/ImageLoader.py "$SCEN/input/plugins/dataloaders/"
cp data/sample_data_cutomized/cross_validation_sample/c1/data/0/*.npz "$SCEN/input/data/0/"
cp data/sample_data_cutomized/cross_validation_sample/c1/data/1/*.npz "$SCEN/input/data/1/"
```

If `config.yml` has `name: efficientnet_b0.py` (filename only), also copy the `.py` and `.pth` into `$SCEN/input/`.

Old `output/` files may be owned by root (`rm` → Permission denied). Skip delete; Docker overwrites the same names. Or: `sudo rm -rf "$SCEN/output/"*` then `sudo chmod -R a+rwx "$SCEN/output"`.

```bash
docker run --rm \
  -e STANDALONE=1 \
  -v "$PWD/data/scenarios/centralized_cv_test1/input:/mnt/input" \
  -v "$PWD/data/scenarios/centralized_cv_test1/output:/mnt/output" \
  featurecloud.ai/fc_deep_networks1
```

Silence after `Training model #0` is normal (~2 min/epoch on CPU). Check `docker stats` if needed.

---

## 4. Observed result

`transition: terminal` after ~9 min.

| Log | Train acc (epoch 1) |
|-----|---------------------|
| Training model #0 (`data/1` first in split list) | 0.8854 |
| Training model #1 (`data/0`) | 0.9098 |

Outputs: `$SCEN/output/data/0/` and `data/1/` → `y_pred.csv`, `y_test.csv`, `model.pt`. Those train numbers are **not** test accuracy; score the CSVs if needed.
