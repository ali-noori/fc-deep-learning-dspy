# Test 2 — Centralized cross-validation, MobileNetV3-Small

Run the agentic layer to build a config and train it with FeatureCloud in **centralized mode**. Two ways to start: **Mode 1 (GUI)** or **Mode 2 (step-by-step agents)**. Both give the same config.

Centralized runs **one** FeatureCloud container and **one** process. There are no clients and no aggregation: each fold is trained on its own, starting from the same initial weights, so you get one separate model per fold.

| Setting | Value |
| --- | --- |
| Dataset | `data/sample_data_cutomized/MNIST/cross_validation_sample/c1` (one client folder; folds `data/0`, `data/1`) |
| Execution mode | centralized (one container, one process, no federation) |
| Model | MobileNetV3-Small — `plugins/models/pre-trained_models/cnn_architectures/mobilenetv3_small.py` |
| Rounds / classes / aggregator | `max_iter` 1 (**epochs per fold** in this mode), 10 classes, `FedAvg` (required by the config, not used for training) |
| Trainer / loader / loss | `BasicTrainer`, `ImageLoader`, `CrossEntropyLoss` |
| Train config | verbose false, batch 32, test batch 32, epochs 1, lr 0.1, batch_count 1 |
| Model input | `in_features` 1 (grayscale; the plugin converts it to 3 channels) |

Paths given to the agents are relative to the `data/` folder, so they do **not** start with `data/`.

## Mode 1 — GUI

```powershell
python -m tools.prompt_extraction.experiment_splitter_dspy.cli --recompile=False
```

**Window 1 — "Please enter you prompt".** Paste, then click **Next**:

```text
Centralized: train on this one machine with one dataset, no clients and no federation. The data folder is sample_data_cutomized/MNIST/cross_validation_sample/c1. Train file is train.npz, test file is test.npz. The split folder under mnt/input is data.

max_iter: 1. n_class: 10. federated_model: FedAvg.

name: BasicTrainer. data_loader: ImageLoader. loss.name: CrossEntropyLoss. n_class: 10.

verbose: false. batch_size: 32. test_batch_size: 32. epochs: 1. lr: 0.1. batch_count: 1.
```

**Window 2 — "Please enter your prompt for model section".** Paste:

```text
mobilenetv3_small.py. number of input is 1 and number of class is 10.
```

Leave both radio buttons on **No** and the name field empty. Click **Start**.

No more input is needed until the final `Are you sure you want to run this command? (yes/no)`. Go to **Check and run**.

On a server without a screen the windows cannot open there. Use Mode 2, or run the windows on your laptop as described in `tools/report/setup.md`, section B.4.

## Mode 2 — Step-by-step agents

```powershell
python -m tools.execution_mode_dspy.cli --recompile=False
```

Each agent asks one question. Paste the answer after `You:`.

| # | Agent / question | Paste |
| --- | --- | --- |
| 1 | `execution_mode_dspy` | `centralized` |
| 2 | `dataset_dspy` | `The data folder is sample_data_cutomized/MNIST/cross_validation_sample/c1. Train file is train.npz, test file is test.npz. The split folder under mnt/input is data.` |
| 3 | `hyper_params_dspy` | `max_iter: 1 n_class: 10 federated_model: FedAvg` |
| 4 | `Do you want to generate a new model? (Yes/No)` | `No` |
| 5 | `model_dspy` | `name: mobilenetv3_small.py n_class: 10 in_features: 1` |
| 6 | `trainer_dspy` | `name: BasicTrainer data_loader: ImageLoader loss.name: CrossEntropyLoss n_class: 10` |
| 7 | `train_config_dspy` | `verbose: false batch_size: 32 test_batch_size: 32 epochs: 1 lr: 0.1 batch_count: 1` |

Then `config_builder` runs by itself and `config_executor` asks `Are you sure you want to run this command? (yes/no)`. Go to **Check and run**.

## Check and run

Every agent must print `Agent: Okay!`. If one prints an error, stop and start again from the beginning. A failed agent does not stop the workflow, and old output files may be reused silently.

Before answering the question, open `tools/config_builder/output/centralized/centralized_config.yml`. It must contain:

```yaml
centralized: true
fed_hyper_params: {max_iter: 1, n_classes: 10, federated_model: FedAvg}
trainer: {name: BasicTrainer, data_loader: ImageLoader, loss: {name: CrossEntropyLoss}}
train_config: {verbose: false, batch_size: 32, test_batch_size: 32, epochs: 1, lr: 0.1, batch_count: 1}
model: {name: plugins/models/pre-trained_models/cnn_architectures/mobilenetv3_small.py, n_classes: 10, in_features: 1}
```

(`local_dataset` has `train.npz` / `test.npz`; `logic` has `mode: directory`, `dir: data`; there is **no** `simulation` key.)

The executor has already copied the config to `data/tools/config/config.yml` when it asks the question, and that staged copy is the one FeatureCloud uses. Check there that the model line shows only `mobilenetv3_small.py` (the executor removed the `plugins/...` path) and that `mobilenetv3_small.py` is in the same folder.

Then type `yes` to run FeatureCloud. The printed command must contain `--app-image featurecloud.ai/fc_deep_v1`, `--client-dirs sample_data_cutomized/MNIST/cross_validation_sample/c1` (one path) and `--generic-dir tools/config`. Type `no` to stop after the config exists.

### What to expect

- One client container. The log shows `Centralized mode`, `Training model #0` and `Training model #1`. There are no communication rounds and no `Local Update` / `Global Aggregation` state lines.
- Each fold gets its own model. The folds are never averaged together.
- When the run finishes, the executor prints the accuracy per fold. Run `python tools/report/report_accuracy.py` to print it again. Expect **one zip** and **two rows** (`data/0` and `data/1`), each with `n=1500` test images. Result layout (one zip):

```text
output/
  config.yml
  data/0/  y_pred.csv  y_test.csv  model.pt
  data/1/  y_pred.csv  y_test.csv  model.pt
```

If Windows blocks FeatureCloud, see `tools/report/troubleshoot.md`.

## Result

Not recorded yet.
