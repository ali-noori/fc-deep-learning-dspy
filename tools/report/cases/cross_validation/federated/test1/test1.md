# Test 1 — Federated cross-validation, EfficientNet-B0

Run the agentic layer to build a config and train it with FeatureCloud in **federated mode**. Two ways to start: **Mode 1 (GUI)** or **Mode 2 (step-by-step agents)**. Both give the same config.

Federated runs **two** FeatureCloud containers, one per client (`c1`, `c2`). They exchange model weights every round and the coordinator averages them per fold (FedAvg).

| Setting | Value |
| --- | --- |
| Dataset | `data/sample_data_cutomized/MNIST/cross_validation_sample` (clients `c1`, `c2`; each has the folds `data/0` and `data/1`) |
| Execution mode | federated (two containers, one per client) |
| Model | EfficientNet-B0 — `plugins/models/pre-trained_models/cnn_architectures/efficientnet_b0.py` |
| Rounds / classes / aggregator | `max_iter` 1 (communication rounds), 10 classes, `FedAvg` |
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
Federated learning on two clients: sample_data_cutomized/MNIST/cross_validation_sample/c1 and sample_data_cutomized/MNIST/cross_validation_sample/c2. Train file is train.npz, test file is test.npz. The split folder under mnt/input is data.

max_iter: 1. n_class: 10. federated_model: FedAvg.

name: BasicTrainer. data_loader: ImageLoader. loss.name: CrossEntropyLoss. n_class: 10.

verbose: false. batch_size: 32. test_batch_size: 32. epochs: 1. lr: 0.1. batch_count: 1.
```

**Window 2 — "Please enter your prompt for model section".** Paste:

```text
efficientnet_b0.py. number of input is 1 and number of class is 10.
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
| 1 | `execution_mode_dspy` | `federated` |
| 2 | `dataset_dspy` | `I use two clients: sample_data_cutomized/MNIST/cross_validation_sample/c1 and sample_data_cutomized/MNIST/cross_validation_sample/c2. Train file is train.npz, test file is test.npz. The split folder under mnt/input is data.` |
| 3 | `hyper_params_dspy` | `max_iter: 1 n_class: 10 federated_model: FedAvg` |
| 4 | `Do you want to generate a new model? (Yes/No)` | `No` |
| 5 | `model_dspy` | `name: efficientnet_b0.py n_class: 10 in_features: 1` |
| 6 | `trainer_dspy` | `name: BasicTrainer data_loader: ImageLoader loss.name: CrossEntropyLoss n_class: 10` |
| 7 | `train_config_dspy` | `verbose: false batch_size: 32 test_batch_size: 32 epochs: 1 lr: 0.1 batch_count: 1` |

Then `config_builder` runs by itself and `config_executor` asks `Are you sure you want to run this command? (yes/no)`. Go to **Check and run**.

## Check and run

Every agent must print `Agent: Okay!`. If one prints an error, stop and start again from the beginning. A failed agent does not stop the workflow, and old output files may be reused silently.

Before answering the question, open `tools/config_builder/output/federated/federated_config.yml`. It must contain:

```yaml
fed_hyper_params: {max_iter: 1, n_classes: 10, federated_model: FedAvg}
trainer: {name: BasicTrainer, data_loader: ImageLoader, loss: {name: CrossEntropyLoss}}
train_config: {verbose: false, batch_size: 32, test_batch_size: 32, epochs: 1, lr: 0.1, batch_count: 1}
model: {name: plugins/models/pre-trained_models/cnn_architectures/efficientnet_b0.py, n_classes: 10, in_features: 1}
```

(`local_dataset` has `train.npz` / `test.npz`; `logic` has `mode: directory`, `dir: data`; there is **no** `simulation` key and **no** `centralized` key.)

The executor has already copied the config to `data/tools/config/config.yml` when it asks the question, and that staged copy is the one FeatureCloud uses. Check there that the model line shows only `efficientnet_b0.py` (the executor removed the `plugins/...` path) and that `efficientnet_b0.py` is in the same folder together with `efficientnet_b0_rwightman-7f5810bc.pth`.

Then type `yes` to run FeatureCloud. The printed command must contain `--app-image featurecloud.ai/fc_deep_v1`, `--client-dirs sample_data_cutomized/MNIST/cross_validation_sample/c1,sample_data_cutomized/MNIST/cross_validation_sample/c2` (two paths, comma-separated) and `--generic-dir tools/config`. Type `no` to stop after the config exists.

### What to expect

- Two client containers. The log shows the states `Local Update` and `Global Aggregation` and the communication round counter.
- FedAvg runs per fold: fold 0 of `c1` + fold 0 of `c2` give the new global model for fold 0; the same for fold 1. Fold 0 and fold 1 are never mixed.
- When the run finishes, the executor prints the accuracy per client and fold. Run `python tools/report/report_accuracy.py` to print it again. Expect **two zips** (`client_0`, `client_1`) and **four rows** (`data/0` and `data/1` for each client), each with `n=1500` test images. Result layout (one zip per client):

```text
output/
  config.yml
  data/0/  y_pred.csv  y_test.csv  model.pt
  data/1/  y_pred.csv  y_test.csv  model.pt
```

If Windows blocks FeatureCloud, see `tools/report/troubleshoot.md`.

## Result

Not recorded yet.
