# Test 3 — Simulation cross-validation, ResNet18

Run the agentic layer to build a config and train it with FeatureCloud in **simulation mode**. Two ways to start: **Mode 1 (GUI)** or **Mode 2 (step-by-step agents)**. Both give the same config.

Simulation runs **one** FeatureCloud container. Inside it the app pretends there are two clients (`c1`, `c2`), runs the same rounds as federated mode (local training, then FedAvg per fold) and keeps the weights in memory, so there is no network traffic between containers.

| Setting | Value |
| --- | --- |
| Dataset | `data/sample_data_cutomized/MNIST/cross_validation_sample` (simulated clients `c1`, `c2`; folds `data/0`, `data/1`) |
| Execution mode | simulation (one container, two simulated clients) |
| Model | ResNet18 — `plugins/models/pre-trained_models/cnn_architectures/resnet18.py` |
| Rounds / classes / aggregator | `max_iter` 1 (communication rounds), 10 classes, `FedAvg` |
| Trainer / loader / loss | `BasicTrainer`, `ImageLoader`, `CrossEntropyLoss` |
| Train config | verbose false, batch 32, test batch 32, epochs 1, lr 0.1, batch_count 1 |
| Model input | `in_features` 1 (grayscale; the plugin converts it to 3 channels) |

Paths given to the agents are relative to the `data/` folder, so they do **not** start with `data/`.

**Simulation only:** create the empty fold folders `data/0` and `data/1` at the dataset root, next to `c1/` and `c2/`. Git does not store empty folders, so do this on every machine.

Laptop (PowerShell):

```powershell
$root = "data\sample_data_cutomized\MNIST\cross_validation_sample"
New-Item -ItemType Directory -Force "$root\data\0", "$root\data\1" | Out-Null
```

Server (bash):

```bash
mkdir -p data/sample_data_cutomized/MNIST/cross_validation_sample/data/0 data/sample_data_cutomized/MNIST/cross_validation_sample/data/1
```

## Mode 1 — GUI

```powershell
python -m tools.prompt_extraction.experiment_splitter_dspy.cli --recompile=False
```

**Window 1 — "Please enter you prompt".** Paste, then click **Next**:

```text
Simulation: pretend there are two clients on this machine, no real remote sites. The simulation folder is sample_data_cutomized/MNIST/cross_validation_sample with client folders c1 and c2. Train file is train.npz, test file is test.npz. The split folder under mnt/input is data.

max_iter: 1. n_class: 10. federated_model: FedAvg.

name: BasicTrainer. data_loader: ImageLoader. loss.name: CrossEntropyLoss. n_class: 10.

verbose: false. batch_size: 32. test_batch_size: 32. epochs: 1. lr: 0.1. batch_count: 1.
```

**Window 2 — "Please enter your prompt for model section".** Paste:

```text
resnet18.py. number of input is 1 and number of class is 10.
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
| 1 | `execution_mode_dspy` | `simulation` |
| 2 | `dataset_dspy` | `The simulation folder is sample_data_cutomized/MNIST/cross_validation_sample with client folders c1 and c2. Train file is train.npz, test file is test.npz. The split folder under mnt/input is data.` |
| 3 | `hyper_params_dspy` | `max_iter: 1 n_class: 10 federated_model: FedAvg` |
| 4 | `Do you want to generate a new model? (Yes/No)` | `No` |
| 5 | `model_dspy` | `name: resnet18.py n_class: 10 in_features: 1` |
| 6 | `trainer_dspy` | `name: BasicTrainer data_loader: ImageLoader loss.name: CrossEntropyLoss n_class: 10` |
| 7 | `train_config_dspy` | `verbose: false batch_size: 32 test_batch_size: 32 epochs: 1 lr: 0.1 batch_count: 1` |

Then `config_builder` runs by itself and `config_executor` asks `Are you sure you want to run this command? (yes/no)`. Go to **Check and run**.

## Check and run

Every agent must print `Agent: Okay!`. If one prints an error, stop and start again from the beginning. A failed agent does not stop the workflow, and old output files may be reused silently.

Before answering the question, open `tools/config_builder/output/simulation/simulation_config.yml`. It must contain:

```yaml
simulation: {clients_dir: "c1,c2"}
fed_hyper_params: {max_iter: 1, n_classes: 10, federated_model: FedAvg}
trainer: {name: BasicTrainer, data_loader: ImageLoader, loss: {name: CrossEntropyLoss}}
train_config: {verbose: false, batch_size: 32, test_batch_size: 32, epochs: 1, lr: 0.1, batch_count: 1}
model: {name: plugins/models/pre-trained_models/cnn_architectures/resnet18.py, n_classes: 10, in_features: 1}
```

(`local_dataset` has `train.npz` / `test.npz`; `logic` has `mode: directory`, `dir: data`; there is **no** `centralized` key.)

The executor has already copied the config to `data/tools/config/config.yml` when it asks the question, and that staged copy is the one FeatureCloud uses. Check there that the model line shows only `resnet18.py` (the executor removed the `plugins/...` path) and that `resnet18.py` is in the same folder.

**Check `clients_dir`.** In the same file it must read `clients_dir: c1,c2` with **no space**. If it reads `c1, c2`, your code is not up to date (run `git pull`); as a stop-gap, change it to `c1,c2` in `data/tools/config/config.yml` and save, before you type `yes`. With a space the app looks for a folder named `" c2"`.

Then type `yes` to run FeatureCloud. The printed command must contain `--app-image featurecloud.ai/fc_deep_v1`, `--client-dirs sample_data_cutomized/MNIST/cross_validation_sample` (one path) and `--generic-dir tools/config`. Type `no` to stop after the config exists.

### What to expect

- One client container (not two). The log shows `Simulation mode`, `Communication round: #1 ...`, `Client #1: Local training`, `Client #2: Local training`. With `max_iter` 1 there is only this one round. There are no `Local Update` / `Global Aggregation` state lines.
- FedAvg runs per fold: `c1` fold 0 + `c2` fold 0 give the new global model for fold 0; the same for fold 1. Fold 0 and fold 1 are never mixed.
- When the run finishes, the executor prints the accuracy per simulated client and fold. Run `python tools/report/report_accuracy.py` to print it again. Expect **one zip** and **four rows** (`c1/data/0`, `c1/data/1`, `c2/data/0`, `c2/data/1`), each with `n=1500` test images. The folder names in the zip must be exactly `c1`, `c2` and `data`; a folder name that starts with a space (` c2`) means `clients_dir` had a space. Result layout (one zip):

```text
output/
  config.yml
  data/0/model.pt   data/1/model.pt
  c1/data/0/  y_pred.csv  y_test.csv
  c1/data/1/  y_pred.csv  y_test.csv
  c2/data/0/  y_pred.csv  y_test.csv
  c2/data/1/  y_pred.csv  y_test.csv
```

If Windows blocks FeatureCloud, see `tools/report/troubleshoot.md`.

## Result

Not recorded yet.
