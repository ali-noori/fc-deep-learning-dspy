# Test 1 — Simulation cross-validation, EfficientNet-B0

Run the agentic layer to build a config and train it with FeatureCloud in **simulation mode**. Two ways to start: **Mode 1 (GUI)** or **Mode 2 (step-by-step agents)**. Both give the same config.

Simulation runs **one** FeatureCloud container. Inside it the app pretends there are two clients (`c1`, `c2`), runs the same rounds as federated mode (local training, then FedAvg per fold) and keeps the weights in memory, so there is no network traffic between containers.

| Setting | Value |
| --- | --- |
| Dataset | `data/sample_data_cutomized/MNIST/cross_validation_sample` (simulated clients `c1`, `c2`; folds `data/0`, `data/1`) |
| Execution mode | simulation (one container, two simulated clients) |
| Model | EfficientNet-B0 — `plugins/models/pre-trained_models/cnn_architectures/efficientnet_b0.py` |
| Rounds / classes / aggregator | `max_iter` 1 (communication rounds), 10 classes, `FedAvg` |
| Trainer / loader / loss | `BasicTrainer`, `ImageLoader`, `CrossEntropyLoss` |
| Train config | verbose false, batch 32, test batch 32, epochs 1, lr 0.1, batch_count 1 |
| Model input | `in_features` 1 (grayscale; the plugin converts it to 3 channels) |

Paths given to the agents are relative to the `data/` folder, so they do **not** start with `data/`.

## What is different from the federated tests

- **One folder is mounted, not two.** The executor runs `--client-dirs sample_data_cutomized/MNIST/cross_validation_sample` (the whole tree). In federated mode it passes `.../c1,.../c2`.
- **The dataset agent needs the simulation answer:** one parent folder (`data_dir`) plus the client folder names (`c1,c2`).
- **The input root needs its own `data/0` and `data/1` folders.** The app finds the folds in `<root>/data` and then trains each simulated client on `<root>/c1/data/<fold>` and `<root>/c2/data/<fold>`.
- **The dataset agent for simulation must be compiled once.** Only the federated one is saved in the repo.

## Before you start (once)

1. Open PowerShell:

   ```powershell
   & "C:\UH\Thesis\FC\fc-deep-learning-env-zenbook\Scripts\Activate.ps1"
   cd "C:\UH\Thesis\FC\fc-deep-learning-master\fc-deep-learning-master"
   ```

2. Start Docker Desktop, then build the app image and start the controller (see `Run_App.md`; skip what is already done):

   ```powershell
   featurecloud app build . featurecloud.ai/fc_deep_networks1
   featurecloud controller start --port=8000
   ```

3. API keys must be in `tools/keys_secrets.py`.

4. **Create the fold folders at the input root.** The customized dataset only has `c1/` and `c2/`. Simulation also needs `data/0` and `data/1` next to them. Only the folder names are read from there (the files come from `c1/` and `c2/`), so empty folders are enough:

   ```powershell
   $root = "data\sample_data_cutomized\MNIST\cross_validation_sample"
   New-Item -ItemType Directory -Force "$root\data\0", "$root\data\1" | Out-Null
   Get-ChildItem $root
   ```

   The listing must show `c1`, `c2` and `data`.

5. **Compile the simulation dataset agent once.** The saved agents exist for every other step, but `tools/sub_config_dspy/dataset_dspy/compiled_agent/` only contains `compiled_agent_federated.json`. Without the simulation file, `--recompile=False` fails at the dataset step and the workflow carries on with an old dataset output file.

   ```powershell
   python -m tools.sub_config_dspy.dataset_dspy.cli --execution-mode simulation --recompile=True
   ```

   When it asks, paste:

   ```text
   The simulation folder is sample_data_cutomized/MNIST/cross_validation_sample with client folders c1 and c2. Train file is train.npz, test file is test.npz. The split folder under mnt/input is data.
   ```

   It must print `Agent: Okay!` and the file `compiled_agent_simulation.json` must now exist in the folder above. This takes some LM calls but is done only once. After that, every test uses `--recompile=False`.

EfficientNet-B0 weights are copied automatically next to the model file, so no internet is needed. Images are resized to 224×224, so training is slow on CPU. `max_iter` is 1 (a single communication round), so this is already a quick smoke run.

## Mode 1 — GUI

```powershell
python -m tools.intake.intake_dspy.cli --recompile=False
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
efficientnet_b0.py. number of input is 1 and number of class is 10.
```

Leave both radio buttons on **No** and the name field empty. Click **Start**.

No more input is needed until the final `Are you sure you want to run this command? (yes/no)`. Go to **Check and run**.

### Mode 1 on a server without a screen (temporary workaround)

The two windows need a screen, so run them on your laptop and run the rest on the server.

1. **Laptop:** do the Mode 1 steps above (command, Window 1, Window 2). When the windows are done, the terminal prints `Agent: wrote ...intake_process.json`. The file now exists, so stop with `Ctrl+C` (or type `no` at the final question).
2. **Laptop:** copy the file to the same place in the repo on the server:

   ```powershell
   scp tools\intake\output\intake_process.json <user>@<server>:<path-to-repo>/tools/intake/output/
   ```

3. **Server:** activate the venv, go to the repo folder, and run:

   ```bash
   python -m tools.execution_mode_dspy.cli --from-intake --recompile=False
   ```

   This reads the file and runs the same agents as Mode 1, with no questions. It ends with the `yes/no` question. Go to **Check and run**.

The server does not need `tkinter`. It does need the repo, the dataset folders (including the `data/0` and `data/1` folders from step 4), the saved agents (they are tracked in git, so run `git pull` there; the simulation dataset agent from step 5 must be copied or committed too), `tools/keys_secrets.py`, Docker and the controller.

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
| 5 | `model_dspy` | `name: efficientnet_b0.py n_class: 10 in_features: 1` |
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
model: {name: plugins/models/pre-trained_models/cnn_architectures/efficientnet_b0.py, n_classes: 10, in_features: 1}
```

(`local_dataset` has `train.npz` / `test.npz`; `logic` has `mode: directory`, `dir: data`; there is **no** `centralized` key.)

**Fix `clients_dir` before you type `yes`.** The dataset agent writes the client list with a space, `clients_dir: c1, c2`. The app splits it on `,` without trimming, so the second client folder would be read as `" c2"` and not be found. The executor has already copied the config to `data/tools/config/config.yml` by the time it asks the question, and that staged copy is the one FeatureCloud uses. While the question is waiting:

1. Open `data/tools/config/config.yml`.
2. Change `clients_dir: c1, c2` to `clients_dir: c1,c2` and save.
3. Check that the model line now shows only `efficientnet_b0.py` (the executor removed the `plugins/...` path) and that `efficientnet_b0.py` and `efficientnet_b0_rwightman-7f5810bc.pth` are next to it.

Then type `yes` to run FeatureCloud. The command must contain `--client-dirs sample_data_cutomized/MNIST/cross_validation_sample` (one path) and `--generic-dir tools/config`. Type `no` to stop after the config exists.

### What to expect

- One client container (not two). The log shows `Simulation mode`, `Communication round: #1 ...`, `Client #1: Local training`, `Client #2: Local training`. With `max_iter` 1 there is only this one round. There are no `Local Update` / `Global Aggregation` state lines.
- FedAvg runs per fold: `c1` fold 0 + `c2` fold 0 give the new global model for fold 0; the same for fold 1. Fold 0 and fold 1 are never mixed.
- When the run finishes, the executor prints accuracy per simulated client and fold; run `python tools/report/report_accuracy.py` to print it again. Result layout (one zip):

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
