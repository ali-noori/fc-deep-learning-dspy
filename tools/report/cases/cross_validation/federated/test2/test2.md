# Test 2 — Federated cross-validation, MobileNetV3-Small

Run the agentic layer to build a config and train it with FeatureCloud. Two ways to start: **Mode 1 (GUI)** or **Mode 2 (step-by-step agents)**. Both give the same config.

| Setting | Value |
| --- | --- |
| Dataset | `data/sample_data_cutomized/MNIST/cross_validation_sample` (clients `c1`, `c2`; folds `data/0`, `data/1`) |
| Execution mode | federated |
| Model | MobileNetV3-Small — `plugins/models/pre-trained_models/cnn_architectures/mobilenetv3_small.py` |
| Rounds / classes / aggregator | `max_iter` 2, 10 classes, `FedAvg` |
| Trainer / loader / loss | `BasicTrainer`, `ImageLoader`, `CrossEntropyLoss` |
| Train config | verbose false, batch 32, test batch 32, epochs 1, lr 0.1, batch_count 1 |
| Model input | `in_features` 1 (grayscale; the plugin converts it to 3 channels) |

Paths given to the agents are relative to the `data/` folder, so they do **not** start with `data/`.

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

4. All saved agents already exist in the repo, including `tools/sub_config_dspy/train_config_dspy/compiled_agent/compiled_agent.json`. Both modes load them with `--recompile=False`, so no compile step is needed.

MobileNetV3-Small ImageNet weights are downloaded by torchvision inside each client container on the first run, so the containers need internet access. Images are resized to 224×224, so training is slow on CPU. Use `max_iter` 1 for a quick smoke run.

## Mode 1 — GUI

```powershell
python -m tools.intake.intake_dspy.cli --recompile=False
```

**Window 1 — "Please enter you prompt".** Paste, then click **Next**:

```text
Federated learning on two clients: sample_data_cutomized/MNIST/cross_validation_sample/c1 and sample_data_cutomized/MNIST/cross_validation_sample/c2. Train file is train.npz, test file is test.npz. The split folder under mnt/input is data.

max_iter: 2. n_class: 10. federated_model: FedAvg.

name: BasicTrainer. data_loader: ImageLoader. loss.name: CrossEntropyLoss. n_class: 10.

verbose: false. batch_size: 32. test_batch_size: 32. epochs: 1. lr: 0.1. batch_count: 1.
```

**Window 2 — "Please enter your prompt for model section".** Paste:

```text
mobilenetv3_small.py. number of input is 1 and number of class is 10.
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

The server does not need `tkinter`. It does need the repo, the dataset folders, the saved agents (they are tracked in git, so run `git pull` there), `tools/keys_secrets.py`, Docker and the controller.

## Mode 2 — Step-by-step agents

```powershell
python -m tools.execution_mode_dspy.cli --recompile=False
```

Each agent asks one question. Paste the answer after `You:`.

| # | Agent / question | Paste |
| --- | --- | --- |
| 1 | `execution_mode_dspy` | `federated` |
| 2 | `dataset_dspy` | `I use two clients: sample_data_cutomized/MNIST/cross_validation_sample/c1 and sample_data_cutomized/MNIST/cross_validation_sample/c2. Train file is train.npz, test file is test.npz. The split folder under mnt/input is data.` |
| 3 | `hyper_params_dspy` | `max_iter: 2 n_class: 10 federated_model: FedAvg` |
| 4 | `Do you want to generate a new model? (Yes/No)` | `No` |
| 5 | `model_dspy` | `name: mobilenetv3_small.py n_class: 10 in_features: 1` |
| 6 | `trainer_dspy` | `name: BasicTrainer data_loader: ImageLoader loss.name: CrossEntropyLoss n_class: 10` |
| 7 | `train_config_dspy` | `verbose: false batch_size: 32 test_batch_size: 32 epochs: 1 lr: 0.1 batch_count: 1` |

Then `config_builder` runs by itself and `config_executor` asks `Are you sure you want to run this command? (yes/no)`. Go to **Check and run**.

## Check and run

Every agent must print `Agent: Okay!`. If one prints an error, stop and start again from the beginning. A failed agent does not stop the workflow, and old output files may be reused silently.

Before answering the question, open `tools/config_builder/output/federated/federated_config.yml`. It must contain:

```yaml
fed_hyper_params: {max_iter: 2, n_classes: 10, federated_model: FedAvg}
trainer: {name: BasicTrainer, data_loader: ImageLoader, loss: {name: CrossEntropyLoss}}
train_config: {verbose: false, batch_size: 32, test_batch_size: 32, epochs: 1, lr: 0.1, batch_count: 1}
model: {name: plugins/models/pre-trained_models/cnn_architectures/mobilenetv3_small.py, n_classes: 10, in_features: 1}
```

(`local_dataset` has `train.npz` / `test.npz`; `logic` has `mode: directory`, `dir: data`.)

Type `yes` to run FeatureCloud. When the run finishes, the executor prints accuracy per client and fold; run `python tools/report/report_accuracy.py` to print it again. Type `no` to stop after the config exists.

If Windows blocks FeatureCloud, see `tools/report/troubleshoot.md`.

## Result

Not recorded yet.
