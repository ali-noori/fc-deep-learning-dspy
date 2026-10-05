# Run app with DSPy agents (PowerShell)

Minimal test guide for the five config agents. For running the app **without** agents, use `Run_App.md` at the project root.

---

## Once

```powershell
& "C:\UH\Thesis\FC\fc-deep-learning-env-zenbook\Scripts\Activate.ps1"
cd "C:\UH\Thesis\FC\fc-deep-learning-master\fc-deep-learning-master"
```

Keys live in `tools/keys_secrets.py` (including `COSY_API_KEY` for COSY.BIO `llama4:latest`).

**Full pipeline:** start agent 1. After it picks a mode, it starts the other four agents, then builds and runs the config.

**Test one agent:** run only that agent’s command below, then type the example for the mode you want.

---

## 1. `execution_mode_dspy`

First run (compile, then save the compiled agent):

```powershell
python -m tools.execution_mode_dspy.cli --recompile=True
```

Later runs (skip compile, load the saved agent):

```powershell
python -m tools.execution_mode_dspy.cli --recompile=False
```

Saved file: `tools/execution_mode_dspy/compiled_agent/compiled_agent.json`.  
`--recompile=False` fails if that file does not exist yet — run `--recompile=True` once first.

If you omit `--recompile`, it falls back to `FC_DSPY_RECOMPILE`, then `RECOMPILE` in `tools/execution_mode_dspy/config.py`.

Type one of:

| Mode | Example input |
|------|----------------|
| Federated | `federated` |
| Simulation | `simulation` |
| Centralized | `centralized` |

This agent then continues to dataset → hyper-params → model → trainer.  
`--recompile=True` / `--recompile=False` is forwarded automatically to dataset, hyper-params, model, and trainer.

---

## 2. `dataset_dspy`

Needs `--execution-mode`. The question (and fields) change per mode.

Standalone (optional `--recompile`; omitted uses dataset `config.py` / `FC_DSPY_RECOMPILE`):

```powershell
python -m tools.sub_config_dspy.dataset_dspy.cli --execution-mode federated --recompile=True
python -m tools.sub_config_dspy.dataset_dspy.cli --execution-mode simulation --recompile=False
python -m tools.sub_config_dspy.dataset_dspy.cli --execution-mode centralized --recompile=True
```

Saved files under `tools/sub_config_dspy/dataset_dspy/compiled_agent/`:
`compiled_agent_federated.json`, `compiled_agent_simulation.json`, `compiled_agent_centralized.json`.

**Federated**

```text
data_dir1: sample_data/c1
data_dir2: sample_data/c2
train_dataset_file_name: train.npz
test_dataset_file_name: test.npz
logic_dir: data
```

**Simulation**

```text
data_dir: sample_data_simulation
client_dir1: c1
client_dir2: c2
train_dataset_file_name: train.npz
test_dataset_file_name: test.npz
logic_dir: data
```

**Centralized**

```text
data_dir: sample_data_centralized
train_dataset_file_name: train.npz
test_dataset_file_name: test.npz
logic_dir: data
```

---

## 3. `hyper_params_dspy`

Same question in all three modes.

```powershell
python -m tools.sub_config_dspy.hyper_params_dspy.cli --recompile=True
```

Saved file: `tools/sub_config_dspy/hyper_params_dspy/compiled_agent/compiled_agent.json`.

Use this for federated, simulation, or centralized:

```text
max_iter: 10
n_class: 10
federated_model: FedAvg
```

---

## 4. `model_dspy`

Same question in all three modes.

```powershell
python -m tools.sub_config_dspy.model_dspy.cli --recompile=True
```

Saved file: `tools/sub_config_dspy/model_dspy/compiled_agent/compiled_agent.json`.

Use this for federated, simulation, or centralized:

```text
name: CNN
n_class: 10
in_features: 1
```

---

## 5. `trainer_dspy`

Same question in all three modes.

```powershell
python -m tools.sub_config_dspy.trainer_dspy.cli --recompile=True
```

Saved file: `tools/sub_config_dspy/trainer_dspy/compiled_agent/compiled_agent.json`.

Use this for federated, simulation, or centralized:

```text
name: BasicTrainer
data_loader: ImageLoader
loss.name: CrossEntropyLoss
n_class: 10
```

---

Success looks like `Agent: Okay!` plus `backend=...` / `model=...`. Dataset, hyper-params, model, and trainer also print an `output=` YAML path.
