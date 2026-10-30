# Run app with DSPy agents (PowerShell)

Minimal test guide for the config agents. For running the app **without** agents, use `Run_App.md` at the project root.

---

## Once

```powershell
& "C:\UH\Thesis\FC\fc-deep-learning-env-zenbook\Scripts\Activate.ps1"
cd "C:\UH\Thesis\FC\fc-deep-learning-master\fc-deep-learning-master"
```

Keys live in `tools/keys_secrets.py` (including `COSY_API_KEY` for COSY.BIO `llama4:latest`).

**Full pipeline:** start agent 1. After it picks a mode, it starts the remaining sub-agents (including train_config), then builds and runs the config.

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

Type a short sentence (the agent maps it to a mode). Examples:

| Mode | Example input |
|------|----------------|
| Federated | `Two hospitals should train on their own machines and only share model weights, not the images.` |
| Simulation | `I am not ready for real sites yet — fake two clients on this laptop so I can debug federated averaging.` |
| Centralized | `Skip federation. Train everything on this one machine as a normal local experiment.` |

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

Still the **small sample folders** (not the large customized set). Phrase it like a person talking, but keep the real path names.

**Federated**

```text
We have two partners. Hospital A’s files are under sample_data/c1 and hospital B’s under sample_data/c2. Inside each fold the training set is train.npz and the hold-out set is test.npz. The split folders live in a directory called data.
```

**Simulation**

```text
Use the local tree sample_data_simulation. Pretend there are two clients, folders c1 and c2. Same filenames as usual: train.npz for training, test.npz for evaluation, and the folds sit in data.
```

**Centralized**

```text
Only one machine: take the sample under sample_data_centralized. Train on train.npz, evaluate on test.npz, and look for the fold folders inside data.
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
Keep it light for a laptop demo: ten rounds (or epochs in centralized), ten digit classes, and the built-in FedAvg aggregator — not a custom .py plugin.
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
Please use the built-in CNN, not a plugin file. It is grayscale (one input channel) and there are ten classes, like MNIST digits.
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
Use the built-in BasicTrainer and ImageLoader, CrossEntropyLoss for a ten-class problem. No custom trainer or loader plugins.
```

---

## 6. `train_config_dspy`

Same question in all three modes. Runs after `trainer_dspy` in the full pipeline.

```powershell
python -m tools.sub_config_dspy.train_config_dspy.cli --recompile=True
```

Saved file: `tools/sub_config_dspy/train_config_dspy/compiled_agent/compiled_agent.json`.

Use this for federated, simulation, or centralized:

```text
Keep it light for a laptop: no verbose logs, train and test batch 32, one epoch, learning rate 0.1, and batch_count 1.
```

---

Success looks like `Agent: Okay!` plus `backend=...` / `model=...`. Dataset, hyper-params, model, trainer, and train_config also print an `output=` YAML path.
