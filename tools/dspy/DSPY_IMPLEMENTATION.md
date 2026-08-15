# DSPy part of the app — implementation guide (junior-friendly)

This document explains **only** the code under `tools/dspy/`: how natural language becomes a YAML config file, and how each file participates.

---

## 1. What problem does this solve?

Researchers describe what they want in **plain English** (e.g. “use MLP, FedAvg, focal loss”). The FeatureCloud app still needs a **structured `config.yml`** with exact keys like `fc_deep.model.name`, `fed_hyper_params.federated_model`, etc.

The DSPy layer’s job is:

1. **Understand** the sentence with a **language model** (Groq or Ollama).
2. **Emit a small JSON object** with fields we care about (`model_name`, `trainer_name`, …).
3. **Merge** those fields into a **safe template** (`config.maximum.yml`) so we do not ask the LM to invent the whole file from scratch.

**Important split of responsibility**

| Layer | Responsibility |
|--------|----------------|
| **LM (via DSPy)** | Interpret vague language → pick strings that match allowed artifact names. |
| **Python (merge)** | Copy template, overwrite only specific keys; no regex “if user said CNN then cnn.py”. |
| **`discover_repo_artifacts`** | Scan `plugins/` (and a few built-ins) so the LM sees **real** filenames in the prompt. |

---

## 2. End-to-end flow (one CLI run)

```
cli.py (main)
  → ConfigAgentPipeline(repo_root, skip_bootstrap)
       → discover_repo_artifacts(repo_root)     # lists models, trainers, …
       → DSPyModules(artifacts, skip_bootstrap) # configures LM + predictor
  → pipeline.run(request)
       → agent.extract_intent(request)           # LM call → UserIntent
       → load_default_template_config()         # reads config.maximum.yml
       → dspy_config_output(intent, base_cfg)  # deepcopy + patch
       → yaml.safe_dump → write_outputs         # config.generated.dspy.yml
```

Read this as: **artifacts → predictor → intent → template merge → file on disk**.

---

## 3. File-by-file roles

### `cli.py`

- **Entry point** for humans: `python -m tools.dspy.cli --request "..."`.
- Parses arguments, builds `ConfigAgentPipeline`, calls `run()`, prints a small JSON summary (paths only).
- **`--no-bootstrap`**: tells the pipeline to **skip** `BootstrapFewShot` (see below) so you get **one** LM call per run instead of many during “compile” (helps Groq rate limits).

### `pipeline.py` — `ConfigAgentPipeline`

- **Orchestrator**: wires discovery, DSPy, template load, merge, dump, write.
- **`run(request)`** is the single method you need to understand: intent → plan → YAML string → disk.

### `optimization.py`

- **`discover_repo_artifacts`**: walks `plugins/models`, `plugins/trainers`, `plugins/aggregators`, `plugins/loss`, `plugins/dataloaders`, plus **built-in strings** that are not files (`FedAvg`, `CrossEntropyLoss`, `ImageLoader`, …). Those lists are turned into strings and passed into the LM as `known_models`, `known_aggregators`, etc.
- **`load_default_template_config`**: loads `config.maximum.yml` (or another template path) into a nested dict.
- **`dspy_config_output`**: **deterministic merge** — copies the template and sets only:

  - `fc_deep.model.name` ← `intent.model_name`
  - `fc_deep.trainer.name` ← `intent.trainer_name`
  - `fc_deep.fed_hyper_params.federated_model` ← `intent.aggregator_name`
  - `fc_deep.trainer.loss.name` ← `intent.loss_name`
  - `fc_deep.trainer.data_loader` ← `intent.data_loader_name`

  Empty intent fields mean **leave template value**.

- **`write_outputs`**: writes `config.generated.dspy.yml`.

### `modules.py` — heart of DSPy

- **`IntentExtraction` (dspy.Signature)**  
  Declares **inputs** (user sentence + `known_*` lists) and **one output**: `intent_json`, a **single-line JSON string**.  
  The **`desc=`** text on each field is part of the **prompt** to the LM: it teaches **key names** and **rules** (e.g. use `data_loader_name`, not only `task`).

- **`DSPyModules`**
  - **`__init__`**: picks **Groq** if a key exists and `FC_DSPY_FORCE_OLLAMA` is off; else **Ollama** if `/api/tags` responds. Calls `dspy.configure(lm=...)`, builds `dspy.Predict(IntentExtraction)`, optionally wraps it with **`BootstrapFewShot`**.
  - **`extract_intent`**: runs the predictor, `json.loads` on `intent_json`, maps to **`UserIntent`**. **`_intent_str`** accepts alternate keys (`loss` vs `loss_name`) because real LMs are inconsistent.
  - **`_wrap_student`**: if not skipping bootstrap, **`BootstrapFewShot.compile(...)`** runs: it uses **`_bootstrap_examples()`** (input/output pairs) and **`_intent_metric`** to score predictions and improve the student module (extra LM calls at compile time).
  - **`_bootstrap_examples`**: **few-shot dataset** — same `known_*` context as production, gold `intent_json` for each example.

### `schemas.py`

- **`UserIntent`**: plain dataclass — what we extracted (strings + `raw_request`).
- **`AgentPlan` / `GenerationResult`**: container for merged config and final YAML text.

### `lm_secrets.py` / `ollama_settings.py`

- **Credentials and URLs**: Groq API key from env or `tools/local_secrets.py`; Ollama base URL and model name from env. No DSPy logic here.

---

## 4. DSPy concepts used here (quick glossary)

- **Signature** — Typed “function spec”: inputs and outputs of the LM step; DSPy turns it into prompts and parses structured outputs.
- **Predict** — One forward pass: fill outputs from inputs using the LM.
- **Example / trainset** — Labeled pairs for optimization; each example marks which fields are **inputs** with `.with_inputs(...)`.
- **BootstrapFewShot** — DSPy optimizer that tries to improve the student on the trainset using a **metric** (`_intent_metric`: overlap of JSON keys between gold and prediction).
- **Student** — The `Predict(IntentExtraction)` module; after compile, `self.predictor` might be the compiled (optimized) module.

---

## 5. Why `--no-bootstrap` exists

`BootstrapFewShot` **compile** can trigger **many** LM calls (slow, hits **Groq TPM** limits).  
`--no-bootstrap` (or env `FC_DSPY_SKIP_BOOTSTRAP=1`) uses the raw **`Predict`** module: **one** LM call per `extract_intent`, with quality relying more on **signature wording** and **`_intent_str` fallbacks**.

---

## 6. Limits and thesis-friendly wording

- The LM does **not** validate FeatureCloud compatibility (e.g. FedAvg vs `global_updates`). See **`CONFIG_INCOMPATIBILITIES.md`**.
- The LM does **not** rewrite `local_dataset` or `train_config` in the current merge; only the five override paths in **`dspy_config_output`**.
- **Hallucinated** names can still appear if they are not caught by later validation (future work).

---

## 7. Where to read the code first

1. `pipeline.py` — `ConfigAgentPipeline.run`
2. `modules.py` — `DSPyModules.extract_intent` and `IntentExtraction`
3. `optimization.py` — `dspy_config_output` and `discover_repo_artifacts`

This matches the mental model: **discover → predict intent → merge template → write file**.
