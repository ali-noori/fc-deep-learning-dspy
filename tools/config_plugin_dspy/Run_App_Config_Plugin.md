# Config plugin DSPy (`tools/config_plugin_dspy`)

Development-time agent that maps a natural-language request to a FeatureCloud `config.yml`, using the same merge rules as `tools/dspy` but with the cleaner layout of `tools/model_plugin_dspy_v2`.

## Quick start

From the project root (with `fc-deep-learning-env` activated and `dspy-ai` installed):

```powershell
cd C:\FC\fc-deep-learning-master\fc-deep-learning-master
python -m tools.config_plugin_dspy.cli --request "Use cnn.py with FedMMb trainer" --no-bootstrap
```

Output: `tools/output/config/generated/config.generated.plugin.yml` (merged from `config.maximum.yml`).

## Architecture (mirrors `model_plugin_dspy_v2`)

| Module | Role |
|--------|------|
| `config.py` | Paths, LM model names, `Settings` |
| `lm_router.py` | Groq → Gemini → Cohere → Mistral → Ollama cascade |
| `intent_extractor.py` | DSPy signature + `ConfigIntentExtractorModule` |
| `optimizer.py` | `BootstrapFewShot` compile |
| `fewshot_examples.py` / `trainset.py` | Few-shot training pairs |
| `metric.py` | Intent JSON key overlap score |
| `artifact_discovery.py` | Lists plugins/builtins for `known_*` prompt fields |
| `config_merger.py` | Patch `fc_deep.model.name`, `trainer`, `fed_hyper_params`, etc. |
| `pipeline.py` | Orchestration |
| `cli.py` | `python -m tools.config_plugin_dspy.cli` |

## API keys

Set any of these in the environment or `tools/local_secrets.py` (not committed):

- `GROQ_API_KEY` / `FC_DSPY_GROQ_API_KEY`
- `GEMINI_API_KEY`, `COHERE_API_KEY`, `MISTRAL_API_KEY`

Cloud providers without a key are skipped; Ollama is always tried last (`FC_DSPY_OLLAMA_BASE_URL`, `FC_DSPY_OLLAMA_MODEL`).

## Flags

- `--request` / `--prompt` — natural language (required)
- `--no-bootstrap` — skip few-shot compilation (faster, fewer API calls)
- `--output-dir` — custom output directory (filename stays `config.generated.plugin.yml`)
- `--repo-root` — project root (default: two levels above this package)

## Using the generated config in FeatureCloud tests

Copy or point `--generic-dir` at a folder that contains your generated file as `config.yml`, or merge fields manually into your sample data `generic/config.yml`.
