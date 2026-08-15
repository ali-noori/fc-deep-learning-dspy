# Model Plugin DSPy

1. Reads **one** `--request` string (all configuration must be in that text).
2. Uses DSPy **ChainOfThought** to produce `architecture_json`.
3. Writes **`plugins/models/customized_model.py`** (repo root is inferred from this package location).

## Run

```powershell
cd "C:\FC\fc-deep-learning-master\fc-deep-learning-master"
python -m tools.model_plugin_dspy.cli --request "YOUR FULL PROMPT HERE"
```

Put in the request, for example:

- Image shape, channel count, `n_classes` / class count, flattened `input_dim` if needed (e.g. 784 for 28×28).
- Layer list (Conv2d with `in_features→…` for the first conv when channels come from `Model`’s `in_features`).
- Or transfer backbone name: `MobileNetV3-Small`, `EfficientNet-B0`, `SqueezeNet`, `ResNet18`, etc.

## Optional eval harness

```powershell
python -m tools.model_plugin_dspy.eval
python -m tools.model_plugin_dspy.eval --json
```

## Backend

- Default **Groq** if `GROQ_API_KEY` / `FC_DSPY_GROQ_API_KEY` is set  
- Else **Ollama** if reachable  
- `FC_MODEL_PLUGIN_DSPY_SKIP_BOOTSTRAP=1` skips few-shot compile  
- `FC_DSPY_FORCE_OLLAMA=1` forces Ollama
