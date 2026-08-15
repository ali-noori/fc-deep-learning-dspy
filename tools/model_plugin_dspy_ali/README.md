# Model Plugin DSPy Ali

Small DSPy implementation for generating PyTorch model architectures from a prompt.

Generated models are written to:

```text
plugins/models/generated_architecture/customized_model_ali.py
```

The pipeline uses:

- `dspy.Signature` for the prompt and architecture output.
- `dspy.ChainOfThought` for better reasoning before producing code.
- Ollama as the only language-model backend.

## Ollama

Run an Ollama container and expose the API on port `11434`:

```powershell
docker run -d --name ollama -p 11434:11434 ollama/ollama
```

Pull the model you want to use, for example:

```powershell
docker exec ollama ollama pull llama3.1
```

The defaults are:

- `FC_DSPY_OLLAMA_BASE_URL=http://localhost:11434`
- `FC_DSPY_OLLAMA_MODEL=llama3.1`

## Run

```powershell
python -m tools.model_plugin_dspy_ali.cli --request "Create a CNN architecture for image classification..."
```

Use `--json` to include backend metadata:

```powershell
python -m tools.model_plugin_dspy_ali.cli --request "Create a CNN architecture..." --json
```
