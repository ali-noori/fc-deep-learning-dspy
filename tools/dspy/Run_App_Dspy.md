# Run DSPy config tool (PowerShell)

This turns a **plain-English request** into:

`tools/output/config/generated/config.generated.dspy.yml`

(using `config.maximum.yml` at the repo root as the base.)

---

## 1. Activate venv and go to the repo

```powershell
& "C:\FC\fc-deep-learning-env\Scripts\Activate.ps1"
cd "C:\FC\fc-deep-learning-master\fc-deep-learning-master"
```

---

## 2. Pick how the LM runs: Groq (cloud) or Ollama (local)

### Option A — Groq (default if a key is set)

1. Put your API key in **`tools/local_secrets.py`** (copy from `tools/local_secrets.example.py`), **or** set **`GROQ_API_KEY`** / **`FC_DSPY_GROQ_API_KEY`** in the environment.
2. Do **not** set `FC_DSPY_FORCE_OLLAMA` unless you want to force local Ollama instead.

### Option B — Ollama (local)

Use this when you have no Groq key, or when you want local inference even though a Groq key exists.

**Prerequisites**

1. **Install [Ollama](https://ollama.com)** and start it (tray app or service) so it listens on the API (default **`http://localhost:11434`**).
2. **Pull a model** that matches what the tool expects. By default the code uses **`llama3.1`**:

   ```powershell
   ollama pull llama3.1
   ```

   If you use another tag, set **`FC_DSPY_OLLAMA_MODEL`** to that name (example: `llama3.2`).

3. Optional: if Ollama runs elsewhere or on another port (e.g. Docker), set:

   ```powershell
   $env:FC_DSPY_OLLAMA_BASE_URL = "http://localhost:11434"
   $env:FC_DSPY_OLLAMA_MODEL = "llama3.1"
   ```

4. To **ignore Groq** and always use Ollama when both could work:

   ```powershell
   $env:FC_DSPY_FORCE_OLLAMA = "1"
   ```

The tool checks that **`GET /api/tags`** works on your base URL before it uses Ollama.

---

## 3. Run the CLI

Replace the sentence with what you want:

```powershell
python -m tools.dspy.cli --request "Set fc_deep model to mlp.py" --no-bootstrap
```

**`--no-bootstrap`** is recommended (fewer LM calls during DSPy setup; avoids Groq rate limits). You can omit it if you want full `BootstrapFewShot` (slower, more API usage).

---

## 4. Output

Generated file:

`tools/output/config/generated/config.generated.dspy.yml`
