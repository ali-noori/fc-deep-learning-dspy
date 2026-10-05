# Troubleshoot — FeatureCloud blocked on Windows

## Problem

Near `featurecloud test start` you may see:

- **WinError 4551** — Application Control blocked a file  
- A **Smart App Control** popup (unsigned or untrusted `.exe` / driver)

This is **not** a bad `config.yml`. Windows blocked a program the run tried to start. Typical files:

- `...\Scripts\featurecloud.exe` (pip launcher)
- Docker or another helper `.exe`

Python can keep running; FeatureCloud training does not start until that file is allowed.

The executor uses `python -m FeatureCloud.api.cli` so it should not launch `featurecloud.exe`. If the popup remains, Smart App Control is blocking **whatever that command starts next** (often Docker).

`C:\Installation\python` is more likely to hit this than the Zenbook venv.

## What to do

1. **See the blocked path**  
   Windows Security → **Virus & threat protection** → **Protection history**. Open the latest Smart App Control / Blocked item. Note the full path.

2. **Turn Smart App Control off** (usual fix on a personal thesis PC; there is no simple per-app allow).  
   Windows Security → **App & browser control** → **Smart App Control settings** → **Off**.  
   Or: Settings → Privacy & security → Windows Security → App & browser control.  
   If **Off** is greyed out, the PC is managed; IT must allow the tools.

3. **Restart Windows.**

4. **Run again from the Zenbook venv:**

   ```powershell
   & "C:\UH\Thesis\FC\fc-deep-learning-env-zenbook\Scripts\Activate.ps1"
   cd "C:\UH\Thesis\FC\fc-deep-learning-master\fc-deep-learning-master"
   python -m tools.config_executor.cli --execution-mode federated
   ```

   Type `yes`. Controller: `http://localhost:8000`. Start Docker / the FeatureCloud controller if the command cannot connect.

5. **If it still pops up:** stay on the Zenbook venv; if the blocked file is Docker, start Docker Desktop first; if Off is unavailable, use another PC or ask IT.
