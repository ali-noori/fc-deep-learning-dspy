# Setup before the test scenarios
## A. Local environment (Windows, PowerShell)

1. Activate the environment:

   ```powershell
   & "C:\UH\Thesis\FC\fc-deep-learning-env-zenbook\Scripts\Activate.ps1"
   cd "C:\UH\Thesis\FC\fc-deep-learning-master\fc-deep-learning-master"
   ```

2. Build the app image (Docker Desktop must be running):

   ```powershell
   python -m FeatureCloud.api.cli app build . featurecloud.ai/fc_deep_v1
   ```

3. Start the controller from the project directory:

   ```powershell
   python -m FeatureCloud.api.cli controller start --port=8000
   ```

Check: `docker images featurecloud.ai/fc_deep_v1` shows the image and `docker ps` shows `fc-controller`.

## B. Server environment (Linux, bash)

1. Activate the environment:

   ```bash
   cd ~/fc-deep-learning-dspy
   source ~/venv-toad/bin/activate
   ```

   `tools/keys_secrets.py` is not in git. Copy it once from your laptop into `~/fc-deep-learning-dspy/tools/` on the server.

2. Build the app image:

   ```bash
   python -m FeatureCloud.api.cli app build . featurecloud.ai/fc_deep_v1
   ```

3. Start the controller from the project directory:

   ```bash
   python -m FeatureCloud.api.cli controller start --port=8000
   ```

Check: `docker images featurecloud.ai/fc_deep_v1` shows the image and `docker ps` shows `fc-controller`.
