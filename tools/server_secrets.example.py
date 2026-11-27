# Copy this file to tools/server_secrets.py and fill in your own values.
# tools/server_secrets.py is gitignored and dockerignored and must not be committed.

SERVER_USER = ""
SERVER_HOST = ""
SERVER_PASSWORD = ""  # leave empty to type the password in the terminal every time

# Project folder on the server, e.g. "~/fc-deep-learning-dspy".
SERVER_PROJECT_DIR = ""

# Python environment on the server (it is activated before the workflow starts), e.g. "~/venv-toad".
SERVER_VENV_DIR = ""

# Command that runs the rest of the workflow on the server (from the project folder, inside the environment), e.g.
# "python -m tools.execution_mode_dspy.cli --from-gui --recompile=False"
SERVER_WORKFLOW_COMMAND = ""
