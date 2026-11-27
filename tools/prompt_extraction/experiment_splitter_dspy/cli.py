"""CLI for experiment_splitter_dspy. GUI by default; --no-gui reads PowerShell input."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from tools.prompt_extraction.gui.experiment_splitter_dspy_prompt import ask_experiment_splitter_dspy_prompt
from tools.prompt_extraction.gui.model_extractor_dspy_prompt import ModelPrompt, ask_model_extractor_dspy_prompt
from tools.prompt_extraction.intake_process import INTAKE_PROCESS_PATH, build_final_intake
from tools.prompt_extraction.model_extractor_dspy.briefs import briefs_path as model_briefs_path
from tools.prompt_extraction.model_extractor_dspy.config import load_settings as load_model_settings
from tools.prompt_extraction.model_extractor_dspy.pipeline import ModelExtractorPipeline

from .briefs import briefs_path
from .config import load_settings, resolve_recompile
from .pipeline import ExperimentSplitterPipeline


def _parse_bool(raw: str) -> bool:
    cleaned = raw.strip().lower()
    if cleaned in ("true", "1", "yes"):
        return True
    if cleaned in ("false", "0", "no"):
        return False
    raise argparse.ArgumentTypeError(f"Expected True or False, got {raw!r}.")


def _read_prompt(use_gui: bool, question: str) -> str | None:
    if not use_gui:
        print(f"Agent: {question}")
        return input("You: ").strip()
    return ask_experiment_splitter_dspy_prompt()


def main() -> int:
    parser = argparse.ArgumentParser(description="Experiment splitter agent.")
    parser.add_argument(
        "--no-gui",
        action="store_true",
        help="Read the request from PowerShell instead of the window.",
    )
    parser.add_argument(
        "--recompile",
        type=_parse_bool,
        default=None,
        metavar="{True,False}",
        help="--recompile=True: compile and save. --recompile=False: load saved agent.",
    )
    args = parser.parse_args()

    settings = load_settings()
    recompile = resolve_recompile(args.recompile)
    pipeline = ExperimentSplitterPipeline(settings=settings, recompile=recompile)

    try:
        user_input = _read_prompt(not args.no_gui, pipeline.EXPERIMENT_SPLITTER_QUESTION)
    except EOFError:
        print("Agent: no prompt was entered.", file=sys.stderr)
        return 1
    if user_input is None:
        return 0
    if not user_input.strip():
        print("Agent: please enter a prompt before continuing.", file=sys.stderr)
        return 1

    try:
        result = pipeline.run(user_input)
    except ValueError as exc:
        print(f"Agent: {exc}", file=sys.stderr)
        return 1
    except RuntimeError as exc:
        print(f"Agent: {exc}", file=sys.stderr)
        return 1

    print(f"Agent: {result['agent_reply']}")
    print(f"Agent: wrote {briefs_path()}")
    print(
        f"(output={result['output_path']}, backend={result['backend']}, "
        f"model={result['model']})"
    )
    if not args.no_gui:
        return _run_model_extractor(ask_model_extractor_dspy_prompt(), recompile)
    return 0


# One line of Python used as the ssh password helper. It answers password prompts only, never a
# host-key question, and reads the password from an environment variable of the ssh process.
_ASKPASS_CODE = (
    "import os,sys; a=' '.join(sys.argv[1:]).lower(); "
    "sys.stdout.write(os.environ['FC_SERVER_PASSWORD']) if 'password' in a or 'passphrase' in a else sys.exit(1)"
)


def _ask_run_location() -> str:
    """Ask where the rest of the workflow runs. Enter (or no keyboard input) means local."""
    print("Where do you want to run the workflow?")
    print("  1) Local (this computer)")
    print("  2) Server")
    while True:
        try:
            answer = input("Choice [1]: ").strip().lower()
        except EOFError:
            return "local"
        if answer in ("", "1", "local", "l"):
            return "local"
        if answer in ("2", "server", "s"):
            return "server"
        print("Please enter 1 or 2.")


def _load_server_settings() -> dict[str, str] | None:
    """Read the server login from tools/server_secrets.py (gitignored)."""
    try:
        from tools import server_secrets
    except ImportError:
        print(
            "Agent: tools/server_secrets.py was not found. "
            "Copy tools/server_secrets.example.py to tools/server_secrets.py and fill it in.",
            file=sys.stderr,
        )
        return None
    settings = {
        "user": str(getattr(server_secrets, "SERVER_USER", "")).strip(),
        "host": str(getattr(server_secrets, "SERVER_HOST", "")).strip(),
        "password": str(getattr(server_secrets, "SERVER_PASSWORD", "")),
        "project_dir": str(getattr(server_secrets, "SERVER_PROJECT_DIR", "")).strip(),
        "venv_dir": str(getattr(server_secrets, "SERVER_VENV_DIR", "")).strip(),
        "workflow_command": str(getattr(server_secrets, "SERVER_WORKFLOW_COMMAND", "")).strip(),
    }
    required = (
        ("SERVER_USER", "user"),
        ("SERVER_HOST", "host"),
        ("SERVER_PROJECT_DIR", "project_dir"),
        ("SERVER_VENV_DIR", "venv_dir"),
        ("SERVER_WORKFLOW_COMMAND", "workflow_command"),
    )
    missing = [name for name, key in required if not settings[key]]
    if missing:
        print(f"Agent: {', '.join(missing)} must be set in tools/server_secrets.py.", file=sys.stderr)
        return None
    return settings


def _find_ssh() -> str | None:
    """Prefer the Windows OpenSSH client (its -t works in PowerShell); otherwise any ssh on the PATH."""
    windows_ssh = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32" / "OpenSSH" / "ssh.exe"
    if os.name == "nt" and windows_ssh.is_file():
        return str(windows_ssh)
    return shutil.which("ssh")


def _write_askpass_helper(directory: Path) -> Path | None:
    """Create the small helper that lets ssh read the password from the environment (no password on disk)."""
    try:
        if os.name == "nt":
            helper = directory / "askpass.cmd"
            helper.write_text(f'@"{sys.executable}" -c "{_ASKPASS_CODE}" %*\r\n', encoding="ascii")
        else:
            helper = directory / "askpass.sh"
            helper.write_text(f'#!/bin/sh\nexec "{sys.executable}" -c "{_ASKPASS_CODE}" "$@"\n', encoding="ascii")
            helper.chmod(0o700)
    except (OSError, UnicodeEncodeError):
        return None
    return helper


def _run_on_server() -> int:
    """Copy intake_process.json to the server and run the rest of the workflow there (same command as local)."""
    settings = _load_server_settings()
    if settings is None:
        return 1
    ssh = _find_ssh()
    if ssh is None:
        print("Agent: ssh was not found. Install the OpenSSH client, or choose Local.", file=sys.stderr)
        return 1

    target = f"{settings['user']}@{settings['host']}"
    project_dir = settings["project_dir"].rstrip("/")
    remote_output_dir = f"{project_dir}/tools/prompt_extraction/output"
    upload_command = f"mkdir -p {remote_output_dir} && cat > {remote_output_dir}/intake_process.json"
    run_command = f"cd {project_dir} && source {settings['venv_dir'].rstrip('/')}/bin/activate && {settings['workflow_command']}"

    with tempfile.TemporaryDirectory() as tmp:
        env = os.environ.copy()
        ssh_options: list[str] = []
        if settings["password"]:
            helper = _write_askpass_helper(Path(tmp))
            if helper is not None:
                env.update(SSH_ASKPASS=str(helper), SSH_ASKPASS_REQUIRE="force", FC_SERVER_PASSWORD=settings["password"])
                ssh_options = ["-o", "NumberOfPasswordPrompts=1"]

        print(f"Agent: copying intake_process.json to {target} ...")
        with open(INTAKE_PROCESS_PATH, "rb") as handle:
            upload = subprocess.run([ssh, *ssh_options, target, upload_command], stdin=handle, env=env, check=False)
        if upload.returncode != 0:
            print(
                "Agent: could not copy the file to the server "
                "(check the login in tools/server_secrets.py and the connection).",
                file=sys.stderr,
            )
            return 1

        print("Agent: starting the workflow on the server ...")
        run = subprocess.run([ssh, "-t", *ssh_options, target, run_command], env=env, check=False)
    if run.returncode != 0:
        print(f"Agent: the server run ended with exit code {run.returncode}.", file=sys.stderr)
    return run.returncode


def _run_model_extractor(form: ModelPrompt | None, recompile: bool) -> int:
    if form is None:
        return 0
    if not form.prompt.strip():
        print("Agent: please enter a prompt before continuing.", file=sys.stderr)
        return 1

    pipeline = ModelExtractorPipeline(settings=load_model_settings(), recompile=recompile)
    try:
        result = pipeline.run(form.prompt)
    except ValueError as exc:
        print(f"Agent: {exc}", file=sys.stderr)
        return 1
    except RuntimeError as exc:
        print(f"Agent: {exc}", file=sys.stderr)
        return 1

    print(f"Agent: {result['agent_reply']}")
    print(f"Agent: wrote {model_briefs_path()}")
    build_final_intake(
        form.new_architecture,
        form.new_architecture_name,
        form.save_new_architecture,
    )
    print(f"Agent: wrote {INTAKE_PROCESS_PATH}")
    print(
        f"(output={result['output_path']}, backend={result['backend']}, "
        f"model={result['model']})"
    )
    if _ask_run_location() == "server":
        return _run_on_server()
    subprocess.run(
        [
            sys.executable,
            "-m",
            "tools.execution_mode_dspy.cli",
            "--from-gui",
            "--recompile=False",
        ],
        check=False,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
