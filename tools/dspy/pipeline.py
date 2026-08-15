"""
Glue between DSPy (intent) and YAML (config on disk).

``ConfigAgentPipeline.run`` is the story in one place:
  English request -> ``extract_intent`` (LM) -> merge into template -> write YAML.

No LM calls here — only orchestration. Details: DSPY_IMPLEMENTATION.md.
"""
from pathlib import Path

import yaml

from .modules import DSPyModules
from .optimization import (
	dspy_config_output,
	default_generated_output_dir,
	discover_repo_artifacts,
	load_default_template_config,
	write_outputs,
)
from .schemas import GenerationResult


class ConfigAgentPipeline:
	def __init__(self, repo_root, skip_bootstrap=None):
		self.repo_root = repo_root
		# Artifact lists are computed once per pipeline instance and fed to the LM as ``known_*``.
		self.artifacts = discover_repo_artifacts(repo_root)
		self.agent = DSPyModules(self.artifacts, skip_bootstrap=skip_bootstrap)

	def run(self, request, output_dir=None):
		# 1) LM step  2) template  3) deterministic merge  4) serialize + write
		intent = self.agent.extract_intent(request)
		base_cfg = load_default_template_config(self.repo_root)
		plan = dspy_config_output(intent, base_cfg, self.artifacts)

		result = GenerationResult(
			intent=intent,
			plan=plan,
			config_yaml=yaml.safe_dump(plan.config_data, sort_keys=False),
		)

		if output_dir is None:
			output_dir = str(default_generated_output_dir(self.repo_root))
		write_outputs(result, output_dir)

		return result
