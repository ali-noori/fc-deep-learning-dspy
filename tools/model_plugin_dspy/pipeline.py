from __future__ import annotations

"""Orchestration: DSPy JSON plan -> layered code or torchvision transfer -> validate."""

from dataclasses import asdict
import importlib.util
from pathlib import Path
from typing import Any, Dict, Tuple

from .generator import render_model_file, write_model_file
from .modules import ModelDesignAgent, PluginModelDesign
from .transfer import TransferPluginConfig, render_transfer_model_file


def _is_conv_net(design: PluginModelDesign) -> bool:
    return any(m.get("type") == "Conv2d" for m in design.modules)


class ModelPluginPipeline:
    """DSPy ChainOfThought -> plugins/models/customized_model.py (fixed name)."""

    def __init__(self, repo_root: str):
        self.repo_root = Path(repo_root)
        self._agent: ModelDesignAgent | None = None

    @property
    def agent(self) -> ModelDesignAgent:
        if self._agent is None:
            self._agent = ModelDesignAgent()
        return self._agent

    def _resolve_with_retries(
        self, request: str, initial_repair_hint: str = ""
    ):
        req = request.strip()
        repair = (initial_repair_hint or "").strip()
        last_err: BaseException | None = None
        for _ in range(5):
            try:
                return self.agent.resolve(request=req, repair_hint=repair)
            except (ValueError, RuntimeError, TypeError, KeyError) as exc:
                last_err = exc
                base = (initial_repair_hint or "").strip()
                repair = (
                    f"{base} Output one valid architecture_json with non-empty modules[] and forward[] "
                    f"(mode layered). Last error: {exc}"
                    if base
                    else (
                        "Output one valid architecture_json with non-empty modules[] and forward[] "
                        f'(mode "layered"). Last error: {exc}'
                    )
                )
        assert last_err is not None
        raise last_err

    def run(self, request: str) -> Dict[str, Any]:
        target = self.repo_root / "plugins" / "models" / "customized_model.py"
        req = request.strip()
        resolved = self._resolve_with_retries(req, "")
        if resolved.kind == "transfer" and resolved.transfer is not None:
            written, cfg = self._write_and_validate_transfer(
                target, resolved.transfer, request
            )
            return {
                "approach": "dspy_torchvision_transfer",
                "transfer_model": cfg.canonical_name,
                "output_model": str(written),
                "input_dim": cfg.input_dim,
                "output_dim": cfg.output_dim,
                "dataset_size": cfg.dataset_size,
                "input_channels": cfg.input_channels,
                "image_height": cfg.image_height,
                "image_width": cfg.image_width,
                "design": asdict(cfg),
            }

        if resolved.layered is None:
            raise RuntimeError("DSPy returned neither transfer nor layered plan.")
        design = resolved.layered
        written, design = self._write_and_validate_layered(target, design, request)

        return {
            "approach": "dspy_layered_model_plugin",
            "output_model": str(written),
            "input_dim": design.input_dim,
            "output_dim": design.output_dim,
            "dataset_size": design.dataset_size,
            "design": asdict(design),
        }

    def _write_and_validate_transfer(
        self, target: Path, cfg: TransferPluginConfig, request: str
    ) -> Tuple[Path, TransferPluginConfig]:
        content = render_transfer_model_file(cfg)
        written = write_model_file(target, content)
        try:
            self._validate_transfer_model(written, cfg)
            return written, cfg
        except Exception as first_error:
            repaired = self._resolve_with_retries(
                request,
                f"Transfer model validation failed: {first_error}",
            )
            if repaired.kind != "transfer" or repaired.transfer is None:
                raise RuntimeError(
                    "Repair pass did not return a valid transfer plan."
                ) from first_error
            cfg2 = repaired.transfer
            content2 = render_transfer_model_file(cfg2)
            written2 = write_model_file(target, content2)
            self._validate_transfer_model(written2, cfg2)
            return written2, cfg2

    def _write_and_validate_layered(
        self, target: Path, design: PluginModelDesign, request: str
    ) -> Tuple[Path, PluginModelDesign]:
        content = render_model_file(design)
        written = write_model_file(target, content)
        try:
            self._validate_generated_model(written, design)
            return written, design
        except Exception as first_error:
            repaired = self._resolve_with_retries(
                request,
                f"Previous generated model failed validation: {first_error}",
            )
            if repaired.kind != "layered" or repaired.layered is None:
                raise RuntimeError("Repair pass did not return a layered plan.") from first_error
            content2 = render_model_file(repaired.layered)
            written2 = write_model_file(target, content2)
            self._validate_generated_model(written2, repaired.layered)
            return written2, repaired.layered

    def _validate_transfer_model(self, model_path: Path, cfg: TransferPluginConfig) -> None:
        try:
            import torch
        except Exception as exc:
            raise RuntimeError("PyTorch is required to validate generated model.") from exc

        module_name = f"generated_model_plugin_{model_path.stem}"
        spec = importlib.util.spec_from_file_location(module_name, str(model_path))
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Could not import generated model module at {model_path}")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        model = mod.Model(n_classes=cfg.output_dim, in_features=cfg.input_channels)
        model.eval()
        x = torch.randn(2, cfg.input_channels, cfg.image_height, cfg.image_width)
        out = model(x)
        if tuple(out.shape) != (2, cfg.output_dim):
            raise RuntimeError(
                f"Transfer model forward shape is {tuple(out.shape)}, expected (2, {cfg.output_dim})."
            )

    def _validate_generated_model(self, model_path: Path, design: PluginModelDesign) -> None:
        try:
            import torch
        except Exception as exc:
            raise RuntimeError("PyTorch is required to validate generated model.") from exc

        module_name = f"generated_model_plugin_{model_path.stem}"
        spec = importlib.util.spec_from_file_location(module_name, str(model_path))
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Could not import generated model module at {model_path}")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        in_features_arg = design.input_channels
        model = mod.Model(n_classes=design.output_dim, in_features=in_features_arg)
        model.eval()

        if _is_conv_net(design):
            x = torch.randn(2, design.input_channels, design.image_height, design.image_width)
        else:
            x = torch.randn(2, design.input_dim)

        out = model(x)
        if tuple(out.shape) != (2, design.output_dim):
            raise RuntimeError(
                f"Generated model forward shape is {tuple(out.shape)}, expected (2, {design.output_dim})."
            )
