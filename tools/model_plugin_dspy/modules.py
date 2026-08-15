from __future__ import annotations

"""DSPy layer: structured prompt -> JSON plan -> PluginModelDesign."""

import json
import math
import os
import re
from dataclasses import dataclass, field
from typing import Any, Literal, Optional
from urllib.request import urlopen

from tools.dspy.lm_secrets import force_ollama_only, get_groq_api_key, get_groq_model_spec
from tools.dspy.ollama_settings import get_ollama_base_url, get_ollama_model

from .transfer import TransferPluginConfig, build_transfer_config, normalize_transfer_model_name

try:
    import dspy

    HAS_DSPY = True

    class LayeredArchitectureSignature(dspy.Signature):
        """Use ChainOfThought (not plain Predict) for structured layer specs."""

        request = dspy.InputField(
            desc=(
                "Model specification: either a structured layer list (Conv2d, MaxPool2d, Linear, …) "
                "or plain English plus numbers."
            )
        )
        repair_hint = dspy.InputField(
            desc="Optional validation error from a previous attempt; empty string if first try."
        )
        architecture_json = dspy.OutputField(
            desc=(
                "One single line of JSON only (no markdown). Parse EVERYTHING from the user request text. "
                'Set mode: "transfer" OR "layered". '
                "Numbers: infer output_dim from n_classes / class count; input_dim as flattened size (e.g. 784 "
                'for 28×28); input_channels for CNNs; image_height and image_width. '
                "dataset_size is optional metadata; use 1 if not mentioned. "
                "If mode is transfer: transfer_model must be one of "
                "ResNet18, ResNet50, EfficientNet-B0, VGG16, MobileNetV3, MobileNetV3-Small, ViT, SqueezeNet. "
                "Lightweight (MobileNetV3-Small, EfficientNet-B0, SqueezeNet): no modules/forward. "
                "If mode is layered: modules[] types include "
                "Conv2d (set use_in_features_for_in_channels true when prompt says in_features→out on first conv), "
                "MaxPool2d, ReLU, SiLU, Dropout, Dropout2d, AdaptiveAvgPool2d, Linear, Flatten, BatchNorm1d. "
                "BatchNorm1d needs num_features matching preceding Linear out_features. "
                "Every Linear needs positive in_features. Final layer uses out_features_from_n_classes true. "
                "forward[] steps: apply, view, flatten, log_softmax dim, return_logits. "
                'use_functional_softmax true if log_softmax in forward. '
                "Match declared layer order exactly."
            )
        )

except Exception:
    HAS_DSPY = False


@dataclass
class PluginModelDesign:
    """Normalized design for generator + validation."""

    dataset_size: int
    input_dim: int
    output_dim: int
    input_channels: int
    image_height: int
    image_width: int
    notes: str
    modules: list[dict[str, Any]] = field(default_factory=list)
    forward: list[dict[str, Any]] = field(default_factory=list)
    use_functional_softmax: bool = False


@dataclass
class ResolvedPluginPlan:
    """DSPy output: either custom layers or a torchvision transfer backbone."""

    kind: Literal["layered", "transfer"]
    layered: Optional[PluginModelDesign] = None
    transfer: Optional[TransferPluginConfig] = None


def _to_int(payload: dict, key: str, default: int) -> int:
    v = payload.get(key, default)
    if v is None:
        return default
    try:
        return int(v)
    except Exception:
        return default


def _to_float(payload: dict, key: str, default: float) -> float:
    v = payload.get(key, default)
    if v is None:
        return default
    try:
        return float(v)
    except Exception:
        return default


def _strip_code_fence(text: str) -> str:
    s = text.strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE)
        s = re.sub(r"\s*```\s*$", "", s)
    return s


def _try_eval_int_expr(expr: str) -> int | None:
    e = expr.strip()
    if not e:
        return None
    if re.fullmatch(r"-?\d+", e):
        return int(e)
    if not re.fullmatch(r"[0-9.+\-*/()\s]+", e):
        return None
    try:
        v = eval(e, {"__builtins__": {}}, {})
    except Exception:
        return None
    if isinstance(v, bool):
        return None
    if isinstance(v, float):
        if math.isfinite(v) and abs(v - round(v)) < 1e-9:
            return int(round(v))
        return int(v)
    if isinstance(v, int):
        return v
    return None


def _sanitize_arithmetic_in_json_values(s: str) -> str:
    """LLMs sometimes emit Python arithmetic where JSON requires a single number."""

    def fix_field(m: re.Match) -> str:
        key = m.group(1)
        val_raw = m.group(2).strip()
        replacement: str | None = None
        if val_raw.startswith('"') and val_raw.endswith('"') and len(val_raw) >= 2:
            inner = val_raw[1:-1]
            if any(c in inner for c in "*+/-") and re.fullmatch(r"[0-9.+\-*/()\s]+", inner):
                n = _try_eval_int_expr(inner.replace("×", "*"))
                if n is not None:
                    replacement = str(n)
        else:
            n = _try_eval_int_expr(val_raw.replace("×", "*"))
            if n is not None:
                replacement = str(n)
        if replacement is None:
            return m.group(0)
        return f'{key}{replacement}'

    return re.sub(
        r'("(?:in_features|out_features|num_features)"\s*:\s*)([^,}\]]+)',
        fix_field,
        s,
    )


def _extract_json_object(text: str) -> dict[str, Any]:
    s = _strip_code_fence(text).strip()
    candidates = [s, _sanitize_arithmetic_in_json_values(s)]
    for cand in candidates:
        try:
            obj = json.loads(cand)
            if isinstance(obj, dict) and ("modules" in obj or "mode" in obj):
                return obj
        except json.JSONDecodeError:
            continue

    dec = json.JSONDecoder()
    best: dict[str, Any] | None = None
    best_rank = -1
    for i, ch in enumerate(s):
        if ch != "{":
            continue
        try:
            obj, end = dec.raw_decode(s[i:])
            if not isinstance(obj, dict):
                continue
            span = end
            rank = span
            if "modules" in obj:
                rank += 10_000
            if "mode" in obj:
                rank += 1_000
            if "forward" in obj:
                rank += 100
            if rank > best_rank:
                best_rank = rank
                best = obj
        except json.JSONDecodeError:
            continue
    if best is not None:
        return best
    raise ValueError("No JSON object found in model output.")


def _skip_bootstrap_from_env() -> bool:
    v = os.environ.get("FC_MODEL_PLUGIN_DSPY_SKIP_BOOTSTRAP", "").strip().lower()
    return v in ("1", "true", "yes")


def _ollama_up(base_url: str) -> bool:
    try:
        with urlopen(f"{base_url}/api/tags", timeout=2):
            return True
    except Exception:
        return False


def _norm_module(m: dict[str, Any]) -> dict[str, Any]:
    out = dict(m)
    out["name"] = str(out.get("name", "")).strip()
    out["type"] = _canonical_module_type(str(out.get("type", "")).strip())
    return out


def _canonical_module_type(raw: str) -> str:
    """Map LLM aliases to generator-supported PyTorch module names."""
    if not raw:
        return raw
    key = raw.replace(" ", "").lower()
    aliases = {
        "flatten": "Flatten",
        "flat": "Flatten",
        "nn.flatten": "Flatten",
        "silu": "SiLU",
        "batchnorm1d": "BatchNorm1d",
        "batch_norm_1d": "BatchNorm1d",
    }
    if key in aliases:
        return aliases[key]
    return raw


_SPATIAL_PAIR_RE = re.compile(r"(\d{1,4})\s*[×xX]\s*(\d{1,4})")


def _merge_request_hints_into_payload(payload: dict[str, Any], text: str) -> None:
    """Fill missing schema numbers from natural language (request or repair_hint)."""
    if not text or not str(text).strip():
        return
    t = str(text)

    def dim_missing_or_bad(key: str) -> bool:
        return _to_int(payload, key, 0) <= 0

    def absent_or_bad_channel() -> bool:
        if "input_channels" not in payload:
            return True
        return _to_int(payload, "input_channels", 0) <= 0

    if dim_missing_or_bad("input_dim"):
        m = re.search(r"input_dim\s*[:=]?\s*(\d+)", t, re.I)
        if m:
            payload["input_dim"] = int(m.group(1))

    if dim_missing_or_bad("output_dim"):
        for pat in (
            r"output_dim\s*[:=]?\s*(\d+)",
            r"n_classes\s*[:=]?\s*(\d+)",
            r"(\d+)\s+classes",
            r"(\d+)\s+class\b",
        ):
            m = re.search(pat, t, re.I)
            if m:
                payload["output_dim"] = int(m.group(1))
                break

    if _to_int(payload, "dataset_size", 0) <= 0:
        m = re.search(r"dataset_size\s*[:=]?\s*(\d+)", t, re.I)
        if m:
            payload["dataset_size"] = int(m.group(1))

    if absent_or_bad_channel():
        m = re.search(r"input_channels?\s*[:=]?\s*(\d+)", t, re.I)
        if m:
            payload["input_channels"] = int(m.group(1))
        elif re.search(r"\brgb\b|3[-\s]?channel|three\s+channel", t, re.I):
            payload["input_channels"] = 3
        elif re.search(r"grayscale|greyscale|1[-\s]?channel|mono(?:chrome)?", t, re.I):
            payload["input_channels"] = 1

    m_sp = _SPATIAL_PAIR_RE.search(t)
    if m_sp:
        h, w = int(m_sp.group(1)), int(m_sp.group(2))
        if payload.get("image_height") is None:
            payload["image_height"] = h
        if payload.get("image_width") is None:
            payload["image_width"] = w
        ch = _to_int(payload, "input_channels", 0)
        if ch <= 0:
            ch = 1
        if dim_missing_or_bad("input_dim") and h > 0 and w > 0:
            payload["input_dim"] = h * w * ch


_ALLOWED_MODULE_TYPES = frozenset(
    {
        "Conv2d",
        "MaxPool2d",
        "ReLU",
        "SiLU",
        "Dropout2d",
        "Dropout",
        "AdaptiveAvgPool2d",
        "Linear",
        "Flatten",
        "BatchNorm1d",
    }
)


def _validate_design(d: PluginModelDesign) -> None:
    names = {m["name"] for m in d.modules}
    if len(names) != len(d.modules):
        raise ValueError("Duplicate module names in design.")
    for m in d.modules:
        if not m["name"]:
            raise ValueError("Module missing name.")
        if m["type"] not in _ALLOWED_MODULE_TYPES:
            raise ValueError(f"Unsupported module type: {m['type']}")
    for step in d.forward:
        if "apply" in step:
            n = str(step["apply"])
            if n not in names:
                raise ValueError(f"forward step references unknown module: {n}")
    _validate_batchnorm1d_after_flatten(d)


def _validate_batchnorm1d_after_flatten(d: PluginModelDesign) -> None:
    if not any(m["type"] == "Conv2d" for m in d.modules):
        return
    by_name = {m["name"]: m for m in d.modules}
    rank4 = True
    for step in d.forward:
        if "apply" in step:
            mod = by_name.get(str(step["apply"]))
            if mod is None:
                continue
            t = mod["type"]
            if t == "BatchNorm1d" and rank4:
                raise ValueError(
                    "BatchNorm1d must come after Flatten (2D activations), not while feature maps are still 4D."
                )
            if t in ("Flatten", "Linear"):
                rank4 = False
        elif "flatten" in step or "view" in step:
            rank4 = False


def _sync_conv2d_in_channels(modules: list[dict[str, Any]]) -> None:
    """Ensure each Conv2d in_features matches the previous Conv2d's out_channels."""
    prev_out: int | None = None
    for m in modules:
        if m["type"] != "Conv2d":
            continue
        if m.get("use_in_features_for_in_channels"):
            prev_out = int(m["out_channels"])
            continue
        if prev_out is not None:
            m["in_channels"] = prev_out
        prev_out = int(m["out_channels"])


def _fix_linear_in_features_from_conv_geometry(d: PluginModelDesign) -> None:
    """Correct first Linear in_features from conv/pool/adaptive/flatten geometry when wrong."""
    if not any(m["type"] == "Conv2d" for m in d.modules):
        return
    by_name = {m["name"]: m for m in d.modules}
    c, h, w = d.input_channels, d.image_height, d.image_width
    flat_dim: int | None = None
    rank4 = True
    first_linear_patched = False

    for step in d.forward:
        if "view" in step:
            v1, v2 = step["view"]
            flat_dim = int(v2)
            rank4 = False
            continue
        if "flatten" in step:
            flat_dim = c * h * w
            rank4 = False
            continue
        if "apply" not in step:
            continue
        mod = by_name.get(str(step["apply"]))
        if mod is None:
            continue
        t = mod["type"]
        if t == "Conv2d":
            kh = int(mod["kernel_size"])
            pad = int(mod.get("padding", 0))
            st = int(mod.get("stride", 1))
            c = int(mod["out_channels"])
            h = (h + 2 * pad - kh) // st + 1
            w = (w + 2 * pad - kh) // st + 1
        elif t == "MaxPool2d":
            k = int(mod["kernel_size"])
            h //= k
            w //= k
        elif t == "AdaptiveAvgPool2d":
            oh, ow = mod["output_size"]
            h, w = int(oh), int(ow)
        elif t == "Flatten":
            flat_dim = c * h * w
            rank4 = False
        elif t == "Linear":
            if first_linear_patched:
                continue
            target = flat_dim if flat_dim is not None else (c * h * w if rank4 else None)
            if target is not None and target > 0:
                mod["in_features"] = target
            first_linear_patched = True


def _parse_payload(payload: dict[str, Any]) -> PluginModelDesign:
    dataset_size = _to_int(payload, "dataset_size", 0)
    if dataset_size <= 0:
        dataset_size = 1
    input_dim = _to_int(payload, "input_dim", 0)
    output_dim = _to_int(payload, "output_dim", 0)
    if input_dim <= 0:
        raise RuntimeError(
            "architecture_json: input_dim must be a positive integer (infer from request, e.g. 784 for 28×28)."
        )
    if output_dim <= 0:
        raise RuntimeError(
            "architecture_json: output_dim must be a positive integer (infer n_classes from request)."
        )

    input_channels = max(1, min(32, _to_int(payload, "input_channels", 1)))
    image_height = max(4, min(1024, _to_int(payload, "image_height", 28)))
    image_width = max(4, min(1024, _to_int(payload, "image_width", 28)))
    notes = str(payload.get("notes", "")).strip()
    use_f = bool(payload.get("use_functional_softmax", False))

    raw_modules = payload.get("modules") or []
    if not isinstance(raw_modules, list) or not raw_modules:
        raise RuntimeError("architecture_json must include a non-empty 'modules' array.")

    modules: list[dict[str, Any]] = []
    for raw in raw_modules:
        if not isinstance(raw, dict):
            continue
        m = _norm_module(raw)
        t = m["type"]
        if t == "Conv2d":
            m["use_in_features_for_in_channels"] = bool(m.get("use_in_features_for_in_channels", False))
            if not m["use_in_features_for_in_channels"]:
                ic = _to_int(m, "in_channels", 0)
                if ic <= 0:
                    ic = _to_int(m, "in_features", 0)
                m["in_channels"] = max(1, ic if ic > 0 else input_channels)
            oc = _to_int(m, "out_channels", 0)
            if oc <= 0:
                oc = _to_int(m, "out_features", 0)
            m["out_channels"] = max(1, oc if oc > 0 else 8)
            ks_raw = m.get("kernel_size", 3)
            if isinstance(ks_raw, (list, tuple)) and len(ks_raw) > 0:
                m["kernel_size"] = max(1, min(15, int(ks_raw[0])))
            else:
                m["kernel_size"] = max(1, min(15, _to_int(m, "kernel_size", 3)))
            m["padding"] = max(0, min(8, _to_int(m, "padding", 0)))
            m["stride"] = max(1, min(4, _to_int(m, "stride", 1)))
        elif t == "MaxPool2d":
            ks_raw = m.get("kernel_size", 2)
            if isinstance(ks_raw, (list, tuple)) and len(ks_raw) > 0:
                m["kernel_size"] = max(1, min(8, int(ks_raw[0])))
            else:
                m["kernel_size"] = max(1, min(8, _to_int(m, "kernel_size", 2)))
        elif t == "AdaptiveAvgPool2d":
            sz = m.get("output_size") or [4, 4]
            if isinstance(sz, list) and len(sz) == 2:
                h, w = int(sz[0]), int(sz[1])
            else:
                h, w = 4, 4
            m["output_size"] = [max(1, min(64, h)), max(1, min(64, w))]
        elif t == "Linear":
            inf = _to_int(m, "in_features", 0)
            if inf <= 0:
                raise RuntimeError(
                    "architecture_json: each Linear must include a positive integer in_features."
                )
            m["in_features"] = inf
            m["out_features_from_n_classes"] = bool(m.get("out_features_from_n_classes", False))
            of = _to_int(m, "out_features", 0)
            if not m["out_features_from_n_classes"] and of == output_dim and output_dim > 0:
                m["out_features_from_n_classes"] = True
            if not m["out_features_from_n_classes"]:
                m["out_features"] = max(1, of if of > 0 else 10)
        elif t == "Flatten":
            m["start_dim"] = max(0, min(3, _to_int(m, "start_dim", 1)))
            end = m.get("end_dim", -1)
            try:
                m["end_dim"] = int(end)
            except Exception:
                m["end_dim"] = -1
        elif t == "Dropout":
            m["p"] = max(0.0, min(0.9, _to_float(m, "p", 0.5)))
        elif t == "Dropout2d":
            m["p"] = max(0.0, min(0.9, _to_float(m, "p", 0.0)))
        elif t == "SiLU":
            pass
        elif t == "BatchNorm1d":
            nf = _to_int(m, "num_features", 0)
            if nf <= 0:
                raise RuntimeError("architecture_json: BatchNorm1d requires positive num_features.")
            m["num_features"] = nf
        modules.append(m)

    for i, m in enumerate(modules):
        if not str(m.get("name", "")).strip():
            m["name"] = f"m{i}"

    _sync_conv2d_in_channels(modules)

    raw_forward = payload.get("forward") or []
    if not isinstance(raw_forward, list):
        raw_forward = []

    forward: list[dict[str, Any]] = []
    for step in raw_forward:
        if not isinstance(step, dict):
            continue
        if "apply" in step:
            forward.append({"apply": str(step["apply"]).strip()})
        elif "view" in step:
            v = step["view"]
            if isinstance(v, list) and len(v) == 2:
                forward.append({"view": [int(v[0]), int(v[1])]})
        elif "flatten" in step:
            fd = step["flatten"]
            sd = 1
            if isinstance(fd, dict):
                sd = _to_int(fd, "start_dim", 1)
            forward.append({"flatten": {"start_dim": sd}})
        elif "log_softmax" in step:
            forward.append({"log_softmax": int(step["log_softmax"])})
        elif step.get("return_logits") is True:
            forward.append({"return_logits": True})

    if not forward and modules:
        for m in modules:
            n = str(m.get("name", "")).strip()
            if n:
                forward.append({"apply": n})
        if forward:
            if bool(payload.get("use_functional_softmax", False)):
                forward.append({"log_softmax": 1})
            else:
                forward.append({"return_logits": True})

    if not forward:
        raise RuntimeError(
            "architecture_json must include a non-empty 'forward' array or at least one module name for "
            "a default sequential forward."
        )

    last = forward[-1]
    if "log_softmax" in last:
        use_f = True

    d = PluginModelDesign(
        dataset_size=dataset_size,
        input_dim=input_dim,
        output_dim=output_dim,
        input_channels=input_channels,
        image_height=image_height,
        image_width=image_width,
        notes=notes,
        modules=modules,
        forward=forward,
        use_functional_softmax=use_f,
    )
    _fix_linear_in_features_from_conv_geometry(d)
    _validate_design(d)
    return d


def _effective_transfer_model_name(payload: dict[str, Any]) -> str | None:
    raw = payload.get("transfer_model") or payload.get("backbone")
    if raw is None:
        return None
    s = str(raw).strip()
    if not s or s.lower() in ("none", "null", "nil", "n/a"):
        return None
    return s


def _transfer_mode_from_payload(payload: dict[str, Any]) -> bool:
    mode = str(payload.get("mode") or "").strip().lower()
    tname = _effective_transfer_model_name(payload)
    if mode in ("layered", "custom", "scratch", "manual"):
        return False
    if mode in ("transfer", "transfer_learning", "pretrained", "torchvision_transfer"):
        return tname is not None
    if tname and not payload.get("modules"):
        return True
    return False


def _parse_transfer_from_payload(payload: dict[str, Any]) -> TransferPluginConfig:
    raw_tm = _effective_transfer_model_name(payload)
    if not raw_tm:
        raise RuntimeError(
            "Transfer mode requires transfer_model in architecture_json (e.g. EfficientNet-B0)."
        )
    canonical = normalize_transfer_model_name(raw_tm)
    ds = _to_int(payload, "dataset_size", 0)
    if ds <= 0:
        ds = 1
    idim = _to_int(payload, "input_dim", 0)
    odim = _to_int(payload, "output_dim", 0)
    if idim <= 0:
        raise RuntimeError(
            "Transfer mode requires a positive input_dim in architecture_json (infer from request)."
        )
    if odim <= 0:
        raise RuntimeError(
            "Transfer mode requires a positive output_dim in architecture_json (infer n_classes from request)."
        )
    ich_raw = _to_int(payload, "input_channels", 0)
    ich: int | None = None if ich_raw <= 0 else max(1, min(32, ich_raw))
    return build_transfer_config(canonical, ds, idim, odim, ich)


def resolve_plugin_plan(payload: dict[str, Any]) -> ResolvedPluginPlan:
    if _transfer_mode_from_payload(payload):
        cfg = _parse_transfer_from_payload(payload)
        return ResolvedPluginPlan(kind="transfer", layered=None, transfer=cfg)
    layered = _parse_payload(payload)
    return ResolvedPluginPlan(kind="layered", layered=layered, transfer=None)


class ModelDesignAgent:
    """DSPy ChainOfThought wrapper: prompt text -> ResolvedPluginPlan."""

    def __init__(self, skip_bootstrap: Optional[bool] = None) -> None:
        self.use_dspy = HAS_DSPY
        self.predictor = None
        self._backend: Optional[str] = None
        self._init_error: Optional[BaseException] = None
        if skip_bootstrap is None:
            self._skip_bootstrap = _skip_bootstrap_from_env()
        else:
            self._skip_bootstrap = bool(skip_bootstrap)

        self._ollama_base_url = get_ollama_base_url()
        self._ollama_model = get_ollama_model()
        self.ollama_available = _ollama_up(self._ollama_base_url)

        groq_key = get_groq_api_key()
        if self.use_dspy and groq_key and not force_ollama_only():
            try:
                dspy.configure(lm=dspy.LM(get_groq_model_spec(), api_key=groq_key))
                student = dspy.ChainOfThought(LayeredArchitectureSignature)
                self.predictor = self._wrap_student(student)
                self._backend = "groq"
            except BaseException as exc:
                self._init_error = exc
                self.predictor = None

        if self.use_dspy and self.predictor is None and self.ollama_available:
            try:
                dspy.configure(
                    lm=dspy.LM(
                        f"ollama/{self._ollama_model}",
                        api_base=self._ollama_base_url,
                    )
                )
                student = dspy.ChainOfThought(LayeredArchitectureSignature)
                self.predictor = self._wrap_student(student)
                self._backend = "ollama"
            except BaseException as exc:
                self._init_error = exc
                self.predictor = None

    def _wrap_student(self, student):
        if self._skip_bootstrap:
            return student
        return dspy.BootstrapFewShot(metric=self._architecture_metric).compile(
            student=student,
            trainset=self._bootstrap_examples(),
        )

    def _bootstrap_examples(self):
        inp = ("request", "repair_hint")

        def ex(request: str, arch: dict[str, Any], repair_hint: str = ""):
            return dspy.Example(
                request=request,
                repair_hint=repair_hint,
                architecture_json=json.dumps(arch, separators=(",", ":")),
            ).with_inputs(*inp)

        classic = {
            "mode": "layered",
            "dataset_size": 60000,
            "input_dim": 784,
            "output_dim": 10,
            "input_channels": 1,
            "image_height": 28,
            "image_width": 28,
            "notes": "mnist-style",
            "use_functional_softmax": True,
            "modules": [
                {
                    "name": "conv1",
                    "type": "Conv2d",
                    "use_in_features_for_in_channels": False,
                    "in_channels": 1,
                    "out_channels": 10,
                    "kernel_size": 5,
                    "padding": 0,
                    "stride": 1,
                },
                {"name": "conv2", "type": "Conv2d", "in_channels": 10, "out_channels": 20, "kernel_size": 5, "padding": 0, "stride": 1},
                {"name": "conv2_drop", "type": "Dropout2d", "p": 0.0},
                {"name": "pool", "type": "MaxPool2d", "kernel_size": 2},
                {"name": "act", "type": "ReLU"},
                {"name": "fc1", "type": "Linear", "in_features": 320, "out_features": 50},
                {"name": "drop", "type": "Dropout", "p": 0.5},
                {"name": "fc2", "type": "Linear", "in_features": 50, "out_features_from_n_classes": True},
            ],
            "forward": [
                {"apply": "conv1"},
                {"apply": "pool"},
                {"apply": "act"},
                {"apply": "conv2"},
                {"apply": "conv2_drop"},
                {"apply": "pool"},
                {"apply": "act"},
                {"view": [-1, 320]},
                {"apply": "fc1"},
                {"apply": "act"},
                {"apply": "drop"},
                {"apply": "fc2"},
                {"log_softmax": 1},
            ],
        }

        two_conv_adapt = {
            "mode": "layered",
            "dataset_size": 12000,
            "input_dim": 784,
            "output_dim": 10,
            "input_channels": 1,
            "image_height": 28,
            "image_width": 28,
            "notes": "two conv + adapt",
            "use_functional_softmax": False,
            "modules": [
                {
                    "name": "conv1",
                    "type": "Conv2d",
                    "use_in_features_for_in_channels": True,
                    "out_channels": 20,
                    "kernel_size": 3,
                    "padding": 1,
                    "stride": 1,
                },
                {"name": "pool", "type": "MaxPool2d", "kernel_size": 2},
                {"name": "act", "type": "ReLU"},
                {"name": "conv2", "type": "Conv2d", "in_channels": 20, "out_channels": 50, "kernel_size": 3, "padding": 1, "stride": 1},
                {"name": "adapt", "type": "AdaptiveAvgPool2d", "output_size": [4, 4]},
                {"name": "fc1", "type": "Linear", "in_features": 800, "out_features": 128},
                {"name": "drop", "type": "Dropout", "p": 0.2},
                {"name": "fc_out", "type": "Linear", "in_features": 128, "out_features_from_n_classes": True},
            ],
            "forward": [
                {"apply": "conv1"},
                {"apply": "pool"},
                {"apply": "act"},
                {"apply": "conv2"},
                {"apply": "pool"},
                {"apply": "act"},
                {"apply": "adapt"},
                {"flatten": {"start_dim": 1}},
                {"apply": "fc1"},
                {"apply": "act"},
                {"apply": "drop"},
                {"apply": "fc_out"},
                {"return_logits": True},
            ],
        }

        three_conv_ls = {
            "mode": "layered",
            "dataset_size": 50000,
            "input_dim": 784,
            "output_dim": 10,
            "input_channels": 1,
            "image_height": 28,
            "image_width": 28,
            "notes": "three conv",
            "use_functional_softmax": True,
            "modules": [
                {
                    "name": "conv1",
                    "type": "Conv2d",
                    "in_channels": 1,
                    "out_channels": 20,
                    "kernel_size": 3,
                    "padding": 1,
                    "stride": 1,
                },
                {"name": "conv2", "type": "Conv2d", "in_channels": 20, "out_channels": 50, "kernel_size": 3, "padding": 1, "stride": 1},
                {"name": "conv3", "type": "Conv2d", "in_channels": 50, "out_channels": 100, "kernel_size": 3, "padding": 1, "stride": 1},
                {"name": "pool", "type": "MaxPool2d", "kernel_size": 2},
                {"name": "act", "type": "ReLU"},
                {"name": "adapt", "type": "AdaptiveAvgPool2d", "output_size": [4, 4]},
                {"name": "fc1", "type": "Linear", "in_features": 1600, "out_features": 128},
                {"name": "drop", "type": "Dropout", "p": 0.2},
                {"name": "fc_out", "type": "Linear", "in_features": 128, "out_features_from_n_classes": True},
            ],
            "forward": [
                {"apply": "conv1"},
                {"apply": "pool"},
                {"apply": "act"},
                {"apply": "conv2"},
                {"apply": "pool"},
                {"apply": "act"},
                {"apply": "conv3"},
                {"apply": "pool"},
                {"apply": "act"},
                {"apply": "adapt"},
                {"flatten": {"start_dim": 1}},
                {"apply": "fc1"},
                {"apply": "act"},
                {"apply": "drop"},
                {"apply": "fc_out"},
                {"log_softmax": 1},
            ],
        }

        mlp_tiny = {
            "mode": "layered",
            "dataset_size": 60000,
            "input_dim": 784,
            "output_dim": 10,
            "input_channels": 1,
            "image_height": 28,
            "image_width": 28,
            "notes": "mlp on flattened vectors",
            "use_functional_softmax": False,
            "modules": [
                {"name": "fc1", "type": "Linear", "in_features": 784, "out_features": 128},
                {"name": "act", "type": "ReLU"},
                {"name": "drop", "type": "Dropout", "p": 0.2},
                {"name": "fc_out", "type": "Linear", "in_features": 128, "out_features_from_n_classes": True},
            ],
            "forward": [
                {"flatten": {"start_dim": 1}},
                {"apply": "fc1"},
                {"apply": "act"},
                {"apply": "drop"},
                {"apply": "fc_out"},
                {"return_logits": True},
            ],
        }

        three_conv_1536 = {
            "mode": "layered",
            "dataset_size": 50000,
            "input_dim": 784,
            "output_dim": 10,
            "input_channels": 1,
            "image_height": 28,
            "image_width": 28,
            "notes": "three conv adapt 1536 256",
            "use_functional_softmax": True,
            "modules": [
                {
                    "name": "conv1",
                    "type": "Conv2d",
                    "in_channels": 1,
                    "out_channels": 24,
                    "kernel_size": 3,
                    "padding": 1,
                    "stride": 1,
                },
                {"name": "pool", "type": "MaxPool2d", "kernel_size": 2},
                {"name": "act", "type": "ReLU"},
                {
                    "name": "conv2",
                    "type": "Conv2d",
                    "in_channels": 24,
                    "out_channels": 48,
                    "kernel_size": 3,
                    "padding": 1,
                    "stride": 1,
                },
                {
                    "name": "conv3",
                    "type": "Conv2d",
                    "in_channels": 48,
                    "out_channels": 96,
                    "kernel_size": 3,
                    "padding": 1,
                    "stride": 1,
                },
                {"name": "adapt", "type": "AdaptiveAvgPool2d", "output_size": [4, 4]},
                {"name": "fc1", "type": "Linear", "in_features": 1536, "out_features": 256},
                {"name": "drop", "type": "Dropout", "p": 0.35},
                {
                    "name": "fc_out",
                    "type": "Linear",
                    "in_features": 256,
                    "out_features_from_n_classes": True,
                },
            ],
            "forward": [
                {"apply": "conv1"},
                {"apply": "pool"},
                {"apply": "act"},
                {"apply": "conv2"},
                {"apply": "pool"},
                {"apply": "act"},
                {"apply": "conv3"},
                {"apply": "pool"},
                {"apply": "act"},
                {"apply": "adapt"},
                {"flatten": {"start_dim": 1}},
                {"apply": "fc1"},
                {"apply": "act"},
                {"apply": "drop"},
                {"apply": "fc_out"},
                {"log_softmax": 1},
            ],
        }

        return [
            ex(
                request=(
                    "Conv2d: 1→10, kernel=5, no padding\n"
                    "MaxPool2d(2) → ReLU\n"
                    "Conv2d: 10→20, kernel=5, no padding\n"
                    "Dropout2d\n"
                    "MaxPool2d(2) → ReLU\n"
                    "Flatten to 320\n"
                    "Linear: 320→50 → ReLU\n"
                    "Dropout\n"
                    "Linear: 50→n_classes\n"
                    "log_softmax output\n"
                    "dataset_size 60000 input_dim 784 output_dim 10"
                ),
                arch=classic,
            ),
            ex(
                request=(
                    "Create a CNN architecture for image classification. The dataset contains 28×28 images "
                    "with in_features channel, and n_classes = 10.\n"
                    "Conv2d: in_features→20, kernel=3, padding=1\n"
                    "MaxPool2d(2) → ReLU\n"
                    "Conv2d: 20→50, kernel=3, padding=1\n"
                    "MaxPool2d(2) → ReLU\n"
                    "AdaptiveAvgPool2d(4×4)\n"
                    "Flatten to 800\n"
                    "Linear: 800→128 → ReLU\n"
                    "Dropout1d(0.2)\n"
                    "Linear: 128→n_classes\n"
                    "no softmax output"
                ),
                arch=two_conv_adapt,
            ),
            ex(
                request=(
                    "Three Conv2d blocks then AdaptiveAvgPool2d 4x4, "
                    "Linear 1600->128, Dropout 0.2, Linear to n_classes, log_softmax. "
                    "Channels 1,20,50,100 kernel 3,3,3 padding 1. "
                    "dataset_size 50000 input_dim 784 output_dim 10"
                ),
                arch=three_conv_ls,
            ),
            ex(
                request=(
                    "Conv2d: 1→24, kernel=3, padding=1\n"
                    "MaxPool2d(2) → ReLU\n"
                    "Conv2d: 24→48, kernel=3, padding=1\n"
                    "MaxPool2d(2) → ReLU\n"
                    "Conv2d: 48→96, kernel=3, padding=1\n"
                    "MaxPool2d(2) → ReLU\n"
                    "AdaptiveAvgPool2d(4×4)\n"
                    "Flatten to 1536\n"
                    "Linear: 1536→256 → ReLU\n"
                    "Dropout(0.35)\n"
                    "Linear: 256→n_classes\n"
                    "log_softmax output\n"
                    "dataset_size 50000 input_dim 784 output_dim 10 grayscale 28x28"
                ),
                arch=three_conv_1536,
            ),
            ex(
                request="Build an MLP: Linear 784->128, ReLU, Dropout 0.2, Linear to n_classes logits. "
                "dataset_size 60000 input_dim 784 output_dim 10",
                arch=mlp_tiny,
            ),
            ex(
                request=(
                    "Use torchvision transfer learning with ResNet18 pretrained on ImageNet. "
                    "dataset_size 60000, input_dim 784, output_dim 10, 1-channel grayscale 28x28."
                ),
                arch={
                    "mode": "transfer",
                    "transfer_model": "ResNet18",
                    "dataset_size": 60000,
                    "input_dim": 784,
                    "output_dim": 10,
                    "input_channels": 1,
                    "notes": "resnet18 transfer",
                },
            ),
            ex(
                request=(
                    "Create an MLP architecture for multiclass classification with n_classes = 10 "
                    "and in_features = 784 input features.\n"
                    "Linear: in_features→64 → SiLU\n"
                    "Dropout(0.2)\n"
                    "Linear: 64→32 → SiLU\n"
                    "Dropout(0.2)\n"
                    "Linear: 32→n_classes\n"
                    "no softmax output"
                ),
                arch={
                    "mode": "layered",
                    "dataset_size": 1,
                    "input_dim": 784,
                    "output_dim": 10,
                    "input_channels": 1,
                    "image_height": 28,
                    "image_width": 28,
                    "notes": "mlp silu",
                    "use_functional_softmax": False,
                    "modules": [
                        {"name": "fc1", "type": "Linear", "in_features": 784, "out_features": 64},
                        {"name": "act1", "type": "SiLU"},
                        {"name": "drop1", "type": "Dropout", "p": 0.2},
                        {"name": "fc2", "type": "Linear", "in_features": 64, "out_features": 32},
                        {"name": "act2", "type": "SiLU"},
                        {"name": "drop2", "type": "Dropout", "p": 0.2},
                        {
                            "name": "fc3",
                            "type": "Linear",
                            "in_features": 32,
                            "out_features_from_n_classes": True,
                        },
                    ],
                    "forward": [
                        {"apply": "fc1"},
                        {"apply": "act1"},
                        {"apply": "drop1"},
                        {"apply": "fc2"},
                        {"apply": "act2"},
                        {"apply": "drop2"},
                        {"apply": "fc3"},
                        {"return_logits": True},
                    ],
                },
            ),
            ex(
                request=(
                    "Create an MLP architecture for multiclass classification with n_classes = 10 "
                    "and in_features = 784 input features.\n"
                    "Linear: in_features→128 → BatchNorm1d → ReLU\n"
                    "Dropout(0.25)\n"
                    "Linear: 128→n_classes\n"
                    "no softmax"
                ),
                arch={
                    "mode": "layered",
                    "dataset_size": 1,
                    "input_dim": 784,
                    "output_dim": 10,
                    "input_channels": 1,
                    "image_height": 28,
                    "image_width": 28,
                    "notes": "mlp bn",
                    "use_functional_softmax": False,
                    "modules": [
                        {"name": "fc1", "type": "Linear", "in_features": 784, "out_features": 128},
                        {"name": "bn1", "type": "BatchNorm1d", "num_features": 128},
                        {"name": "act", "type": "ReLU"},
                        {"name": "drop", "type": "Dropout", "p": 0.25},
                        {
                            "name": "fc_out",
                            "type": "Linear",
                            "in_features": 128,
                            "out_features_from_n_classes": True,
                        },
                    ],
                    "forward": [
                        {"apply": "fc1"},
                        {"apply": "bn1"},
                        {"apply": "act"},
                        {"apply": "drop"},
                        {"apply": "fc_out"},
                        {"return_logits": True},
                    ],
                },
            ),
            ex(
                request=(
                    "Create a CNN architecture for image classification using MobileNetV3-Small transfer "
                    "learning as pretrained backbone. 28×28 images, 1 input channel, n_classes 10. "
                    "Use log_softmax output."
                ),
                arch={
                    "mode": "transfer",
                    "transfer_model": "MobileNetV3-Small",
                    "dataset_size": 1,
                    "input_dim": 784,
                    "output_dim": 10,
                    "input_channels": 1,
                    "notes": "mobilenet small",
                },
            ),
            ex(
                request=(
                    "Create a CNN architecture for image classification using EfficientNet-B0 transfer "
                    "learning as pretrained backbone. input_dim 784 output_dim 10 one grayscale channel."
                ),
                arch={
                    "mode": "transfer",
                    "transfer_model": "EfficientNet-B0",
                    "dataset_size": 1,
                    "input_dim": 784,
                    "output_dim": 10,
                    "input_channels": 1,
                    "notes": "efficientnet b0",
                },
            ),
            ex(
                request=(
                    "Create a CNN architecture for image classification using SqueezeNet transfer "
                    "learning as pretrained backbone. 28×28 n_classes 10 input channel 1. log_softmax output."
                ),
                arch={
                    "mode": "transfer",
                    "transfer_model": "SqueezeNet",
                    "dataset_size": 1,
                    "input_dim": 784,
                    "output_dim": 10,
                    "input_channels": 1,
                    "notes": "squeezenet",
                },
            ),
        ]

    @staticmethod
    def _architecture_metric(gold, pred, trace=None):
        try:
            g = json.loads(getattr(gold, "architecture_json", "{}"))
            p = json.loads(getattr(pred, "architecture_json", "{}"))
            g_transfer = g.get("transfer_model") or g.get("backbone")
            if g_transfer is not None and str(g.get("mode", "")).lower() != "layered":
                score = 0.0
                p_transfer = p.get("transfer_model") or p.get("backbone")
                if p_transfer is not None:
                    try:
                        if normalize_transfer_model_name(str(p_transfer)) == normalize_transfer_model_name(
                            str(g_transfer)
                        ):
                            score += 1.0
                    except ValueError:
                        pass
                if _to_int(p, "output_dim", 0) == _to_int(g, "output_dim", -1):
                    score += 0.5
                return min(1.0, score / 1.5)
            score = 0.0
            if _to_int(p, "output_dim", 0) == _to_int(g, "output_dim", -1):
                score += 1.0
            gm = g.get("modules") or []
            pm = p.get("modules") or []
            if isinstance(gm, list) and isinstance(pm, list) and len(pm) >= max(1, len(gm) - 1):
                score += 0.5
            gf = g.get("forward") or []
            pf = p.get("forward") or []
            if isinstance(gf, list) and isinstance(pf, list) and len(pf) >= max(1, len(gf) - 2):
                score += 0.5
            return min(1.0, score / 2.0)
        except Exception:
            return 0.0

    def resolve(self, request: str, repair_hint: str = "") -> ResolvedPluginPlan:
        if not self.use_dspy:
            raise RuntimeError("DSPy is not installed. Install dspy-ai in your venv.")
        if not self.predictor:
            hint = f" init_error={self._init_error!r}" if self._init_error else ""
            raise RuntimeError(
                "Could not initialize DSPy model-design predictor "
                "(Groq default, Ollama fallback). "
                "Set GROQ_API_KEY/FC_DSPY_GROQ_API_KEY or run Ollama at "
                f"{self._ollama_base_url}." + hint
            )

        out = self.predictor(request=request, repair_hint=repair_hint)
        raw_json = getattr(out, "architecture_json", None) or getattr(out, "design_json", "")
        payload = _extract_json_object(str(raw_json))
        _merge_request_hints_into_payload(payload, request)
        _merge_request_hints_into_payload(payload, repair_hint or "")
        return resolve_plugin_plan(payload)


# Backwards-compatible alias for imports
ModelDesign = PluginModelDesign
