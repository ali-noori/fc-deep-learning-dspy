"""Score predicted model briefs against gold briefs."""

from __future__ import annotations

_STOP = {
    "the", "and", "for", "with", "that", "this", "from", "into", "are", "was",
    "use", "using", "not", "but", "only", "each", "per", "one", "new",
}


def _tokens(text: object) -> set[str]:
    raw = str(text or "").lower().replace("/", " ").replace("\\", " ")
    cleaned = []
    token = []
    for char in raw:
        if char.isalnum() or char in {".", "_", "-"}:
            token.append(char)
        elif token:
            cleaned.append("".join(token))
            token = []
    if token:
        cleaned.append("".join(token))
    return {item for item in cleaned if len(item) >= 3 and item not in _STOP}


def _covers(gold: object, predicted: object) -> bool:
    gold_tokens = _tokens(gold)
    pred_tokens = _tokens(predicted)
    if not gold_tokens:
        return not pred_tokens
    if not pred_tokens:
        return False
    hit = len(gold_tokens & pred_tokens) / len(gold_tokens)
    return hit >= 0.45


def _exact(gold: object, predicted: object) -> bool:
    """Word numbers must match the user's spelling. ten is not 10."""
    return str(gold or "").strip() == str(predicted or "").strip()


def _get_field(obj: object, name: str) -> object | None:
    if hasattr(obj, name):
        return getattr(obj, name)
    if isinstance(obj, dict):
        return obj.get(name)
    return None


def model_extractor_metric(example, predicted, trace=None) -> float:
    try:
        if not _covers(
            _get_field(example, "architecture_description"),
            _get_field(predicted, "architecture_description"),
        ):
            return 0.0
        if not _covers(
            _get_field(example, "model_selection"),
            _get_field(predicted, "model_selection"),
        ):
            return 0.0
        if not _exact(_get_field(example, "in_feature"), _get_field(predicted, "in_feature")):
            return 0.0
        if not _exact(_get_field(example, "n_classes"), _get_field(predicted, "n_classes")):
            return 0.0
        return 1.0
    except Exception as exc:
        raise RuntimeError("model_extractor_metric evaluation failed.") from exc
