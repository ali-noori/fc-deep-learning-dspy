"""Score predicted experiment splitter briefs against gold briefs."""

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
        if char.isalnum() or char in {".", "_"}:
            token.append(char)
        elif token:
            cleaned.append("".join(token))
            token = []
    if token:
        cleaned.append("".join(token))
    return {item for item in cleaned if len(item) >= 3 and item not in _STOP}


def _covers(gold: object, predicted: object) -> bool:
    gold_tokens = _tokens(gold)
    if not gold_tokens:
        return bool(str(predicted or "").strip())
    pred_tokens = _tokens(predicted)
    if not pred_tokens:
        return False
    hit = len(gold_tokens & pred_tokens) / len(gold_tokens)
    return hit >= 0.45


def _get_field(obj: object, name: str) -> object | None:
    if hasattr(obj, name):
        return getattr(obj, name)
    if isinstance(obj, dict):
        return obj.get(name)
    return None


def experiment_splitter_metric(example, predicted, trace=None) -> float:
    try:
        fields = (
            "execution_mode_dspy",
            "dataset_dspy",
            "hyper_params_dspy",
            "trainer_dspy",
            "train_config_dspy",
        )
        for name in fields:
            if not _covers(_get_field(example, name), _get_field(predicted, name)):
                return 0.0
        return 1.0
    except Exception as exc:
        raise RuntimeError("experiment_splitter_metric evaluation failed.") from exc
