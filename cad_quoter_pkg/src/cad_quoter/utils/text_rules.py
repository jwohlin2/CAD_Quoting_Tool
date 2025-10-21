"""Helpers for loading user-facing text normalization rules."""
from __future__ import annotations

import re
from functools import lru_cache
from typing import Any, Iterable, Mapping

from cad_quoter.resources.loading import load_json


@lru_cache(maxsize=None)
def _load_amortized_rules() -> Mapping[str, Any]:
    return load_json("amortized_label_rules.json")


@lru_cache(maxsize=None)
def amortized_label_pattern() -> re.Pattern[str]:
    """Return the compiled amortized label pattern."""

    rules = _load_amortized_rules()
    pattern_text = str(rules.get("pattern") or "")
    if not pattern_text:
        raise ValueError("amortized_label_rules.json must define a 'pattern' entry")
    flags = 0
    for flag_name in rules.get("pattern_flags", ["IGNORECASE"]):
        flag = getattr(re, str(flag_name), None)
        if flag is None:
            raise ValueError(f"Unsupported regex flag '{flag_name}' in amortized_label_rules.json")
        flags |= flag
    return re.compile(pattern_text, flags)


def _token_sets_contain(token_set: set[str], groups: Iterable[Iterable[str]]) -> bool:
    for group in groups:
        if all(token in token_set for token in group):
            return True
    return False


def canonicalize_amortized_label(label: Any) -> tuple[str, bool]:
    """Return a canonical label and flag for amortized cost rows."""

    text = str(label or "").strip()
    if not text:
        return "", False

    normalized = re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()
    normalized = normalized.replace("perpart", "per part")
    normalized = normalized.replace("perpiece", "per piece")
    tokens = normalized.split()
    token_set = set(tokens)

    rules = _load_amortized_rules()
    amortized_tokens = set(map(str, rules.get("amortized_tokens", [])))
    has_amortized = bool(amortized_tokens & token_set)

    if has_amortized:
        per_part_tokens = [
            tuple(map(str, group))
            for group in rules.get("per_part_token_sets", [])
        ]
        per_part_phrases = [str(p) for p in rules.get("per_part_phrases", [])]
        per_part = _token_sets_contain(token_set, per_part_tokens) or any(
            phrase in normalized for phrase in per_part_phrases
        )

        for entry in rules.get("canonical_labels", []):
            tokens_any = {str(token) for token in entry.get("tokens_any", [])}
            if tokens_any & token_set:
                if per_part and entry.get("per_part"):
                    return str(entry["per_part"]), True
                if entry.get("default"):
                    return str(entry["default"]), True
        return text, True

    match = amortized_label_pattern().search(text)
    if match:
        prefix = text[: match.start()].rstrip()
        canonical = f"{prefix} (amortized)" if prefix else match.group(1).lower()
        return canonical, True

    return text, False


@lru_cache(maxsize=None)
def get_proc_mult_targets() -> Mapping[str, tuple[str, float]]:
    """Return process multiplier targets for propagating derived hours."""

    raw = load_json("proc_mult_targets.json")
    result: dict[str, tuple[str, float]] = {}
    for key, value in raw.items():
        if not isinstance(value, Mapping):
            continue
        label = str(value.get("label") or "").strip()
        if not label:
            continue
        scale_raw = value.get("scale", 1.0)
        try:
            scale = float(scale_raw)
        except Exception as exc:  # pragma: no cover - defensive
            raise TypeError(
                f"Invalid scale for PROC_MULT target '{key}': {scale_raw!r}"
            ) from exc
        result[str(key)] = (label, scale)
    return result


PROC_MULT_TARGETS: Mapping[str, tuple[str, float]] = get_proc_mult_targets()


__all__ = [
    "PROC_MULT_TARGETS",
    "amortized_label_pattern",
    "canonicalize_amortized_label",
    "get_proc_mult_targets",
]
