"""Evaluation metrics for generated campaign JSON outputs."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd

from .config import DEFAULT_EXPECTED_FIELDS


EXPECTED_SFT_OUTPUT_FIELDS = set(DEFAULT_EXPECTED_FIELDS)


def extract_json_object(text: Any) -> dict[str, Any] | None:
    if isinstance(text, dict):
        return text
    if not isinstance(text, str):
        return None
    text = text.strip()
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        pass
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return None
    try:
        parsed = json.loads(match.group(0))
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        return None


def tokens_for_jaccard(value: Any) -> set[str]:
    return set(re.findall(r"\w+", str(value).lower()))


def jaccard(a: Any, b: Any) -> float:
    aw = tokens_for_jaccard(a)
    bw = tokens_for_jaccard(b)
    if not aw and not bw:
        return 1.0
    return len(aw & bw) / max(1, len(aw | bw))


def list_jaccard(a: Sequence[Any] | None, b: Sequence[Any] | None) -> float:
    aset = {str(x).strip().lower() for x in (a or [])}
    bset = {str(x).strip().lower() for x in (b or [])}
    if not aset and not bset:
        return 1.0
    return len(aset & bset) / max(1, len(aset | bset))


def score_generated_record(
    item: Mapping[str, Any],
    expected_fields: set[str] | None = None,
) -> dict[str, Any]:
    expected_fields = expected_fields or EXPECTED_SFT_OUTPUT_FIELDS
    parsed = extract_json_object(item.get("generated", ""))
    expected = item.get("expected", {})
    json_valid = parsed is not None
    if not parsed:
        parsed = {}
    expected_channel = str(expected.get("recommended_channel", "")).strip().lower()
    generated_channel = str(parsed.get("recommended_channel", "")).strip().lower()
    expected_text = json.dumps(expected, ensure_ascii=False, sort_keys=True)
    generated_text = json.dumps(parsed, ensure_ascii=False, sort_keys=True)
    return {
        "example_id": item.get("example_id"),
        "model": item.get("model"),
        "json_valid": float(json_valid),
        "field_coverage": len(expected_fields & set(parsed.keys())) / len(expected_fields),
        "recommended_channel_match": float(bool(expected_channel) and expected_channel == generated_channel),
        "kpi_jaccard": list_jaccard(expected.get("kpis"), parsed.get("kpis")),
        "text_jaccard_vs_expected": jaccard(expected_text, generated_text),
        "image_prompt_present": float(bool(str(parsed.get("image_prompt", "")).strip())),
        "latency_sec": item.get("latency_sec"),
    }


def score_llm_outputs(
    outputs: Sequence[Mapping[str, Any]],
    expected_fields: set[str] | None = None,
) -> pd.DataFrame:
    return pd.DataFrame([score_generated_record(item, expected_fields=expected_fields) for item in outputs])
