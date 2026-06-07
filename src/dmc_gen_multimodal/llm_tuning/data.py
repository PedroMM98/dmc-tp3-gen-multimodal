"""Dataset loading, validation, and deterministic split helpers."""

from __future__ import annotations

import json
import random
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .config import LLMTuningConfig


def load_sft_records(path: str | Path) -> list[dict[str, Any]]:
    records = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError(f"SFT dataset must be a list, got {type(records).__name__}")
    return records


def validate_sft_records(
    records: Sequence[dict[str, Any]],
    expected_output_fields: set[str],
    require_min_examples: int,
) -> None:
    if len(records) < require_min_examples:
        raise ValueError(f"Expected at least {require_min_examples} SFT examples, got {len(records)}")

    for idx, item in enumerate(records):
        for key in ["instruction", "input", "output"]:
            if key not in item:
                raise ValueError(f"Record {idx} is missing required key: {key}")
        if not isinstance(item["output"], dict):
            raise ValueError(f"Record {idx} output must be a dict")
        missing = expected_output_fields - set(item["output"].keys())
        if missing:
            raise ValueError(f"Record {idx} output is missing fields: {sorted(missing)}")


def deterministic_split(
    records: Sequence[dict[str, Any]],
    train_fraction: float = 0.8,
    seed: int = 3407,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    shuffled = list(records)
    rng = random.Random(seed)
    rng.shuffle(shuffled)
    split_idx = int(len(shuffled) * train_fraction)
    return shuffled[:split_idx], shuffled[split_idx:]


def prepare_sft_data(config: "LLMTuningConfig") -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    config.ensure_directories()
    records = load_sft_records(config.paths.sft_json_path)
    validate_sft_records(
        records=records,
        expected_output_fields=set(config.dataset.expected_output_fields),
        require_min_examples=config.dataset.require_min_sft_examples,
    )
    train_records, eval_records = deterministic_split(
        records,
        train_fraction=config.dataset.train_fraction,
        seed=config.dataset.seed,
    )
    config.paths.sft_train_path.write_text(
        json.dumps(train_records, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    config.paths.sft_eval_path.write_text(
        json.dumps(eval_records, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return train_records, eval_records
