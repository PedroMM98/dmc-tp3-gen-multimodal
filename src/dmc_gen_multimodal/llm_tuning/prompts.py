"""Prompt formatting helpers for Qwen3 text-only SFT."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any


SYSTEM_INSTRUCTION = (
    "Actua como estratega publicitario senior para una concesionaria de vehiculos. "
    "Responde exclusivamente con un objeto JSON valido que tenga exactamente estos campos: "
    "strategy, recommended_channel, channel_rationale, channel_plan, ad_copy, image_prompt, kpis, business_note."
)


def format_sft_text(record: Mapping[str, Any], system_instruction: str = SYSTEM_INSTRUCTION) -> str:
    response = json.dumps(record["output"], ensure_ascii=False)
    return f"""### Instruction:
{system_instruction}

### Input:
{record['input']}

### Response:
{response}"""


def build_generation_prompt(record: Mapping[str, Any], system_instruction: str = SYSTEM_INSTRUCTION) -> str:
    return f"""### Instruction:
{system_instruction}

### Input:
{record['input']}

### Response:
"""
