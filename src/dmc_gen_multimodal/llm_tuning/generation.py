"""Generation utilities for Qwen3 models patched by Unsloth."""

from __future__ import annotations

import gc
import json
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .prompts import build_generation_prompt

if TYPE_CHECKING:
    from .config import LLMTuningConfig


def validate_hf_config_available(model_name_or_path: str | Path) -> None:
    model_ref = str(model_name_or_path)
    if Path(model_ref).exists() or "/" not in model_ref:
        return
    try:
        from huggingface_hub import hf_hub_download

        hf_hub_download(repo_id=model_ref, filename="config.json")
    except Exception as exc:
        raise RuntimeError(
            f"No se pudo descargar config.json de {model_ref}. "
            "Verifica conexion a Hugging Face, nombre del modelo y permisos del runtime."
        ) from exc


def cleanup_model(model: Any = None, tokenizer: Any = None) -> None:
    if model is not None:
        del model
    if tokenizer is not None:
        del tokenizer
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass


def load_qwen3_for_generation(config: "LLMTuningConfig", model_name_or_path: str | Path):
    validate_hf_config_available(model_name_or_path)
    try:
        from unsloth import FastModel

        model, tokenizer = FastModel.from_pretrained(
            model_name=str(model_name_or_path),
            max_seq_length=config.model.max_seq_length,
            load_in_4bit=config.model.load_in_4bit,
            load_in_16bit=config.model.load_in_16bit,
            fast_inference=config.model.fast_inference,
        )
        FastModel.for_inference(model)
    except Exception as exc:
        raise RuntimeError(
            f"No se pudo cargar {model_name_or_path} con FastModel. "
            "Verifica runtime limpio, dependencias y disponibilidad del modelo."
        ) from exc

    if getattr(tokenizer, "pad_token_id", None) is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def generate_one(
    model: Any,
    tokenizer: Any,
    prompt: str,
    max_seq_length: int,
    max_new_tokens: int,
) -> str:
    import torch

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_seq_length).to(model.device)
    input_len = inputs["input_ids"].shape[-1]
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = outputs[0][input_len:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def run_generation_phase(
    config: "LLMTuningConfig",
    model_label: str,
    model_name_or_path: str | Path,
    records: Sequence[Mapping[str, Any]],
    output_path: str | Path,
) -> list[dict[str, Any]]:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model, tokenizer = load_qwen3_for_generation(config, model_name_or_path)
    results: list[dict[str, Any]] = []
    try:
        for idx, record in enumerate(records):
            prompt = build_generation_prompt(record)
            start = time.perf_counter()
            generated = generate_one(
                model,
                tokenizer,
                prompt,
                max_seq_length=config.model.max_seq_length,
                max_new_tokens=config.evaluation.max_new_tokens,
            )
            latency = time.perf_counter() - start
            results.append(
                {
                    "example_id": idx,
                    "model": model_label,
                    "input": record["input"],
                    "expected": record["output"],
                    "generated": generated,
                    "latency_sec": latency,
                }
            )
            print(f"{model_label} example {idx}: {latency:.2f}s")
        output_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        print("Outputs guardados en:", output_path)
        return results
    finally:
        cleanup_model(model, tokenizer)
