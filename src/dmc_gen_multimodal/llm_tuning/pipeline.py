"""High-level orchestration for the Qwen3 LLM tuning workflow."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .data import load_sft_records, prepare_sft_data
from .generation import run_generation_phase
from .training import train_lora_adapter

if TYPE_CHECKING:
    import pandas as pd

    from .config import LLMTuningConfig


def prepare_data(config: "LLMTuningConfig") -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    return prepare_sft_data(config)


def load_eval_records(config: "LLMTuningConfig") -> list[dict[str, Any]]:
    if config.paths.sft_eval_path.exists():
        return load_sft_records(config.paths.sft_eval_path)
    _, eval_records = prepare_data(config)
    return eval_records


def run_baseline_eval(config: "LLMTuningConfig", eval_records: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    eval_records = eval_records or load_eval_records(config)
    eval_subset = eval_records[: config.evaluation.max_eval_examples]
    return run_generation_phase(
        config=config,
        model_label="base",
        model_name_or_path=config.model.base_model,
        records=eval_subset,
        output_path=config.base_output_path,
    )


def run_training(
    config: "LLMTuningConfig",
    train_records: list[dict[str, Any]] | None = None,
    eval_records: list[dict[str, Any]] | None = None,
) -> None:
    if train_records is None or eval_records is None:
        train_records, eval_records = prepare_data(config)
    train_lora_adapter(config, train_records=train_records, eval_records=eval_records)


def run_finetuned_eval(config: "LLMTuningConfig", eval_records: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    eval_records = eval_records or load_eval_records(config)
    eval_subset = eval_records[: config.evaluation.max_eval_examples]
    if not any(config.paths.adapter_output_dir.glob("*")):
        raise RuntimeError(f"No se encontro adapter en {config.paths.adapter_output_dir}. Ejecuta entrenamiento primero.")
    return run_generation_phase(
        config=config,
        model_label="fine_tuned",
        model_name_or_path=config.paths.adapter_output_dir,
        records=eval_subset,
        output_path=config.finetuned_output_path,
    )


def build_comparison_report(
    config: "LLMTuningConfig",
    eval_records: list[dict[str, Any]] | None = None,
) -> tuple["pd.DataFrame", "pd.DataFrame"]:
    import pandas as pd

    from .metrics import score_llm_outputs

    eval_records = eval_records or load_eval_records(config)
    all_outputs: list[dict[str, Any]] = []
    for path in [config.base_output_path, config.finetuned_output_path]:
        if path.exists():
            all_outputs.extend(json.loads(path.read_text(encoding="utf-8")))
        else:
            print("No existe aun:", path)

    if not all_outputs:
        raise RuntimeError("No hay outputs para comparar. Ejecuta baseline y/o fine-tuned eval primero.")

    metrics_df = score_llm_outputs(all_outputs, expected_fields=set(config.dataset.expected_output_fields))
    config.paths.evaluation_output_dir.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(config.metrics_path, index=False)

    by_model = {(item["example_id"], item["model"]): item for item in all_outputs}
    table_rows = []
    for idx, record in enumerate(eval_records[: config.evaluation.max_eval_examples]):
        base_item = by_model.get((idx, "base"), {})
        ft_item = by_model.get((idx, "fine_tuned"), {})
        table_rows.append(
            {
                "example_id": idx,
                "input": record["input"],
                "expected": json.dumps(record["output"], ensure_ascii=False),
                "base_generated": base_item.get("generated", ""),
                "fine_tuned_generated": ft_item.get("generated", ""),
            }
        )

    comparison_df = pd.DataFrame(table_rows)
    comparison_df.to_csv(config.comparison_table_path, index=False)
    print("Metricas guardadas en:", config.metrics_path)
    print("Tabla comparativa guardada en:", config.comparison_table_path)
    return metrics_df, comparison_df


def run_from_config(config: "LLMTuningConfig") -> None:
    config.ensure_directories()
    train_records, eval_records = prepare_data(config)
    if config.run.run_baseline_eval:
        run_baseline_eval(config, eval_records=eval_records)
    if config.run.run_llm_training:
        run_training(config, train_records=train_records, eval_records=eval_records)
    if config.run.run_finetuned_eval:
        run_finetuned_eval(config, eval_records=eval_records)
    if config.run.build_comparison_report:
        build_comparison_report(config, eval_records=eval_records)
