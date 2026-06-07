"""Pydantic configuration models for the Qwen3 LLM fine-tuning flow."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator


DEFAULT_EXPECTED_FIELDS = (
    "strategy",
    "recommended_channel",
    "channel_rationale",
    "channel_plan",
    "ad_copy",
    "image_prompt",
    "kpis",
    "business_note",
)


class PathConfig(BaseModel):
    """Filesystem layout for data, adapters, and evaluation outputs."""

    project_root: Path = Field(default=Path("."))
    data_root: Path = Field(default=Path("data"))
    sft_dir: Path = Field(default=Path("data/commercial_campaing_sft"))
    sft_json_path: Path = Field(default=Path("data/commercial_campaing_sft/commercial_campaign_sft_corrected_300.json"))
    sft_train_path: Path = Field(default=Path("data/commercial_campaing_sft/train_qwen3_text.json"))
    sft_eval_path: Path = Field(default=Path("data/commercial_campaing_sft/eval_qwen3_text.json"))
    output_root: Path = Field(default=Path("outputs/sft_llm_qwen3"))
    adapter_output_dir: Path = Field(default=Path("outputs/sft_llm_qwen3/qwen3-text-commercial-lora"))
    evaluation_output_dir: Path = Field(default=Path("outputs/sft_llm_qwen3/evaluation"))

    @field_validator("*", mode="before")
    @classmethod
    def coerce_path(cls, value: object) -> object:
        if isinstance(value, str):
            return Path(value)
        return value

    def resolve_against_project_root(self) -> "PathConfig":
        root = self.project_root.expanduser().resolve()

        def resolve(path: Path) -> Path:
            path = path.expanduser()
            return path if path.is_absolute() else root / path

        return self.model_copy(
            update={
                "project_root": root,
                "data_root": resolve(self.data_root),
                "sft_dir": resolve(self.sft_dir),
                "sft_json_path": resolve(self.sft_json_path),
                "sft_train_path": resolve(self.sft_train_path),
                "sft_eval_path": resolve(self.sft_eval_path),
                "output_root": resolve(self.output_root),
                "adapter_output_dir": resolve(self.adapter_output_dir),
                "evaluation_output_dir": resolve(self.evaluation_output_dir),
            }
        )


class ModelConfig(BaseModel):
    base_model: str = "unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit"
    fallback_model: str = "unsloth/Qwen3-1.7B-unsloth-bnb-4bit"
    max_seq_length: int = Field(default=1024, ge=128)
    load_in_4bit: bool = True
    load_in_16bit: bool = False
    fast_inference: bool = False


class DatasetConfig(BaseModel):
    train_fraction: float = Field(default=0.8, gt=0.0, lt=1.0)
    seed: int = 3407
    require_min_sft_examples: int = Field(default=300, ge=1)
    expected_output_fields: tuple[str, ...] = DEFAULT_EXPECTED_FIELDS


class EvaluationConfig(BaseModel):
    max_eval_examples: int = Field(default=3, ge=1)
    max_new_tokens: int = Field(default=512, ge=1)
    base_output_filename: str = "qwen3_text_base_outputs.json"
    finetuned_output_filename: str = "qwen3_text_finetuned_outputs.json"
    metrics_filename: str = "qwen3_text_llm_metrics.csv"
    comparison_table_filename: str = "qwen3_text_comparison_table.csv"


class TrainingConfig(BaseModel):
    lora_r: int = Field(default=8, ge=1)
    lora_alpha: int = Field(default=16, ge=1)
    lora_dropout: float = Field(default=0.0, ge=0.0, le=1.0)
    lora_bias: Literal["none", "all", "lora_only"] = "none"
    target_modules: tuple[str, ...] = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    )
    use_gradient_checkpointing: str = "unsloth"
    per_device_train_batch_size: int = Field(default=1, ge=1)
    gradient_accumulation_steps: int = Field(default=8, ge=1)
    num_train_epochs: float = Field(default=1.0, gt=0)
    learning_rate: float = Field(default=2e-4, gt=0)
    warmup_ratio: float = Field(default=0.05, ge=0, lt=1)
    lr_scheduler_type: str = "linear"
    optim: str = "adamw_8bit"
    weight_decay: float = Field(default=0.01, ge=0)
    logging_steps: int = Field(default=5, ge=1)
    save_strategy: str = "epoch"
    eval_strategy: str = "no"
    report_to: str = "none"


class RunConfig(BaseModel):
    run_baseline_eval: bool = False
    run_llm_training: bool = False
    run_finetuned_eval: bool = False
    build_comparison_report: bool = True


class LLMTuningConfig(BaseModel):
    paths: PathConfig = Field(default_factory=PathConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    run: RunConfig = Field(default_factory=RunConfig)

    @model_validator(mode="after")
    def resolve_paths(self) -> "LLMTuningConfig":
        object.__setattr__(self, "paths", self.paths.resolve_against_project_root())
        return self

    @property
    def base_output_path(self) -> Path:
        return self.paths.evaluation_output_dir / self.evaluation.base_output_filename

    @property
    def finetuned_output_path(self) -> Path:
        return self.paths.evaluation_output_dir / self.evaluation.finetuned_output_filename

    @property
    def metrics_path(self) -> Path:
        return self.paths.evaluation_output_dir / self.evaluation.metrics_filename

    @property
    def comparison_table_path(self) -> Path:
        return self.paths.evaluation_output_dir / self.evaluation.comparison_table_filename

    def ensure_directories(self) -> None:
        for path in [
            self.paths.sft_dir,
            self.paths.output_root,
            self.paths.adapter_output_dir,
            self.paths.evaluation_output_dir,
        ]:
            path.mkdir(parents=True, exist_ok=True)


def load_config(path: str | Path) -> LLMTuningConfig:
    config_path = Path(path)
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    return LLMTuningConfig.model_validate(payload)
