"""LoRA/QLoRA training utilities for the Qwen3 text-only campaign model."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

from .generation import cleanup_model, validate_hf_config_available
from .prompts import format_sft_text

if TYPE_CHECKING:
    from .config import LLMTuningConfig


def train_lora_adapter(
    config: "LLMTuningConfig",
    train_records: Sequence[Mapping[str, Any]],
    eval_records: Sequence[Mapping[str, Any]] | None = None,
) -> None:
    import torch
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer
    from unsloth import FastModel

    if not torch.cuda.is_available():
        raise RuntimeError("Se requiere GPU para entrenar Qwen3 text-only con LoRA.")

    validate_hf_config_available(config.model.base_model)
    model, tokenizer = FastModel.from_pretrained(
        model_name=config.model.base_model,
        max_seq_length=config.model.max_seq_length,
        load_in_4bit=config.model.load_in_4bit,
        load_in_16bit=config.model.load_in_16bit,
        fast_inference=config.model.fast_inference,
    )

    model = FastModel.get_peft_model(
        model,
        r=config.training.lora_r,
        target_modules=list(config.training.target_modules),
        lora_alpha=config.training.lora_alpha,
        lora_dropout=config.training.lora_dropout,
        bias=config.training.lora_bias,
        use_gradient_checkpointing=config.training.use_gradient_checkpointing,
        random_state=config.dataset.seed,
        max_seq_length=config.model.max_seq_length,
    )

    train_dataset = Dataset.from_list([{"text": format_sft_text(record)} for record in train_records])
    eval_dataset = Dataset.from_list([{"text": format_sft_text(record)} for record in (eval_records or [])]) if eval_records else None

    training_args = SFTConfig(
        output_dir=str(config.paths.adapter_output_dir),
        dataset_text_field="text",
        max_seq_length=config.model.max_seq_length,
        per_device_train_batch_size=config.training.per_device_train_batch_size,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        num_train_epochs=config.training.num_train_epochs,
        learning_rate=config.training.learning_rate,
        warmup_ratio=config.training.warmup_ratio,
        lr_scheduler_type=config.training.lr_scheduler_type,
        optim=config.training.optim,
        weight_decay=config.training.weight_decay,
        logging_steps=config.training.logging_steps,
        save_strategy=config.training.save_strategy,
        eval_strategy=config.training.eval_strategy,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        seed=config.dataset.seed,
        report_to=config.training.report_to,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    try:
        trainer.train()
        model.save_pretrained(str(config.paths.adapter_output_dir))
        tokenizer.save_pretrained(str(config.paths.adapter_output_dir))
        print("Adapter guardado en:", config.paths.adapter_output_dir)
    finally:
        cleanup_model(model, tokenizer)
