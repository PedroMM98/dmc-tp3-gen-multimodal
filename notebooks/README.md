# Notebooks Guide

This folder contains the Colab/Kaggle notebooks used for the multimodal automotive marketing demo. Use GPU runtimes for training notebooks; local execution is mainly for inspection and light validation.

## Recommended Order

1. Text model fine-tuning:

```text
proyecto_final_qwen3_text_llm_sft_colab_kaggle.ipynb
```

Use this to train/evaluate the Qwen3 text model with Unsloth + LoRA/QLoRA. It reads the SFT dataset from:

```text
data/commercial_campaing_sft/commercial_campaign_sft_corrected_300.json
```

Main outputs:

```text
outputs/sft_llm_qwen3/
outputs/sft_llm_qwen3/evaluation/
```

Key report:

```text
outputs/sft_llm_qwen3/evaluation/qwen3_text_technical_report.html
```

2. Visual SDXL LoRA fine-tuning:

```text
proyecto_final_diffusion_lora_colab_drive.ipynb
```

Use this to train/evaluate the Stable Diffusion XL DreamBooth LoRA. It reads images and metadata from:

```text
data/car_campaign_lora/images/
data/car_campaign_lora/image_metadata.csv
```

The metadata must include:

```text
file_path, caption, training_caption
```

The notebook prepares normalized `.png` images and matching `.txt` sidecar captions from `training_caption`. The shared `INSTANCE_PROMPT` is only a fallback.

Main outputs:

```text
outputs/stable_diffusion_lora/
outputs/stable_diffusion_lora/lora_adapter/
outputs/stable_diffusion_lora/evaluation/
outputs/stable_diffusion_lora/generated/
```

Key report:

```text
outputs/stable_diffusion_lora/evaluation/stable_diffusion_lora_technical_report.html
```

3. Qwen3 technical report only:

```text
reporte_tecnico_qwen3_evaluation_colab_kaggle.ipynb
```

Use this after the text notebook has produced evaluation files. It does not train a model; it reads:

```text
outputs/sft_llm_qwen3/evaluation/
```

and writes:

```text
outputs/sft_llm_qwen3/evaluation/qwen3_text_technical_report.html
```

4. Caption generation helper:

```text
caption_generator_using-granite_vision_4_1_4b.ipynb
```

Use this when new training images need initial visual descriptions. Review and clean generated captions before putting them into `training_caption`.

5. Full multimodal notebook:

```text
proyecto_final_automotive_lora_marketing_full_colab_kaggle.ipynb
```

Use this for the end-to-end demo path. For debugging or repeatable runs, prefer the dedicated text and diffusion notebooks above.

## Runtime Notes

- Use Colab/Kaggle GPU for training.
- For SDXL LoRA, start with a small/smoke run before longer training.
- If Colab runs out of memory after training but the LoRA adapter was saved, set `RUN_TRAINING = False` and continue generation from the saved adapter.
- For diffusion generation, `RUN_BASELINE_GENERATION = False` and `RUN_LORA_GENERATION = True` reduces memory use.
- Avoid committing temporary notebook cache folders such as `.ipynb_checkpoints`.

## Output Reading Guide

For text model results, start with:

```text
outputs/sft_llm_qwen3/evaluation/qwen3_text_technical_report.html
```

Then inspect:

```text
qwen3_text_llm_metrics.csv
qwen3_text_comparison_table.csv
qwen3_text_finetuned_outputs.json
```

For diffusion results, start with:

```text
outputs/stable_diffusion_lora/evaluation/stable_diffusion_lora_technical_report.html
```

Then inspect:

```text
outputs/stable_diffusion_lora/generated/comparison_canvases/
outputs/stable_diffusion_lora/evaluation/base_vs_lora_metrics.csv
outputs/stable_diffusion_lora/evaluation/caption_training_manifest.csv
outputs/stable_diffusion_lora/lora_adapter/pytorch_lora_weights.safetensors
```

