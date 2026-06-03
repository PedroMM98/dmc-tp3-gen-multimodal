# Plan De Codigo: AdAgent Creative Campaign Copilot

## Summary

Construir una extension generativa multimodal del chatbot RAG actual para campanas de marketing. El flujo final sera: brief de campana -> contexto RAG -> modelo Qwen fine-tuned con LoRA -> recomendacion/copy/prompt visual -> imagen generada con Diffusers.

La entrega tendra dos superficies: un notebook reproducible en Colab y una pestana nueva en Streamlit para demo local. No se entrenara LoRA visual; el componente de imagenes sera generacion text-to-image con Diffusers.

## Key Changes

- Mantener el RAG actual como base de conocimiento y agregar una capa creativa especializada para marketing.
- Crear un dataset SFT desde `data/corpus/Social_Media_Advertising.csv`:
  - sample deterministico de 300 filas con `random_state=3407`
  - split `240 train / 60 eval`
  - formato JSONL con `instruction`, `input`, `output`
  - salida esperada en JSON con: `campaign_strategy`, `channel_rationale`, `ad_copy`, `image_prompt`, `recommended_kpis`
- Fine-tuning textual con Unsloth + LoRA sobre `unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit`.
- Generacion visual con Diffusers usando:
  - default: `stabilityai/sdxl-turbo` para demo rapida
  - modo calidad opcional: `stabilityai/stable-diffusion-xl-base-1.0`

## Implementation Changes

- Agregar modulo `src/generative/` con:
  - esquemas Pydantic: `CreativeBrief`, `CreativePlan`, `GeneratedAsset`
  - preparacion de dataset y formateo tipo instruct/chat
  - carga de modelo base/fine-tuned para inferencia
  - generacion de imagenes con Diffusers
  - pipeline integrado que combina retrieval + generacion textual + imagen
- Agregar scripts:
  - `scripts/prepare_sft_dataset.py`
  - `scripts/train_adagent_lora.py`
  - `scripts/evaluate_adagent_lora.py`
  - `scripts/generate_creative_assets.py`
- Agregar notebook `notebooks/proyecto_final_adagent_pipeline.ipynb` con celdas secuenciales:
  - instalacion Colab
  - preparacion dataset
  - carga Qwen 4B con Unsloth
  - configuracion LoRA
  - entrenamiento
  - inferencia base vs fine-tuned
  - evaluacion cuantitativa
  - generacion de imagenes
  - demo integrada
- Extender `app/streamlit_app.py` con una tercera pestana `Creative Pipeline`:
  - formulario de brief: objetivo, segmento, canal opcional, idioma, ciudad, duracion, presupuesto aproximado
  - retrieval del corpus existente
  - generacion de plan creativo con modelo fine-tuned si existe
  - fallback a proveedor actual si no hay adapter local
  - boton separado para generar imagen, porque Diffusers puede ser lento sin GPU
  - galeria de outputs guardados en `data/generated_assets/`
- Agregar `requirements-generative.txt` para no romper el entorno RAG actual:
  - `unsloth`, `datasets`, `trl`, `peft`, `accelerate`, `bitsandbytes`
  - `diffusers`, `safetensors`, `Pillow`, `torchvision`

## Public Interfaces

- `CreativeBrief`:
  - `campaign_goal`, `target_audience`, `customer_segment`, `channel`, `language`, `location`, `duration`, `budget_hint`, `business_context`
- `CreativePlan`:
  - `campaign_strategy`, `channel_rationale`, `ad_copy`, `image_prompt`, `recommended_kpis`, `sources`
- CLI:
  - `prepare_sft_dataset.py --input data/corpus/Social_Media_Advertising.csv --output data/processed/adagent_sft_dataset.jsonl`
  - `train_adagent_lora.py --dataset data/processed/adagent_sft_dataset.jsonl --output models/adagent-qwen3-4b-lora`
  - `evaluate_adagent_lora.py --adapter models/adagent-qwen3-4b-lora --output data/processed/finetune_eval_report.json`
  - `generate_creative_assets.py --brief data/examples/creative_brief.json --output data/generated_assets/`

## Test Plan

- Unit tests:
  - dataset builder creates at least 300 examples and preserves required fields
  - train/eval split is deterministic
  - generated output parser accepts valid JSON and handles invalid model text gracefully
  - image prompt builder includes objective, product/segment, visual style, and negative prompt
- Script checks:
  - `prepare_sft_dataset.py` produces JSONL and summary counts
  - `evaluate_adagent_lora.py` compares base vs fine-tuned on the 60 eval examples
- Metrics:
  - training/eval loss
  - JSON structure validity rate
  - Jaccard overlap vs reference output
  - latency per response
  - qualitative table with 5 base vs fine-tuned examples
- Demo acceptance:
  - notebook runs top-to-bottom in Colab GPU
  - Streamlit still supports existing RAG tabs
  - new creative tab can produce a plan without image generation
  - image generation saves at least one PNG when GPU/runtime supports Diffusers

## Assumptions

- La industria final sigue siendo marketing digital/publicidad, usando AdAgent Copilot como marco.
- El fine-tuning obligatorio se cumple con Qwen 4B + Unsloth + LoRA.
- La generacion de imagenes cumple el requisito usando Diffusers sin entrenar un LoRA visual.
- El entrenamiento se ejecutara preferentemente en Google Colab con GPU; la app local sera demo/inferencia y podra funcionar en modo prompt-only si no hay GPU.
