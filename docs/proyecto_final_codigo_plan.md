# Plan De Codigo: Automotive Marketing Content LoRA Studio

## Summary

Construir una extension generativa multimodal para campanas automotrices. El flujo final sera: brief de campana -> modelo Qwen fine-tuned con LoRA/QLoRA -> recomendacion/copy/KPIs/prompt visual -> SDXL base + DreamBooth LoRA visual -> assets publicitarios generados con Diffusers.

La entrega principal sera un notebook reproducible en Colab o Kaggle. Si se mantiene la demo local, Streamlit funcionara como superficie opcional de inferencia y visualizacion, no como requisito para entrenar.

El proyecto tiene dos entradas de datos separadas:

- LLM fine-tuning: archivo JSON con minimo 200 ejemplos SFT en el formato `instruction`, `input`, `output` propuesto en `docs/proyecto_final_comercial_demo_notebook_plan.md`.
- Diffusion fine-tuning: carpeta de imagenes del auto y `metadata.csv` con una fila por imagen, usando `file_path` y `caption`.

## Key Changes

- Agregar una capa creativa especializada para marketing automotriz.
- Crear o cargar un dataset SFT del LLM desde `data/commercial_campaign_sft/commercial_campaign_sft.json`:
  - minimo 200 ejemplos
  - recomendado: 300 ejemplos con split `240 train / 60 eval`
  - formato JSON como lista de objetos con `instruction`, `input`, `output`
  - opcional: generar ese JSON desde `data/Social_Media_Advertising.csv` con enriquecimiento automotriz deterministico
  - salida esperada en JSON con: `strategy`, `channel_plan`, `ad_copy`, `image_prompt`, `negative_prompt`, `kpis`, `business_note`
- Fine-tuning textual con Unsloth + LoRA sobre `unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit`.
- Crear dataset visual DreamBooth desde `data/car_campaign_lora/metadata.csv`:
  - columnas obligatorias: `file_path`, `caption`
  - las imagenes viven en `data/car_campaign_lora/images/`
  - las captions deben incluir el trigger word visual, por ejemplo `REALCARMODEL`
- Fine-tuning visual con Diffusers + DreamBooth LoRA sobre `stabilityai/stable-diffusion-xl-base-1.0`.
- Generacion visual con SDXL base + LoRA visual usando prompts producidos por el LLM fine-tuned y ajustados por placement.

## Implementation Changes

- Agregar modulo `src/generative/` con:
  - esquemas Pydantic: `CreativeBrief`, `CreativePlan`, `GeneratedAsset`
  - preparacion/carga del dataset SFT JSON y formateo tipo instruct/chat
  - validacion del `metadata.csv` visual y copia normalizada para DreamBooth
  - carga de modelo base/fine-tuned para inferencia
  - carga de SDXL base con adapter LoRA visual
  - generacion de imagenes con Diffusers por placement
  - pipeline integrado que combina generacion textual + prompt builder + generacion visual
- Agregar scripts:
  - `scripts/build_commercial_sft_dataset.py`
  - `scripts/validate_visual_metadata.py`
  - `scripts/train_commercial_llm_lora.py`
  - `scripts/evaluate_commercial_llm_lora.py`
  - `scripts/train_visual_dreambooth_lora.py`
  - `scripts/generate_campaign_assets.py`
- Agregar notebook `notebooks/proyecto_final_automotive_lora_marketing_colab_kaggle.ipynb` con celdas secuenciales:
  - instalacion Colab
  - preparacion/carga del dataset SFT JSON del LLM
  - carga Qwen 4B con Unsloth
  - configuracion LoRA
  - entrenamiento LLM
  - inferencia LLM base vs fine-tuned
  - evaluacion cuantitativa del LLM
  - carga de imagenes y `metadata.csv`
  - entrenamiento DreamBooth LoRA visual
  - comparacion SDXL base vs SDXL con LoRA visual
  - generacion de imagenes por placement
  - demo integrada end-to-end
- Opcional: extender `app/streamlit_app.py` con una pestana `Creative Pipeline`:
  - formulario de brief: objetivo, segmento, canal opcional, idioma, ciudad, duracion, presupuesto aproximado
  - generacion de plan creativo con modelo fine-tuned si existe
  - fallback a proveedor actual si no hay adapter local
  - boton separado para generar imagen, porque SDXL + LoRA puede ser lento sin GPU
  - galeria de outputs guardados en `data/generated_assets/`
- Agregar `requirements-generative.txt` para aislar dependencias pesadas de entrenamiento e inferencia generativa:
  - `unsloth`, `datasets`, `trl`, `peft`, `accelerate`, `bitsandbytes`
  - `diffusers`, `safetensors`, `Pillow`, `torchvision`

## Public Interfaces

- `CreativeBrief`:
  - `brand`, `model_series`, `trigger_word`, `campaign_goal`, `vehicle_type`, `vehicle_model`, `price_range`, `target_audience`, `customer_sector`, `preferred_channels`, `language`, `location`, `duration`, `promotion_type`, `budget_hint`, `tone`
- `CreativePlan`:
  - `strategy`, `channel_plan`, `ad_copy`, `image_prompt`, `negative_prompt`, `kpis`, `business_note`
- `VisualTrainingExample`:
  - `file_path`, `caption`
- CLI:
  - `build_commercial_sft_dataset.py --input data/Social_Media_Advertising.csv --output data/commercial_campaign_sft/commercial_campaign_sft.json --min-examples 200`
  - `train_commercial_llm_lora.py --dataset data/commercial_campaign_sft/commercial_campaign_sft.json --output outputs/commercial-qwen-lora`
  - `evaluate_commercial_llm_lora.py --adapter outputs/commercial-qwen-lora --eval data/commercial_campaign_sft/eval.json --output outputs/evaluation/llm_evaluation_report.json`
  - `validate_visual_metadata.py --metadata data/car_campaign_lora/metadata.csv --images data/car_campaign_lora/images`
  - `train_visual_dreambooth_lora.py --metadata data/car_campaign_lora/metadata.csv --images data/car_campaign_lora/images --output outputs/automotive-lora`
  - `generate_campaign_assets.py --brief data/examples/campaign_brief.json --llm-adapter outputs/commercial-qwen-lora --visual-lora outputs/automotive-lora --output outputs/generated_assets/`

## Test Plan

- Unit tests:
  - SFT dataset loader accepts JSON with at least 200 examples and preserves required fields
  - optional CSV-to-SFT builder creates at least 300 examples when using `Social_Media_Advertising.csv`
  - train/eval split is deterministic
  - generated output parser accepts valid JSON and handles invalid model text gracefully
  - visual metadata validator checks `file_path`, `caption`, existing image paths and trigger word coverage
  - image prompt builder includes objective, vehicle/product, segment, visual style, placement and negative prompt
- Script checks:
  - `build_commercial_sft_dataset.py` produces JSON and summary counts
  - `evaluate_commercial_llm_lora.py` compares base vs fine-tuned on the eval examples
  - `validate_visual_metadata.py` fails clearly if the CSV references missing images
- Metrics:
  - training/eval loss
  - JSON structure validity rate
  - Jaccard overlap vs reference output
  - latency per response
  - qualitative table with 5 base vs fine-tuned examples
  - visual training loss if reported by Diffusers
  - number of images/captions used for DreamBooth
  - asset generation latency by placement
- Demo acceptance:
  - notebook runs top-to-bottom in Colab GPU
  - LLM adapter is saved in `outputs/commercial-qwen-lora/`
  - visual adapter is saved in `outputs/automotive-lora/`
  - base vs fine-tuned LLM comparison is shown
  - SDXL base vs SDXL with LoRA visual comparison is shown
  - generated assets save PNG files and metadata in `outputs/generated_assets/`

## Assumptions

- La industria final es marketing automotriz/publicidad para concesionarias o equipos comerciales.
- El fine-tuning obligatorio se cumple con Qwen 4B + Unsloth + LoRA.
- El dataset SFT del LLM puede venir ya preparado como JSON o generarse desde un CSV comercial enriquecido.
- La generacion de imagenes cumple el requisito usando Diffusers y ademas entrena un DreamBooth LoRA visual.
- El input principal del modelo de difusion es `metadata.csv` + imagenes; el input principal del fine-tuning LLM es JSON con 200+ ejemplos.
- El entrenamiento se ejecutara preferentemente en Google Colab con GPU; la app local sera demo/inferencia y podra funcionar en modo prompt-only si no hay GPU.
