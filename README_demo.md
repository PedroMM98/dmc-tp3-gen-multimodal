# Automotive Marketing Content LoRA Studio Demo

Demo academica para generar piezas visuales de marketing automotriz entrenando un LoRA visual con Diffusers. El flujo toma imagenes del auto + captions, entrena DreamBooth LoRA sobre SDXL y genera assets para web, social, showroom, email, print y highway banner.

## Entregable principal

`notebooks/proyecto_final_automotive_lora_marketing_colab_kaggle.ipynb`

## Requerimientos de GPU

- Recomendado: GPU con 16 GB VRAM o mas.
- Para prueba rapida en T4: resolucion 512, batch size 1, gradient accumulation 4, 250 steps.
- En CPU, el entrenamiento y la generacion SDXL no son practicos.

## Dataset

Coloca las imagenes en:

`data/car_campaign_lora/images/`

Cada imagen debe declararse en `data/car_campaign_lora/metadata.csv`:

```csv
file_path,caption
./images/real_car_model_01.png,"REALCARMODEL real car model, front three quarter view, metallic blue paint, studio automotive photography, premium lighting"
```

Usamos metadata CSV en lugar de archivos `.txt` por imagen porque es mas facil auditar, editar en lote, versionar y validar cuando el dataset use modelos reales. Tambien deja el proyecto listo para scripts caption-aware si luego se decide entrenar con captions por imagen.

Caption recomendado:

```text
REALCARMODEL real car model, front three quarter view, metallic blue paint, studio automotive photography, premium lighting
```

Para la demo final se recomiendan 20-40 imagenes con vistas frontal, lateral, trasera, interior, detalles y lifestyle.

Nota tecnica: el comando DreamBooth LoRA SDXL de esta demo usa un `instance_prompt` comun. El metadata se usa para QA del dataset, trazabilidad, documentacion y construccion de prompts.

## Como correr en Colab o Kaggle

1. Activar GPU.
2. Subir o clonar el repositorio.
3. Agregar imagenes en `data/car_campaign_lora/images/` y captions en `data/car_campaign_lora/metadata.csv`.
4. Abrir el notebook principal.
5. Ejecutar celdas en orden.
6. Ajustar `BRAND`, `MODEL_SERIES` y `TRIGGER_WORD` cuando se defina el modelo real.

## Entrenamiento LoRA visual

El notebook usa:

- `stabilityai/stable-diffusion-xl-base-1.0`
- `train_dreambooth_lora_sdxl.py` de Diffusers
- LoRA rank 16
- 250 steps para demo rapida en free tier

Los adaptadores se guardan en:

`outputs/automotive-lora/`

## Generacion de assets

El notebook genera piezas para:

- website hero banner
- Instagram feed
- vertical story/TikTok
- showroom poster
- highway billboard
- email header

Los PNG se guardan en:

`outputs/generated_assets/`

La metadata se guarda en:

`outputs/evaluation/image_generation_metadata.json`

## LLM opcional

El LLM no se entrena. Solo puede activarse como refinador opcional de prompts y negative prompts. Por defecto, el notebook usa plantillas deterministicas.

## Limitaciones conocidas

- El resultado depende de la calidad y variedad de imagenes/captions.
- Los modelos de difusion pueden deformar texto o logos; los prompts piden `no readable text`.
- Las imagenes finales requieren revision humana antes de uso comercial.
- Si el curso exige fine-tuning de LLM con Unsloth, se debe agregar como extension separada; esta demo se enfoca en LoRA visual.
