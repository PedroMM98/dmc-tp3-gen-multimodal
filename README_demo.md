# Automotive Marketing Content LoRA Studio Demo

Demo academica multimodal para generar campañas automotrices completas:

1. fine-tuning de un LLM con Unsloth + LoRA/QLoRA para producir estrategia, canal, copy, KPIs y prompts visuales;
2. fine-tuning visual con DreamBooth LoRA sobre SDXL para aprender el estilo de campañas del concesionario Autoespar y referencias visuales de autos/logos;
3. generacion de assets para web, social, showroom, email, print y highway banner;
4. comparacion LLM base vs LLM fine-tuned y SDXL base vs SDXL + LoRA visual.

## Entregable Principal

Notebook completo:

```text
notebooks/proyecto_final_automotive_lora_marketing_full_colab_kaggle.ipynb
```

Notebook visual anterior, conservado como referencia:

```text
notebooks/proyecto_final_automotive_lora_marketing_colab_kaggle.ipynb
```

Arquitectura:

```text
docs/demo_architecture.md
```

## Fuentes De Datos Esperadas

### LLM Fine-Tuning

El flujo de LLM ya esta modularizado en `src/` bajo:

```text
src/dmc_gen_multimodal/llm_tuning/
```

La configuracion principal se valida con Pydantic y se carga desde:

```text
configs/llm_tuning/qwen3_text_sft.json
```

El pipeline modular cubre:

- carga y split deterministico del dataset SFT;
- formateo de prompts para entrenamiento y generacion;
- entrenamiento LoRA/QLoRA con Unsloth sobre Qwen3;
- evaluacion baseline vs fine-tuned;
- construccion de metricas y tablas comparativas.

El dataset esperado para este flujo es el JSON SFT corregido con 300 ejemplos:

```text
data/commercial_campaing_sft/commercial_campaign_sft_corrected_300.json
```

Cada ejemplo debe tener:

```json
{
  "instruction": "Act as an advertising strategist for an automotive dealership. Generate a campaign proposal in JSON.",
  "input": "Goal: Lead Generation | Vehicle: hybrid SUV | Price range: mid-range | Audience: Families 35-44 | Customer sector: urban families | Historical channel: Instagram | City: Miami | Language: English | Duration: 30 Days | Promotion: test drive + financing | ROI: 2.10 | Conversion rate: 0.08 | Engagement: 9",
  "output": {
    "strategy": "Promote safety, family space, and fuel efficiency, closing with a clear invitation to book a test drive.",
    "recommended_channel": "Instagram",
    "channel_rationale": "Instagram matches a visual family audience and supports lead forms for test drive intent.",
    "channel_plan": "Use Instagram for visual awareness and lead generation forms; reinforce with Meta Ads remarketing for interested prospects.",
    "ad_copy": "Give your family more space, technology, and efficiency. Book your test drive today and discover the hybrid SUV built for city life.",
    "image_prompt": "AUTOESPAR automotive dealership marketing banner in an English Instagram ad for a mid-range hybrid SUV campaign targeting urban families in Miami, bright city background, premium automotive commercial photography, clear space for headline, no readable text",
    "kpis": ["Leads", "Cost per Lead", "Test Drive Bookings", "Conversion Rate", "ROI"],
    "business_note": "Prioritize qualified leads and measure test drive bookings before scaling the media budget."
  }
}
```

`negative_prompt` no es obligatorio en el JSON SFT. El flujo LLM no lo usa como campo requerido; ese prompt negativo se resuelve despues, en la etapa visual con Diffusers.

Nota de ruta: se conserva la carpeta existente `data/commercial_campaing_sft/`, incluido el typo `campaing`, para mantener compatibilidad con notebooks y outputs previos.

Ejecucion recomendada en entorno Python del proyecto:

```text
requirements-llm-finetuning.txt
src/dmc_gen_multimodal/llm_tuning/
configs/llm_tuning/qwen3_text_sft.json
```

Si se quiere correr el flujo LLM en Colab con GPU T4 o Kaggle, sigue disponible este notebook dedicado:

```text
notebooks/proyecto_final_qwen3_text_llm_sft_colab_kaggle.ipynb
```

Ese notebook ahora funciona como orquestador practico para runtime GPU y tambien incluye una celda final para consumir la version modular basada en `src/`.

Para construir el reporte tecnico del LLM y generar el HTML final de diagnostico:

```text
notebooks/reporte_tecnico_qwen3_evaluation_colab_kaggle.ipynb
```

Ese notebook consume los outputs de evaluacion del LLM y escribe:

```text
outputs/sft_llm_qwen3/evaluation/qwen3_text_technical_report.html
```

### Diffusion Fine-Tuning

Coloca las imagenes de campañas, autos y logos en:

```text
data/car_campaign_lora/images/
```

Cada imagen debe declararse en:

```text
data/car_campaign_lora/metadata.csv
```

Formato:

```csv
file_path,caption
./images/1.autoespar_toyota_corolla_cross_2025.jpg,"AUTOESPAR automotive dealership marketing banner for Toyota Corolla Cross Hybrid Electric 2025. ... includes the conditions/legal disclaimer area at the bottom of the image, usually inside a black, white, or red horizontal rectangle with very small Spanish legal text."
```

Para el dataset actual, `metadata_template.csv` ya esta preparado con 29 filas:

- 23 filas de banners/campañas del concesionario Autoespar.
- 6 filas suplementarias tomadas directamente de `docs/images_descriptions_start/lora_caption_dataset.csv`.

Las primeras 23 captions usan `AUTOESPAR` como trigger word e incluyen:

- estilo de banner/campaña de concesionario Autoespar;
- modelo Toyota, ano, motor, transmision, campaña, garantia y concesionario cuando aplica;
- area de `condiciones` al pie de la imagen, normalmente dentro de un rectangulo negro, blanco o rojo con texto legal pequeno;
- telefono de cotizacion o servicio cuando aparece en la fuente, por ejemplo despues de `cotiza al` o `agenda tu servicio al`.

Las 6 captions suplementarias se mantienen exactamente como vienen en `lora_caption_dataset.csv`, porque son referencias limpias de logos/autos y no piezas publicitarias Autoespar.

Para la demo final se recomiendan 20-40 imagenes con vistas frontal, lateral, trasera, interior, detalles y lifestyle. Para una prueba rapida pueden bastar 8-15 imagenes.

## Requerimientos De GPU

- Recomendado: GPU con 16 GB VRAM o mas.
- Prueba rapida visual en T4: resolucion 512, batch size 1, gradient accumulation 4, 250 steps.
- El LLM text-only se entrena con Qwen3 usando Unsloth + LoRA/QLoRA, con configuracion pensada para Colab T4.
- Si la GPU no alcanza para todo en una sesion, ejecuta primero el fine-tuning del LLM y luego el LoRA visual en otra sesion.
- En CPU, los entrenamientos y la generacion SDXL no son practicos.

## Como Correr En Colab O Kaggle

1. Activar GPU.
2. Subir o clonar el repositorio.
3. Usar el JSON SFT corregido en `data/commercial_campaing_sft/commercial_campaign_sft_corrected_300.json`.
4. Para correr solo el flujo LLM, abrir `notebooks/proyecto_final_qwen3_text_llm_sft_colab_kaggle.ipynb`.
5. Para generar el reporte tecnico HTML del LLM despues de evaluar, abrir `notebooks/reporte_tecnico_qwen3_evaluation_colab_kaggle.ipynb`.
6. Para correr el flujo multimodal completo, abrir `notebooks/proyecto_final_automotive_lora_marketing_full_colab_kaggle.ipynb`.
7. Agregar imagenes en `data/car_campaign_lora/images/` solo si tambien vas a ejecutar el LoRA visual.
8. Usar `data/car_campaign_lora/metadata_template.csv` como base; si el notebook espera `metadata.csv`, copiarlo a `data/car_campaign_lora/metadata.csv`.
9. Ejecutar celdas de validacion de datos.
10. Cambiar flags cuando los datos esten listos:

```python
RUN_LLM_TRAINING = True
RUN_VISUAL_TRAINING = True
RUN_IMAGE_GENERATION = True
```

11. Usar `TRIGGER_WORD = "AUTOESPAR"` para el dataset visual actual.
12. Ejecutar el notebook de arriba hacia abajo.

## Outputs Esperados

LLM fine-tuned:

```text
outputs/sft_llm_qwen3/qwen3-text-commercial-lora/
```

LoRA visual:

```text
outputs/automotive-lora/
```

Assets generados:

```text
outputs/generated_assets/
```

Evaluaciones y metadata:

```text
outputs/sft_llm_qwen3/evaluation/
```

Archivos relevantes:

- `outputs/sft_llm_qwen3/evaluation/qwen3_text_base_outputs.json`
- `outputs/sft_llm_qwen3/evaluation/qwen3_text_finetuned_outputs.json`
- `outputs/sft_llm_qwen3/evaluation/qwen3_text_llm_metrics.csv`
- `outputs/sft_llm_qwen3/evaluation/qwen3_text_comparison_table.csv`
- `outputs/sft_llm_qwen3/evaluation/qwen3_text_technical_report.html`
  Este archivo se genera desde `notebooks/reporte_tecnico_qwen3_evaluation_colab_kaggle.ipynb`.
- `outputs/evaluation/visual_training_dataset_summary.csv`
- `outputs/evaluation/image_generation_metadata.json`
- `outputs/evaluation/sdxl_base_vs_visual_lora_comparison.json`

## Evaluacion

El notebook debe mostrar:

- comparacion LLM base vs LLM fine-tuned sobre ejemplos de evaluacion;
- porcentaje de JSON valido y cobertura de campos esperados;
- comparacion SDXL base vs SDXL + LoRA visual con mismo prompt y seed;
- tabla de assets con prompt, negative prompt, seed, dimensiones, ruta y latencia;
- interpretacion de valor comercial/ROI.

## Valor Comercial

Hipotesis de ROI:

```text
ROI estimado = ((horas creativas ahorradas * costo hora equipo creativo * campañas mensuales) + (horas comerciales ahorradas * costo hora comercial * propuestas mensuales) - costo operativo IA) / costo operativo IA
```

Ejemplo:

```text
ahorro creativo mensual = 12 * 6 * 35 = USD 2,520
ahorro comercial mensual = 2 * 40 * 25 = USD 2,000
costo IA mensual = USD 300
ROI = (2,520 + 2,000 - 300) / 300 = 14.07x
```

## Limitaciones Conocidas

- La calidad depende de los 300 ejemplos SFT corregidos, las captions y la variedad de imagenes.
- Los modelos de difusion pueden deformar texto o logos; los prompts piden `no readable text`.
- Las imagenes finales requieren revision humana antes de uso comercial.
- Colab/Kaggle pueden cambiar disponibilidad de GPU y memoria.
- Si falla el entrenamiento por VRAM, reduce steps, resolution o ejecuta LLM y LoRA visual en sesiones separadas.
