# Automotive Marketing Content LoRA Studio Demo

Demo academica multimodal para generar campanas automotrices completas:

1. fine-tuning de un LLM con Unsloth + LoRA/QLoRA para producir estrategia, canal, copy, KPIs y prompts visuales;
2. fine-tuning visual con DreamBooth LoRA sobre SDXL para aprender la identidad de un auto o serie;
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

El notebook completo espera un JSON con minimo 200 ejemplos:

```text
data/commercial_campaign_sft/commercial_campaign_sft.json
```

Cada ejemplo debe tener:

```json
{
  "instruction": "Act as an advertising strategist for an automotive dealership. Generate a campaign proposal in JSON.",
  "input": "Goal: Lead Generation | Vehicle: hybrid SUV | Price range: mid-range | Audience: Families 35-44 | Customer sector: urban families | Historical channel: Instagram | City: Miami | Language: English | Duration: 30 Days | Promotion: test drive + financing | ROI: 2.10 | Conversion rate: 0.08 | Engagement: 9",
  "output": {
    "strategy": "Promote safety, family space, and fuel efficiency, closing with a clear invitation to book a test drive.",
    "channel_plan": "Use Instagram for visual awareness and lead generation forms; reinforce with Meta Ads remarketing for interested prospects.",
    "ad_copy": "Give your family more space, technology, and efficiency. Book your test drive today and discover the hybrid SUV built for city life.",
    "image_prompt": "REALCARMODEL real car model in an English Instagram ad for a mid-range hybrid SUV dealership campaign targeting urban families in Miami, bright city background, premium automotive commercial photography, clear space for headline, no readable text",
    "negative_prompt": "blurry, low quality, watermark, distorted text, malformed logo, extra wheels, deformed car, bad perspective",
    "kpis": ["Leads", "Cost per Lead", "Test Drive Bookings", "Conversion Rate", "ROI"],
    "business_note": "Prioritize qualified leads and measure test drive bookings before scaling the media budget."
  }
}
```

Nota de ruta: actualmente hay un archivo en `data/commercial_campaing_sft/commercial_campaign_sft.json`. La carpeta tiene un typo: `campaing`. Para que el notebook lo lea sin cambios, mueve o copia ese archivo a:

```text
data/commercial_campaign_sft/commercial_campaign_sft.json
```

### Diffusion Fine-Tuning

Coloca las imagenes del auto en:

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
./images/real_car_model_01.png,"REALCARMODEL real car model, front three quarter view, metallic blue paint, studio automotive photography, premium lighting"
```

Las captions deben estar en ingles e incluir:

- trigger word unico, por ejemplo `REALCARMODEL`
- marca/modelo o serie del auto
- angulo o vista
- color/materiales
- contexto visual
- tipo de fotografia

Para la demo final se recomiendan 20-40 imagenes con vistas frontal, lateral, trasera, interior, detalles y lifestyle. Para una prueba rapida pueden bastar 8-15 imagenes.

## Requerimientos De GPU

- Recomendado: GPU con 16 GB VRAM o mas.
- Prueba rapida visual en T4: resolucion 512, batch size 1, gradient accumulation 4, 250 steps.
- El LLM se entrena con Qwen 4B en 4-bit usando Unsloth.
- Si la GPU no alcanza para todo en una sesion, ejecuta primero el fine-tuning del LLM y luego el LoRA visual en otra sesion.
- En CPU, los entrenamientos y la generacion SDXL no son practicos.

## Como Correr En Colab O Kaggle

1. Activar GPU.
2. Subir o clonar el repositorio.
3. Copiar el JSON SFT a `data/commercial_campaign_sft/commercial_campaign_sft.json`.
4. Agregar imagenes en `data/car_campaign_lora/images/`.
5. Crear `data/car_campaign_lora/metadata.csv` con `file_path` y `caption`.
6. Abrir `notebooks/proyecto_final_automotive_lora_marketing_full_colab_kaggle.ipynb`.
7. Ejecutar celdas de validacion de datos.
8. Cambiar flags cuando los datos esten listos:

```python
RUN_LLM_TRAINING = True
RUN_VISUAL_TRAINING = True
RUN_IMAGE_GENERATION = True
```

9. Ajustar `BRAND`, `MODEL_SERIES` y `TRIGGER_WORD` segun el modelo real.
10. Ejecutar el notebook de arriba hacia abajo.

## Outputs Esperados

LLM fine-tuned:

```text
outputs/commercial-qwen-lora/
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
outputs/evaluation/
```

Archivos relevantes:

- `outputs/evaluation/llm_base_vs_finetuned_outputs.json`
- `outputs/evaluation/llm_evaluation_report.csv`
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
ROI estimado = ((horas creativas ahorradas * costo hora equipo creativo * campanas mensuales) + (horas comerciales ahorradas * costo hora comercial * propuestas mensuales) - costo operativo IA) / costo operativo IA
```

Ejemplo:

```text
ahorro creativo mensual = 12 * 6 * 35 = USD 2,520
ahorro comercial mensual = 2 * 40 * 25 = USD 2,000
costo IA mensual = USD 300
ROI = (2,520 + 2,000 - 300) / 300 = 14.07x
```

## Limitaciones Conocidas

- La calidad depende de los 200+ ejemplos SFT, las captions y la variedad de imagenes.
- Los modelos de difusion pueden deformar texto o logos; los prompts piden `no readable text`.
- Las imagenes finales requieren revision humana antes de uso comercial.
- Colab/Kaggle pueden cambiar disponibilidad de GPU y memoria.
- Si falla el entrenamiento por VRAM, reduce steps, resolution o ejecuta LLM y LoRA visual en sesiones separadas.
