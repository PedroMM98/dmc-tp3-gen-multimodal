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

Notebooks actuales de trabajo:

```text
notebooks/proyecto_final_automotive_lora_marketing_full_colab_kaggle.ipynb
notebooks/proyecto_final_qwen3_text_llm_sft_colab_kaggle.ipynb
notebooks/proyecto_final_diffusion_lora_colab_drive.ipynb
notebooks/reporte_tecnico_qwen3_evaluation_colab_kaggle.ipynb
notebooks/caption_generator_using-granite_vision_4_1_4b.ipynb
```

Notebook visual anterior, conservado como referencia:

```text
notebooks/archive/proyecto_final_automotive_lora_marketing_colab_kaggle.ipynb
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
data/car_campaign_lora/image_metadata.csv
```

Formato:

```csv
file_path,caption,training_caption
./images/1.autoespar_toyota_corolla_cross_2025.jpg,"source/audit description","AUTOESPAR dealership automotive advertisement, Toyota Corolla Cross, clean promotional banner layout, vehicle-focused composition, bold red white and black graphic blocks, blank copy area, small unreadable legal footer bar"
```

Para el dataset actual, `image_metadata.csv` ya esta preparado con 154 filas:

- 108 filas de banners/campañas del concesionario Autoespar.
- 40 filas de referencias Toyota para estilo visual automotriz.
- 6 filas suplementarias tomadas directamente de `docs/images_descriptions_start/lora_caption_dataset.csv`.

El CSV conserva dos textos por imagen:

- `caption`: descripcion fuente/auditoria.
- `training_caption`: caption limpia que se usa para entrenar DreamBooth LoRA.

Las `training_caption` mantienen los trigger words (`AUTOESPAR` para campanas del concesionario y `Toyota automotive advertising reference` para referencias Toyota), pero evitan precios exactos, telefonos, fechas, garantias, texto legal y copys literales. La notebook normaliza las imagenes a `.png`, escribe sidecars `.txt` con `training_caption`, y usa `INSTANCE_PROMPT` solo como fallback.

Por defecto, `proyecto_final_diffusion_lora_colab_drive.ipynb` entrena con las primeras 65 imagenes validadas. Cambia `TRAINING_IMAGE_LIMIT` a `None`, `0` o `""` para usar las 154 filas.

Para la demo final se recomiendan 20-40 imagenes con vistas frontal, lateral, trasera, interior, detalles y lifestyle. Para una prueba rapida pueden bastar 8-15 imagenes.

## Requerimientos De GPU

- Recomendado: GPU con 16 GB VRAM o mas.
- Prueba rapida visual en T4: resolucion 512, batch size 1, gradient accumulation 4, 250 steps.
- El LLM text-only se entrena con Qwen3 usando Unsloth + LoRA/QLoRA, con configuracion pensada para Colab T4.
- Si la GPU no alcanza para todo en una sesion, ejecuta primero el fine-tuning del LLM y luego el LoRA visual en otra sesion.
- En CPU, los entrenamientos y la generacion SDXL no son practicos.

## Guia De Notebooks Actuales

Esta es la forma recomendada de usar los notebooks del proyecto. Los notebooks estan pensados principalmente para Colab o Kaggle con GPU; la ejecucion local sirve para inspeccion, preparacion de datos, pruebas chicas o para correr partes modulares, pero no es recomendable para entrenar modelos grandes sin una GPU NVIDIA con suficiente VRAM.

### 1. Notebook Multimodal Completo

```text
notebooks/proyecto_final_automotive_lora_marketing_full_colab_kaggle.ipynb
```

Usalo cuando quieras correr la demo final completa: fine-tuning del LLM Qwen3, evaluacion textual, entrenamiento LoRA visual SDXL y generacion de assets de campana.

Requisitos previos:

- GPU activa en Colab o Kaggle.
- Dataset SFT disponible en `data/commercial_campaing_sft/commercial_campaign_sft_corrected_300.json`.
- Imagenes de entrenamiento visual en `data/car_campaign_lora/images/`.
- Metadata visual en `data/car_campaign_lora/image_metadata.csv`.
- Trigger word visual: `AUTOESPAR`.

Ejecucion sugerida:

1. Abrir el notebook en Colab o Kaggle.
2. Activar GPU antes de ejecutar celdas.
3. Subir o clonar el repositorio completo.
4. Ejecutar las celdas de instalacion y diagnostico de entorno.
5. Ejecutar las celdas de validacion de datasets.
6. Activar los flags necesarios:

```python
RUN_LLM_TRAINING = True
RUN_VISUAL_TRAINING = True
RUN_IMAGE_GENERATION = True
```

7. Si la memoria de GPU no alcanza, correr primero el bloque LLM y luego, en otra sesion, el bloque visual.
8. Revisar los outputs en `outputs/sft_llm_qwen3/`, `outputs/stable_diffusion_lora/` y `outputs/generated_assets/`.

### 2. Notebook Solo LLM Qwen3

```text
notebooks/proyecto_final_qwen3_text_llm_sft_colab_kaggle.ipynb
```

Usalo cuando quieras entrenar y evaluar solamente el modelo textual. Es el camino mas simple para probar el flujo SFT con Unsloth + LoRA/QLoRA y comparar Qwen3 base vs Qwen3 fine-tuned.

Requisitos previos:

- GPU T4, L4, A100 o equivalente.
- Dataset SFT en `data/commercial_campaing_sft/commercial_campaign_sft_corrected_300.json`.
- Configuracion modular en `configs/llm_tuning/qwen3_text_sft.json`.
- Dependencias base listadas en `requirements-llm-finetuning.txt`; en Colab/Kaggle, `torch` CUDA y `unsloth` se instalan desde las celdas del notebook porque dependen del runtime.

Ejecucion sugerida:

1. Abrir el notebook con GPU activa.
2. Ejecutar la instalacion de dependencias.
3. Validar que el dataset tenga los campos `instruction`, `input` y `output`.
4. Ejecutar entrenamiento, generacion y evaluacion.
5. Confirmar que se generen los archivos de evaluacion:

```text
outputs/sft_llm_qwen3/evaluation/qwen3_text_base_outputs.json
outputs/sft_llm_qwen3/evaluation/qwen3_text_finetuned_outputs.json
outputs/sft_llm_qwen3/evaluation/qwen3_text_llm_metrics.csv
outputs/sft_llm_qwen3/evaluation/qwen3_text_comparison_table.csv
```

Tambien puede consumir el pipeline modular de `src/dmc_gen_multimodal/llm_tuning/` usando la configuracion JSON.

### 3. Notebook Solo Diffusion LoRA SDXL

```text
notebooks/proyecto_final_diffusion_lora_colab_drive.ipynb
```

Usalo cuando quieras entrenar solamente el LoRA visual SDXL con Diffusers, guardar artefactos en Google Drive y comparar SDXL base vs SDXL + LoRA.

Requisitos previos:

- GPU activa; para pruebas rapidas T4 puede funcionar con resolucion 512, batch size 1, gradient accumulation 4 y pocos steps.
- Imagenes en `data/car_campaign_lora/images/`.
- Metadata en `data/car_campaign_lora/image_metadata.csv`.
- Google Drive montado si se quieren persistir pesos y resultados fuera de la sesion de Colab.
- Trigger word `AUTOESPAR` para conservar consistencia con el dataset actual.

Ejecucion sugerida:

1. Abrir el notebook en Colab.
2. Activar GPU y montar Drive si se van a guardar checkpoints.
3. Ejecutar instalacion de `diffusers`, `accelerate`, `peft`, `transformers` y dependencias visuales desde el notebook.
4. Validar imagenes y metadata.
5. Ejecutar entrenamiento LoRA.
6. Generar comparaciones con el mismo prompt y seed para SDXL base y SDXL + LoRA.
7. Revisar resultados en `outputs/stable_diffusion_lora/generated/` y `outputs/stable_diffusion_lora/evaluation/`.

### 4. Notebook De Reporte Tecnico Qwen3

```text
notebooks/reporte_tecnico_qwen3_evaluation_colab_kaggle.ipynb
```

Usalo despues de correr el notebook de LLM. No entrena modelos; lee los outputs de evaluacion y genera un HTML autocontenido de diagnostico.

Requisitos previos:

- Tener generados los archivos en `outputs/sft_llm_qwen3/evaluation/`.
- Como minimo, `qwen3_text_llm_metrics.csv` y `qwen3_text_finetuned_outputs.json`.
- Opcionalmente, `qwen3_text_base_outputs.json` y `qwen3_text_comparison_table.csv` para un reporte mas completo.

Ejecucion sugerida:

1. Abrir el notebook.
2. Verificar que `TECHNICAL_EVAL_DIR` apunte a `outputs/sft_llm_qwen3/evaluation/`.
3. Ejecutar todas las celdas.
4. Abrir el reporte generado:

```text
outputs/sft_llm_qwen3/evaluation/qwen3_text_technical_report.html
```

### 5. Notebook De Captions Con Granite Vision

```text
notebooks/caption_generator_using-granite_vision_4_1_4b.ipynb
```

Usalo para generar o revisar captions de imagenes con el modelo visual `granite-vision-4.1-4b` de IBM. Es util como apoyo para preparar el dataset visual antes del entrenamiento LoRA.

Requisitos previos:

- Imagenes de campana disponibles para procesar.
- Runtime con GPU recomendado para inferencia visual.
- Dependencias del notebook instaladas en la sesion.

Ejecucion sugerida:

1. Abrir el notebook.
2. Ajustar la ruta de imagenes a la carpeta de entrada que se vaya a procesar.
3. Ejecutar la carga del modelo visual.
4. Generar captions.
5. Revisar manualmente las captions antes de incorporarlas a `data/car_campaign_lora/image_metadata.csv`.

## Ejecucion Local Opcional

La forma mas practica de correr entrenamientos sigue siendo Colab/Kaggle. Para trabajar localmente con el flujo modular de LLM, preparar datos o ejecutar partes livianas:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements-llm-finetuning.txt
```

Si usas Conda:

```powershell
conda create -n dmc-gen-multimodal python=3.11
conda activate dmc-gen-multimodal
pip install -r requirements-llm-finetuning.txt
```

Notas:

- Instala `torch` con CUDA segun tu GPU y version de CUDA antes de entrenar.
- Instala `unsloth` siguiendo las celdas del notebook o la version compatible con tu runtime.
- Para abrir notebooks localmente puedes usar Jupyter:

```powershell
pip install notebook
jupyter notebook
```

En local, evita correr entrenamiento SDXL o Qwen3 si no tienes GPU suficiente. Para notebooks pesados, abre el archivo, revisa rutas/configuracion y ejecuta en Colab/Kaggle.

## Como Correr En Colab O Kaggle

1. Activar GPU.
2. Subir o clonar el repositorio.
3. Usar el JSON SFT corregido en `data/commercial_campaing_sft/commercial_campaign_sft_corrected_300.json`.
4. Elegir el notebook segun el flujo:
   - multimodal completo: `notebooks/proyecto_final_automotive_lora_marketing_full_colab_kaggle.ipynb`;
   - solo LLM: `notebooks/proyecto_final_qwen3_text_llm_sft_colab_kaggle.ipynb`;
   - solo visual SDXL LoRA: `notebooks/proyecto_final_diffusion_lora_colab_drive.ipynb`;
   - reporte tecnico: `notebooks/reporte_tecnico_qwen3_evaluation_colab_kaggle.ipynb`;
   - captions visuales: `notebooks/caption_generator_using-granite_vision_4_1_4b.ipynb`.
5. Agregar imagenes en `data/car_campaign_lora/images/` solo si tambien vas a ejecutar el LoRA visual.
6. Usar `data/car_campaign_lora/image_metadata.csv` como metadata visual canonica.
7. Ejecutar celdas de validacion de datos.
8. Cambiar flags cuando los datos esten listos:

```python
RUN_LLM_TRAINING = True
RUN_VISUAL_TRAINING = True
RUN_IMAGE_GENERATION = True
```

9. Usar `TRIGGER_WORD = "AUTOESPAR"` para el dataset visual actual.
10. Ejecutar el notebook elegido de arriba hacia abajo.

## Outputs Esperados

Los outputs finales del proyecto se organizan por flujo. La forma mas rapida de auditarlos es abrir los reportes HTML en el navegador y luego revisar los CSV/JSON de soporte cuando se necesite detalle.

### LLM Qwen3 SFT

```text
outputs/sft_llm_qwen3/qwen3-text-commercial-lora/
```

Contiene el adapter/checkpoint del modelo textual fine-tuned. La evaluacion vive en:

```text
outputs/sft_llm_qwen3/evaluation/
```

Archivos principales:

- `outputs/sft_llm_qwen3/evaluation/qwen3_text_technical_report.html`
  Reporte tecnico HTML del LLM: parseo JSON, cobertura de schema, latencia, fallos comunes y comparacion base vs fine-tuned.
- `outputs/sft_llm_qwen3/evaluation/qwen3_text_base_outputs.json`
  Respuestas generadas por el modelo base para comparacion.
- `outputs/sft_llm_qwen3/evaluation/qwen3_text_finetuned_outputs.json`
  Respuestas generadas por el modelo fine-tuned.
- `outputs/sft_llm_qwen3/evaluation/qwen3_text_llm_metrics.csv`
  Metrica por ejemplo evaluado.
- `outputs/sft_llm_qwen3/evaluation/qwen3_text_comparison_table.csv`
  Tabla comparativa base vs fine-tuned.

### Stable Diffusion SDXL LoRA

```text
outputs/stable_diffusion_lora/
```

Contiene el run visual caption-aware de DreamBooth LoRA sobre SDXL. Las subcarpetas mas relevantes son:

```text
outputs/stable_diffusion_lora/lora_adapter/
outputs/stable_diffusion_lora/evaluation/
outputs/stable_diffusion_lora/generated/
```

Archivos relevantes:

- `outputs/stable_diffusion_lora/evaluation/stable_diffusion_lora_technical_report.html`
  Reporte tecnico HTML del LoRA visual: configuracion, captions usadas, metricas CLIP, galeria base vs LoRA, artefactos y notas operativas.
- `outputs/stable_diffusion_lora/lora_adapter/pytorch_lora_weights.safetensors`
  Adapter LoRA final para cargar sobre `stabilityai/stable-diffusion-xl-base-1.0`.
- `outputs/stable_diffusion_lora/evaluation/training_config.json`
  Configuracion del run y evidencia de que se uso `training_caption_sidecar_txt`.
- `outputs/stable_diffusion_lora/evaluation/caption_training_manifest.csv`
  Trazabilidad entre imagen fuente, PNG preparado y caption sidecar.
- `outputs/stable_diffusion_lora/evaluation/base_vs_lora_metrics.csv`
  Metricas por imagen generada: latencia, CLIP text-image y similitud contra training set.
- `outputs/stable_diffusion_lora/generated/base/`
  Imagenes generadas con SDXL base.
- `outputs/stable_diffusion_lora/generated/lora/`
  Imagenes generadas con SDXL + LoRA.
- `outputs/stable_diffusion_lora/generated/comparison_canvases/`
  Canvases lado a lado usando el mismo prompt y seed.
- `outputs/stable_diffusion_lora/logs/`
  Logs stdout/stderr del entrenamiento.

Subcarpetas como `prepared_dataset/`, `training_output/` y `checkpoints/` son utiles para auditoria o reanudacion, pero pueden ser pesadas. Para presentar resultados, normalmente basta con `lora_adapter/`, `evaluation/` y `generated/`.

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
