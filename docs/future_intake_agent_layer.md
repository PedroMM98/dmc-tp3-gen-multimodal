# Propuesta futura: Intake Agent para normalizar briefs comerciales

## Contexto

El notebook actual trabaja con un contrato de entrada estructurado para el fine-tuning del LLM. Cada ejemplo SFT usa un campo `input` con datos separados por etiquetas:

```text
Objetivo: Financing Promotion | Vehiculo: SUV hibrida | Rango: mid-range | Audiencia: Commuters 27-45 | Sector cliente: usuarios de movilidad diaria | Canal historico: LinkedIn | Ciudad: Los Angeles | Idioma: Spanish | Duracion: 14 Days | Promocion: bono de mantenimiento | ROI: 2.16 | Conversion rate: 0.05 | Engagement: 12
```

Ese formato es util para el fine-tuning porque reduce ambiguedad y permite que el modelo aprenda una relacion estable:

```text
instruction + input estructurado -> output JSON comercial
```

Para el alcance actual, se mantiene este contrato y se asume que el usuario o el notebook construyen el `input` con esa estructura.

## Idea futura

En una version posterior, se podria agregar una capa previa al LLM fine-tuned: un **Intake Agent** o agente de normalizacion de brief.

Su objetivo seria permitir que el usuario escriba un pedido en lenguaje natural y que el sistema lo convierta automaticamente al formato estructurado que espera el LLM fine-tuned.

Flujo propuesto:

```text
Usuario escribe un prompt libre
   ↓
Intake Agent interpreta intencion
   ↓
Extrae campos disponibles
   ↓
Consulta base de datos / CRM / historico de campanas
   ↓
Completa campos faltantes o enriquecibles
   ↓
Si falta informacion critica, repregunta
   ↓
Construye input estructurado
   ↓
LLM fine-tuned genera propuesta comercial JSON
   ↓
Prompt builder + SDXL generan assets visuales
```

## Ejemplo

Prompt libre del usuario:

```text
Quiero una campana para promocionar una SUV hibrida en Miami para familias jovenes, enfocada en test drives.
```

El Intake Agent podria convertirlo en:

```text
Objetivo: Test Drive Bookings | Vehiculo: SUV hibrida | Rango: mid-range | Audiencia: Families 30-44 | Sector cliente: familias urbanas | Canal historico: Instagram | Ciudad: Miami | Idioma: Spanish | Duracion: 30 Days | Promocion: test drive + financiamiento | ROI: 2.10 | Conversion rate: 0.08 | Engagement: 9
```

Campos inferidos desde el prompt:

- `Objetivo`
- `Vehiculo`
- `Audiencia`
- `Ciudad`

Campos enriquecidos desde datos internos:

- `Rango`
- `Sector cliente`
- `Canal historico`
- `Duracion`
- `Promocion`
- `ROI`
- `Conversion rate`
- `Engagement`

## Repreguntas

Si el prompt del usuario no trae informacion suficiente, el agente no deberia inventar los campos criticos. Deberia repreguntar.

Ejemplos:

```text
¿La campana debe priorizar leads, reservas de test drive o promocion de financiamiento?
```

```text
¿A que audiencia quieres orientar la campana: familias, commuters, compradores premium o conductores eco-conscious?
```

```text
¿En que ciudad o mercado se activara la campana?
```

## Campos criticos minimos

Para pasar al LLM fine-tuned sin repreguntar, el agente deberia tener suficiente informacion sobre:

- `Objetivo`
- `Vehiculo`
- `Audiencia`
- `Ciudad` o mercado
- `Idioma`
- `Promocion` u oferta

Campos como `ROI`, `Conversion rate`, `Engagement`, `Canal historico` o `Duracion` podrian completarse desde datos historicos si existen.

## Arquitectura futura

La arquitectura podria extenderse asi:

```text
1. Intake Agent
   Entiende lenguaje natural, extrae campos y detecta faltantes.

2. Data Enrichment Layer
   Consulta CRM, historico de campanas, catalogo de vehiculos, promociones y metricas.

3. Structured Brief Builder
   Convierte los datos normalizados al formato usado en el SFT.

4. Fine-tuned Campaign Generator
   Genera strategy, channel_plan, ad_copy, image_prompt, kpis y business_note.

5. Visual Prompt Builder + SDXL LoRA
   Produce assets visuales por placement.
```

## Regla de decision

El agente podria operar con una regla simple:

```text
Si campos criticos completos y confianza suficiente:
    construir input estructurado
    llamar al LLM fine-tuned
Si faltan campos criticos:
    repreguntar al usuario
Si faltan campos enriquecibles:
    consultar base de datos o aplicar defaults documentados
```

## Valor de esta capa

Esta capa haria que el sistema sea mas natural para usuarios finales, porque no tendrian que conocer el formato exacto del `input` usado en el fine-tuning.

Tambien reduce errores, porque separa responsabilidades:

- El usuario expresa una necesidad en lenguaje normal.
- El Intake Agent estructura y completa el brief.
- El LLM fine-tuned genera la propuesta comercial.
- El pipeline visual genera los assets.

## Nota de alcance

Esta capa queda fuera del alcance actual del notebook. Por ahora, el proyecto seguira trabajando directamente con inputs estructurados, consistentes con el dataset SFT.

La propuesta se conserva como mejora futura para una version mas cercana a producto.
