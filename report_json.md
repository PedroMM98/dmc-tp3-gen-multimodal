Entendido. Voy directo al análisis del JSON y los ejemplos corregidos.

---

## Diagnóstico del JSON actual

**Lo que está bien:**

- Estructura base `instruction / input / output` correcta para SFT
- 200 ejemplos (cumple el mínimo del README)
- Los 6 campos del output existen en todos los ejemplos
- Diversidad de ciudades, audiencias y objetivos es razonable

**Problemas encontrados:**

**1. Vehículos genéricos, no modelos Toyota reales**

El dataset usa nombres como "SUV híbrida", "Van comercial", "Pickup 4x4". Para acotarlo a Toyota, cada ejemplo debe referenciar un modelo concreto con año: `Toyota Corolla Cross Hybrid 2025`, `Toyota Hilux GR-S 2025`, `Toyota RAV4 2026`, etc.

Los modelos actualmente disponibles en Toyota Perú incluyen RAV4, Corolla, Corolla Cross, Fortuner, Hilux, Land Cruiser Prado, Yaris Cross, Rush, Raize, Avanza, Hiace y 4Runner, lo que da una base real y variada para poblar el campo `Vehiculo`.

**2. `channel_plan` tiene solo 8 valores únicos en 200 ejemplos**

Hay exactamente 8 strings copiados y pegados, uno por canal. No hay variación por objetivo, duración, audiencia ni ciudad. El modelo aprenderá a ignorar el input y copiar la plantilla.

**3. `business_note` tiene solo 16 variantes**

Sigue el mismo patrón: `"Priorizar [X] y validar desempeño por segmento antes de escalar inversión en [canal]"`. No hay nota de negocio real: sin referencia a competidores, sin umbrales de CPL, sin contexto de mercado local.

**4. `image_prompt` no considera el formato del canal**

El prompt siempre termina igual: `"premium dealership aesthetic, clear space for headline"`. No incluye dimensiones, ratio ni instrucciones visuales específicas del canal. Un banner de Instagram (1080x1080) necesita un prompt diferente a un thumbnail de YouTube (1280x720) o un banner físico para impresión (3×1m a 300dpi).

**5. El campo `image_prompt` está cortado**

Varios prompts terminan a mitad de palabra porque el generador truncó la cadena. Ejemplo: `"highlighting ahorro de combus"`.

**6. No existe un campo `recommended_channel`**

El canal histórico viene en el input como dato pasado, pero el output nunca recomienda un canal óptimo. El modelo no aprende a razonar sobre qué canal es mejor según el objetivo; simplemente usa el canal del input como dado.

**7. Solo idioma español, sin inglés**

Todos los 200 ejemplos son `Idioma: Spanish`. El README muestra el ejemplo en inglés, lo que sugiere que la diversidad de idioma era intencionada.

---

## Estructura de output propuesta

Estos son los cambios concretos al schema:

```json
{
  "instruction": "Actua como estratega publicitario para una concesionaria Toyota. Genera una propuesta de campaña en JSON con canal recomendado, imagen adaptada a ese canal y KPIs alineados al objetivo.",

  "input": "Objetivo: Lead Generation | Vehiculo: Toyota Corolla Cross Hybrid 2025 | Rango: mid-range ($26,140) | Audiencia: Familias urbanas 32-45 | Sector cliente: familias con hijos en etapa escolar | Canal historico: Instagram | Ciudad: Lima | Idioma: Spanish | Duracion: 30 Days | Promocion: test drive + cuota inicial reducida | ROI: 2.10 | Conversion rate: 0.08 | Engagement: 9",

  "output": {
    "strategy": "Posicionar el Corolla Cross Hybrid como la elección inteligente para familias limeñas que priorizan eficiencia y seguridad. Capitalizar la tecnología híbrida auto-recargable Toyota para reducir el costo de operación mensual, un argumento concreto para familias con múltiples compromisos urbanos. Cerrar con invitación directa al test drive usando la cuota inicial reducida como detonador de acción.",

    "recommended_channel": "Instagram + Meta Ads",

    "channel_rationale": "Instagram lidera el consumo visual de familias urbanas peruanas de 30-45 años. Reels cortos mostrando el interior familiar y la pantalla táctil generan alta intención de lead. Meta Ads permite segmentar por intereses de familia, educación y movilidad en Lima Metropolitana con formularios de lead nativos.",

    "channel_plan": "Semanas 1-2: Reels de 15s mostrando espacio interior y eficiencia híbrida (awareness). Semanas 3-4: Carrusel comparativo con CTA de cotización y formulario de lead nativo en Instagram. Remarketing en Facebook a quienes interactuaron con el Reel pero no convirtieron.",

    "ad_copy": "¿Listos para el próximo paso familiar? El Toyota Corolla Cross Hybrid 2025 te da tecnología, espacio y eficiencia para cada etapa. Agenda tu test drive y descubre la cuota inicial que se adapta a ti.",

    "image_prompt": "AUTOESPAR Toyota dealership Instagram post 1080x1080px for Toyota Corolla Cross Hybrid 2025, pearl white exterior, Lima Miraflores urban background, family of four walking toward the vehicle, bright morning light, premium automotive commercial photography, Toyota Safety Sense badge visible, no readable text on vehicle body, clear lower third space for headline overlay",

    "kpis": ["Leads generados", "Costo por Lead (CPL)", "Test Drive Bookings", "Tasa de conversión Lead→Test Drive", "ROI de campaña"],

    "business_note": "Monitorear CPL en primeras 72h; si supera S/80 por lead, pausar audiencia fría y reforzar remarketing. El Corolla Cross Hybrid compite con Kia Sportage Hybrid y Hyundai Tucson Hybrid en Lima: enfatizar la garantía Toyota 3 años / 100,000 km como diferenciador clave."
  }
}
```

---

## Tabla de cambios: original vs propuesto

| Campo | Dataset actual | Propuesto |
|---|---|---|
| `Vehiculo` (input) | Genérico: "SUV híbrida" | Modelo real: "Toyota Corolla Cross Hybrid 2025" |
| `recommended_channel` | No existe | Nuevo campo: canal óptimo razonado |
| `channel_rationale` | No existe | Nuevo campo: por qué ese canal para ese input |
| `channel_plan` | 8 strings idénticos rotados | Específico por semana/etapa según duración del input |
| `image_prompt` | Sin dimensiones, truncado | Incluye canal + dimensiones (ej. 1080x1080, 1200x628, 1280x720) |
| `business_note` | 16 plantillas genéricas | Competidores reales, umbrales de KPI, contexto de mercado local |
| `image_prompt` físico | No existe | Para OOH/print: "3000x1000px 300dpi, bleed area, impresión banner..." |

---

## Mapeo de modelos Toyota reales por categoría

Para reemplazar los vehículos genéricos, este es el mapeo directo basado en el lineup actual:

| Categoría original | Modelo Toyota real |
|---|---|
| SUV híbrida | Toyota Corolla Cross Hybrid 2025 / RAV4 Hybrid 2026 / Yaris Cross Hybrid 2024 |
| SUV premium | Toyota Fortuner 2025 / Land Cruiser Prado 2025 / 4Runner 2025 |
| SUV de 3 filas | Toyota Fortuner 7 asientos 2025 |
| Pickup 4x4 | Toyota Hilux GR-S 2025 / Hilux SRV 4x4 2025 |
| Sedán ejecutivo | Toyota Corolla 2025 / Camry Hybrid |
| Sedán híbrido | Toyota Corolla Hybrid 2025 |
| Crossover compacta | Toyota Yaris Cross 2024 / Raize 2024 |
| Hatchback urbano | Toyota Yaris Hatchback 2024 / Agya 2024 |
| Van comercial | Toyota Hiace 2025 / Avanza 2024 |
| SUV eléctrica | Toyota bZ4X (mercados con disponibilidad) |

---

## Sobre el `image_prompt` por canal

Este es el cambio más importante que resuelve tu requerimiento. El prompt de imagen debe variar según el canal recomendado en el output, no según el canal histórico del input:

**Instagram (post cuadrado):** `...1080x1080px, square format, strong central composition, no readable text on vehicle, clear lower third space for headline overlay`

**Instagram/TikTok (Story/Reel vertical):** `...1080x1920px vertical format, 9:16 ratio, subject centered in top 60%, safe zone bottom 200px for CTA`

**Facebook (banner horizontal):** `...1200x628px horizontal, vehicle on left third, negative space on right for text overlay, no readable text`

**YouTube (thumbnail):** `...1280x720px, 16:9 ratio, high contrast, clear focal point visible at small size, top-left space for title text`

**Email (header banner):** `...600x200px horizontal banner, clean white or light background, vehicle centered, professional dealership aesthetic, high resolution for retina`

**Banner físico / highway print:** `...3000x1000px 300dpi CMYK, horizontal billboard format, bleed area 5mm, vehicle on left two-thirds, right third clear for headline and dealership logo, no small text elements`

**WhatsApp (mensaje visual):** `...800x800px square, clean background, vehicle centered, high contrast, readable thumbnail at 200px preview size`