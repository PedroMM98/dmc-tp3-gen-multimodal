# Automotive LoRA Training Dataset

Place the campaign images in:

```text
data/car_campaign_lora/images/
```

Use metadata CSV as the source of truth for image captions:

```csv
file_path,caption
./images/1.autoespar_toyota_corolla_cross_2025.jpg,"AUTOESPAR automotive dealership marketing banner for Toyota Corolla Cross Hybrid Electric 2025. ... includes the conditions/legal disclaimer area at the bottom of the image, usually inside a black, white, or red horizontal rectangle with very small Spanish legal text."
```

## Current Metadata

The current `metadata_template.csv` has 29 rows:

- 23 Autoespar dealership campaign/banner rows built from `docs/images_descriptions_start/datos-concesionario - Hoja 1.csv` and `docs/images_descriptions_start/image_descriptions.csv`.
- 6 direct supplemental reference rows from `docs/images_descriptions_start/lora_caption_dataset.csv`.

Use `AUTOESPAR` as the trigger word for the dealership advertising style.

The first 23 captions describe:

- Autoespar dealership campaign/banner style.
- Toyota model, model year, engine/powertrain, transmission, campaign, warranty, and dealer details when available.
- The `condiciones`/legal disclaimer area at the bottom of the image, usually inside a black, white, or red horizontal rectangle with very small Spanish legal text.
- Quote or service phone numbers when present, such as wording after `cotiza al` or `agenda tu servicio al`.

The final 6 supplemental captions are kept exactly as written in `lora_caption_dataset.csv`; they are clean logo/car reference assets, not Autoespar ad-layout captions.

## Why Metadata CSV

- It is easier to audit image/caption pairs in one place.
- It is easier to edit captions in bulk.
- It makes path validation straightforward.
- It versions better once the dataset uses real campaign images.
- It keeps the dataset ready for future caption-aware training scripts.

Technical note: the current DreamBooth SDXL command uses one shared `instance_prompt`. The metadata captions are used for dataset QA, traceability, documentation, and prompt-building.

## Regenerate Metadata

When the starter CSVs in `docs/images_descriptions_start/` change, regenerate `metadata_template.csv` with:

```powershell
conda run -n chatbot_llm_rag_taller python scripts/prepare_automotive_lora_metadata.py
```
