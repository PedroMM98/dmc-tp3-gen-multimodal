# Automotive LoRA Training Dataset

Place the campaign images in:

```text
data/car_campaign_lora/images/
```

Use metadata CSV as the source of truth for image captions:

```csv
file_path,caption,training_caption
./images/1.autoespar_toyota_corolla_cross_2025.jpg,"source/audit description","AUTOESPAR dealership automotive advertisement, Toyota Corolla Cross, clean promotional banner layout, vehicle-focused composition, bold red white and black graphic blocks, blank copy area, small unreadable legal footer bar"
```

## Current Metadata

The current `image_metadata.csv` is the canonical metadata file and has 154 rows:

- 108 Autoespar dealership campaign/banner rows built from the dealership CSVs and image description files.
- 40 Toyota visual reference rows built from generated Toyota captions and image descriptions.
- 6 direct supplemental reference rows from `docs/images_descriptions_start/lora_caption_dataset.csv`.

Use `AUTOESPAR` as the trigger word for the dealership advertising style.

Caption roles:

- `caption` preserves the source/audit description.
- `training_caption` is the cleaned diffusion caption used for LoRA training.
- Autoespar training captions start with `AUTOESPAR dealership automotive advertisement`.
- Toyota reference training captions start with `Toyota automotive advertising reference`.
- Training captions avoid exact prices, phone numbers, dates, warranty/legal wording, and quoted ad text.
- Noisy ad text should be described visually, for example `small unreadable legal footer bar`, `blank copy area`, or `promotional headline blocks`.

The final 6 supplemental rows are clean logo/car reference assets, not Autoespar ad-layout captions.

## Why Metadata CSV

- It is easier to audit image/caption pairs in one place.
- It is easier to edit captions in bulk.
- It makes path validation straightforward.
- It versions better once the dataset uses real campaign images.
- It keeps `caption` available for traceability while `training_caption` stays optimized for diffusion training.

Technical note: the Colab DreamBooth SDXL workflow prepares normalized `.png` images plus same-stem `.txt` sidecars from `training_caption`. The shared `instance_prompt` remains only as a fallback when a sidecar caption is absent.

## Regenerate Metadata

When the starter CSVs or caption sources change, regenerate `image_metadata.csv` with:

```powershell
conda run -n chatbot_llm_rag_taller python scripts/prepare_automotive_lora_metadata.py
```
