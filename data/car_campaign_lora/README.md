# Automotive LoRA Training Dataset

Place the car campaign training images in `data/car_campaign_lora/images/`.

Use `metadata.csv` as the source of truth for image captions:

```csv
file_path,caption
./images/real_car_model_01.png,"REALCARMODEL real car model, front three quarter view, metallic blue paint, studio automotive photography, premium lighting"
```

Each caption should include the unique trigger word, car model/series, angle, color, setting, and photo style.

Why metadata instead of one `.txt` file per image:

- It is easier to audit image/caption pairs in one place.
- It is easier to edit captions in bulk.
- It makes route validation straightforward.
- It versions better once the dataset uses real car models.
- It keeps the dataset ready for future caption-aware training scripts.

Recommended for the final demo: 20-40 images covering front, side, rear, interior, detail shots, and lifestyle contexts.

Captions should be written in English for the current project scope. Replace `REALCARMODEL` with the unique trigger word chosen for the real car model once the final image set is ready.

Technical note: the current DreamBooth SDXL command uses one shared `instance_prompt`. The metadata captions are used for dataset QA, traceability, documentation, and prompt-building.
