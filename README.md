# Bird Classification From `.pt` Data

This repo now has:

1. `BirdTrain.py` to generate per-bird `.pt` files from your dataset path in `dataPath.txt`
2. Three separate model files for training and prediction from those `.pt` files:
   - `resnet_model.py`
   - `vgg_model.py`
   - `yolo_model.py`

The model scripts now train in a **streaming / lazy-loading** way, so they do **not** concatenate every `.pt` tensor into RAM at once.

Default training settings were also lowered to be safer on a **32 GB RAM** machine:

- ResNet default: **224 x 224**, batch size **4**
- VGG default: **224 x 224**, batch size **4**
- YOLO default: **224 x 224**, batch size **4**

You can still override all of these from CLI whenever you want.

## Dataset and `.pt` generation

Expected source dataset layout:

```text
<dataset root>/
  Train/
    <bird_name>/
      *.jpg
  Test/
    <bird_name>/
      *.jpg
```

Generate `.pt` files:

```powershell
python .\BirdTrain.py
```

Generated output in `pt_data/` includes files like:

- `Acadian_Flycatcher_Train.pt`
- `Acadian_Flycatcher_Test.pt`
- `dataset_metadata.json`

## Install

```powershell
pip install -r requirements.txt
```

## Model Files

Each model file supports:

- `train` command (uses `pt_data/`)
- `predict` command (single image -> bird class)
- config save in `model_artifacts/`

You can now train either of these ways:

```powershell
python .\resnet_model.py
python .\resnet_model.py train
python .\resnet_model.py --epochs 5 --batch-size 2
```

If you omit the subcommand, the script assumes `train`.

### 1) ResNet model (`resnet_model.py`)

Train with defaults:

```powershell
python .\resnet_model.py
```

Train with explicit options:

```powershell
python .\resnet_model.py train --image-size 224 --epochs 3 --batch-size 4
```

Predict one image:

```powershell
python .\resnet_model.py predict --image-path "C:\path\to\bird.jpg" --image-size 224
```

### 2) VGG model (`vgg_model.py`)

Train with defaults:

```powershell
python .\vgg_model.py
```

Train with explicit options:

```powershell
python .\vgg_model.py train --image-size 224 --epochs 3 --batch-size 4
```

Predict one image:

```powershell
python .\vgg_model.py predict --image-path "C:\path\to\bird.jpg" --image-size 224
```

### 3) YOLO model (`yolo_model.py`)

Train with defaults:

```powershell
python .\yolo_model.py
```

Train with explicit options:

```powershell
python .\yolo_model.py train --image-size 224 --epochs 3 --batch-size 4
```

Predict one image:

```powershell
python .\yolo_model.py predict --image-path "C:\path\to\bird.jpg" --image-size 224
```

## Config and artifacts

Training writes into `model_artifacts/`:

- ResNet checkpoint + `resnet_config.json`
- VGG checkpoint + `vgg_config.json`
- YOLO checkpoint + `yolo_config.json`

These config files track key parameters (`image_size`, `epochs`, `batch_size`, etc.) so you can quickly see/reuse settings.

## Notes

- Training now streams from each per-class `.pt` file instead of loading the whole dataset into memory at once.
- `224x224` is now the safe default. You can still try larger values like `--image-size 2024`, but memory usage will go up sharply.
- If you hit memory issues, lower `--batch-size` first.
- For quick smoke tests, use `--max-files` on model training commands.
- `BirdTrain.py --prepare-training` still loads all generated tensors back into memory, so avoid that option unless you are testing with a small subset.
- Final prediction output is the top-1 bird class and confidence.

