# Bird Classification From `.pt` Data

## How it works

1. `BirdTrain.py` converts your image dataset into per-class `.pt` files in `pt_data/`
2. Any of the four model scripts reads those files with **lazy / streaming loading** — only one class at a time is in RAM, so you never blow out 32 GB
3. Training augments, schedules the learning rate, uses differential per-layer learning rates, and saves the best-epoch checkpoint automatically

---

## Install

```powershell
pip install -r requirements.txt
```

---

## Dataset layout expected by BirdTrain.py

```
<dataset root>/
  Train/
    <bird_name>/
      *.jpg
  Test/
    <bird_name>/
      *.jpg
```

Put the path to `<dataset root>` in `dataPath.txt`, then:

```powershell
python .\BirdTrain.py
```

Generated files in `pt_data/`: `Acadian_Flycatcher_Train.pt`, `Acadian_Flycatcher_Test.pt`, … `dataset_metadata.json`

---

## Models

| Model | Architecture | Params | Notes |
|-------|-------------|--------|-------|
| `resnet_model.py` | ResNet-50 | 23.9 M | solid baseline |
| `vgg_model.py` | VGG-16 | 135 M | heavy, try with small batch |
| `efficientnet_model.py` | EfficientNet-B0/B1/**B2**/B3 | 5–12 M | **best accuracy/RAM ratio — try this first** |
| `yolo_model.py` | YOLOv8n-cls | ~3 M | fastest, exports images to disk |

### Common training options (all scripts share them)

| Flag | Default | Meaning |
|------|---------|---------|
| `--image-size` | `224` | Resize to N × N |
| `--batch-size` | `4` | Lower this first if RAM is tight |
| `--epochs` | `10` | Training epochs |
| `--learning-rate` | `5e-4` | Head learning rate |
| `--unfreeze-layers` | `1` (ResNet/VGG) or `2` (EfficientNet) | How many backbone blocks to fine-tune |
| `--backbone-lr-multiplier` | `0.1` | Backbone LR = head LR × this (prevents catastrophic forgetting) |
| `--lr-scheduler` | `cosine` | `cosine` / `step` / `none` |
| `--label-smoothing` | `0.1` | Helps with 200 classes |
| `--weight-decay` | `1e-4` | AdamW regularisation |
| `--dropout` | `0.3` (ResNet/EfficientNet), `0.5` (VGG) | Dropout before final layer |
| `--augment` / `--no-augment` | on | Random flip, rotate, erase applied each epoch |
| `--max-files` | off | Load only N class files — quick smoke test |

> **Tip — `--unfreeze-layers` guide (ResNet-50):**
> - `0` = only the new classifier head trains (fastest, least accurate)
> - `1` = head + `layer4` — **default**
> - `2` = + `layer3` — **recommended for a good accuracy boost**
> - `3` = + `layer2`
> - `4` = full backbone (most RAM, most accurate)

> **Tip — `--unfreeze-layers` guide (EfficientNet-B2 has 9 feature blocks):**
> - `1` = classifier + last conv head block
> - `2` = + last MBConv stage — **default**
> - `3`+ = progressively deeper fine-tuning

---

### 1) EfficientNet — recommended starting point

EfficientNet-B2 has **3× fewer parameters than ResNet-50** and typically achieves **higher accuracy** on fine-grained classification.

```powershell
# One-line default run (B2, 224 px, batch 4, 10 epochs)
python .\efficientnet_model.py

# Unfreeze more blocks and run longer for best accuracy
python .\efficientnet_model.py train --unfreeze-layers 3 --epochs 20 --batch-size 8

# Lighter B0 variant if RAM is tight
python .\efficientnet_model.py train --model-variant b0 --batch-size 8
```

Predict:
```powershell
python .\efficientnet_model.py predict --image-path "C:\path\to\bird.jpg"
```

---

### 2) ResNet-50

```powershell
# Default run (unfreeze layer4, cosine LR, augmentation on)
python .\resnet_model.py

# Unfreeze two blocks for better accuracy (what you want after epoch 4 at 53%)
python .\resnet_model.py train --unfreeze-layers 2 --epochs 15 --batch-size 8

# High-res run (memory-heavy)
python .\resnet_model.py train --image-size 2024 --batch-size 1
```

Predict:
```powershell
python .\resnet_model.py predict --image-path "C:\path\to\bird.jpg"
```

---

### 3) VGG-16

> VGG is large (135 M params). Keep `--batch-size 2` or `--batch-size 4` to stay inside 32 GB.

```powershell
# Default run
python .\vgg_model.py

# Unfreeze last two conv blocks
python .\vgg_model.py train --unfreeze-layers 2 --epochs 10 --batch-size 2
```

Predict:
```powershell
python .\vgg_model.py predict --image-path "C:\path\to\bird.jpg"
```

---

### 4) YOLO (YOLOv8n-cls)

```powershell
# Default run
python .\yolo_model.py

# Custom run
python .\yolo_model.py train --image-size 224 --epochs 10 --batch-size 8
```

Predict:
```powershell
python .\yolo_model.py predict --image-path "C:\path\to\bird.jpg"
```

---

## Checkpoints and config

After training, `model_artifacts/` contains:

| File | Description |
|------|-------------|
| `resnet_bird_classifier_u2_ep15_bs8_img224.pt` | Final epoch checkpoint (`u`=unfreeze layers) |
| `best_resnet_bird_classifier_u2_ep15_bs8_img224.pt` | **Best val-accuracy epoch** — use this for prediction |
| `resnet_config.json` | Full config snapshot |
| (same pattern for vgg, efficientnet, yolo) | |

`model_artifacts/` is the folder that stores your trained model files. A `.pt` file is a PyTorch checkpoint. In this project, it includes:
- model weights (`model_state_dict`)
- class names (`classes`)
- training config (`config`)

So yes, the `.pt` file is the working model output you use for bird prediction.

### Use your trained `.pt` model for bird prediction

```powershell
# Use the best checkpoint from training
python .\resnet_model.py predict --checkpoint-path ".\model_artifacts\best_resnet_bird_classifier_u2_ep15_bs8_img224.pt" --image-path "C:\path\to\bird.jpg"
```

You should see:
- predicted bird class
- confidence score

---

## Memory tips

- Training streams one `.pt` file at a time — the whole dataset is never concatenated into RAM
- Lower `--batch-size` first if you hit OOM errors (`1` or `2` usually fits)
- EfficientNet-B0 (`--model-variant b0`) is the lightest backbone option
- `--unfreeze-layers 0` trains only the new head — very low memory, surprisingly decent accuracy
- Avoid `BirdTrain.py --prepare-training` on large datasets — it loads everything into RAM at once
- `224×224` default is safe; `--image-size 2024` multiplies RAM and VRAM usage sharply

Most important memory knobs (highest impact first):
1. `--image-size`
2. `--batch-size`
3. model family/variant (`resnet` vs `efficientnet b0/b1/b2/...`)
4. `--unfreeze-layers`
5. dataloader worker count (if you later add `num_workers`)

---

## FAQ

**Q: What is `model_artifacts/`?**  
A: It is your trained-model output folder: checkpoints (`.pt`) plus config JSON.

**Q: Can I really use a `.pt` file directly?**  
A: Yes. Run `predict` with `--checkpoint-path` and `--image-path`; the script rebuilds the network and loads the checkpoint weights.

**Q: Which checkpoint should I use?**  
A: Use the `best_*.pt` checkpoint for inference because it is saved from the epoch with best validation accuracy.

**Q: What about the `torch.load` `FutureWarning`?**  
A: The loaders now explicitly set `weights_only=False` (with backward-compatible fallback). This removes the warning and makes behavior explicit for current/future torch versions.

**Q: How do I get faster training without crashing memory?**  
A: Start with `--image-size 224 --batch-size 4`, then increase `batch-size` gradually. If memory is tight, lower `image-size` or use EfficientNet-B0.

