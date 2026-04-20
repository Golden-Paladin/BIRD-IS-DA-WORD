# Bird Classification From `.pt` Data

A modular PyTorch bird-species classifier supporting **EfficientNet**, **ResNet-50**, **VGG-16**, and **YOLOv8n-cls** as backbones.  
Training streams one class file at a time so you never load the entire dataset into RAM simultaneously.

---

## How it works

1. `BirdTrain.py` reads your image dataset and converts each class to a separate `.pt` tensor file in `pt_data/`.
2. Any model script reads those files through `LazyPtDataset` — **only one class is in RAM at a time**, so memory stays flat even for 200 classes.
3. Each training epoch now reports:
   - **training accuracy** — how well the model fits the training set
   - **validation accuracy** — generalisation on the held-out test split
   - **delta_val_acc** — change vs. the previous epoch (positive = improving, negative = regressing)
   - **per-epoch wall-clock time** split into train seconds / val seconds / total seconds
4. The best-epoch checkpoint is saved automatically; the final-epoch checkpoint is saved separately.

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
| `efficientnet_model.py` | EfficientNet-B0/B1/**B2**/B3 | 5–12 M | **Best accuracy/RAM ratio — try this first** |
| `resnet_model.py` | ResNet-50 | 23.9 M | Solid, well-understood baseline |
| `vgg_model.py` | VGG-16 | 135 M | Heavy — keep batch_size ≤ 4 |
| `yolo_model.py` | YOLOv8n-cls | ~3 M | Fastest; exports images to disk |

---

## Common training options (all PyTorch scripts share these)

| Flag | Default | Meaning |
|------|---------|---------|
| `--image-size` | `224` | Resize all images to N × N pixels |
| `--batch-size` | `4` | Samples per gradient step — lower first if RAM is tight |
| `--epochs` | `10` | Total training epochs |
| `--learning-rate` | `5e-4` | Head / classifier learning rate |
| `--unfreeze-layers` | `1` (ResNet/VGG) or `2` (EfficientNet) | How many backbone blocks to fine-tune |
| `--backbone-lr-multiplier` | `0.1` | Backbone LR = head LR × this (prevents catastrophic forgetting) |
| `--lr-scheduler` | `cosine` | `cosine` / `step` / `none` — see details below |
| `--label-smoothing` | `0.1` | Softens targets slightly — helps with 200 nearly-identical classes |
| `--weight-decay` | `1e-4` | AdamW L2 regularisation |
| `--dropout` | `0.3` (ResNet/EfficientNet), `0.5` (VGG) | Dropout before final layer |
| `--augment` / `--no-augment` | on | Random flip, rotate, erase each epoch |
| `--max-files` | off | Load only N class files — useful for quick smoke tests |
| `--adaptive-lr` / `--no-adaptive-lr` | **off** | Reduce LR automatically when val_acc drops — see below |
| `--adaptive-lr-factor` | `0.5` | How much to scale LRs when adaptive-lr fires (0.5 = halve) |

---

## Adaptive Learning Rate (`--adaptive-lr`)

> **Default: disabled.** By default every script uses only the static `--lr-scheduler`.

When `--adaptive-lr` is enabled the training loop watches `delta_val_acc` after every epoch.  
If validation accuracy **decreases** compared to the previous epoch, the learning rate of **every parameter group** is immediately multiplied by `--adaptive-lr-factor` (default `0.5`, i.e. halved).

**Why this helps:**
- Prevents the model from overshooting a good valley in loss space.
- Allows you to start with an aggressive LR (fast early learning) and automatically slow down when progress stalls.
- Acts as a safety net when you are unsure of the optimal LR.

**Example — start at 5e-3 and adapt:**

```powershell
python .\efficientnet_model.py train `
  --unfreeze-layers 5 --epochs 20 --batch-size 128 `
  --learning-rate 5e-3 --adaptive-lr --adaptive-lr-factor 0.5
```

When a bad epoch is detected you see an inline message before the epoch summary:

```
  [adaptive LR] val_acc dropped → LR scaled ×0.5: ['2.50e-04', '5.00e-03']
Epoch 8/20 - loss: 1.1052 - train_acc: 0.8812 - val_acc: 0.7987 - delta_val_acc: -0.0012 [best: 0.7999] - time: train=91.4s val=4.2s total=95.7s
```

**Interaction with `--lr-scheduler`:**  
Both can be active simultaneously. The scheduler step and the adaptive scaling are independent — the scheduler ticks every epoch while the adaptive scale fires only on a negative delta. For the most reactive setup use `--lr-scheduler none --adaptive-lr`.

---

## LR Scheduler details

| Value | Behaviour |
|-------|-----------|
| `cosine` | `CosineAnnealingLR` — smooth decay from initial LR to ~0 over all epochs. Best general choice. |
| `step` | `StepLR` — halves the LR every `epochs // 3` epochs. More abrupt; good for short runs. |
| `none` | Fixed LR for the entire run. Combine with `--adaptive-lr` for fully reactive scheduling. |

---

## Epoch output explained

Every epoch prints one line in this format:

```
Epoch 7/15 - loss: 1.1273 - train_acc: 0.8941 - val_acc: 0.7936 - delta_val_acc: +0.0049 [best: 0.7936] - time: train=92.3s val=4.1s total=96.5s
```

| Field | Meaning |
|-------|---------|
| `loss` | Average cross-entropy training loss for this epoch |
| `train_acc` | Fraction of **training** samples predicted correctly this epoch |
| `val_acc` | Fraction of **test** samples predicted correctly — the primary quality metric |
| `delta_val_acc` | val_acc minus last epoch's val_acc (`N/A` on epoch 1). Positive = still improving. |
| `[best: x.xxxx]` | Best val_acc ever seen — the epoch that produced this was saved as `best_*.pt` |
| `time: train=Xs val=Xs total=Xs` | Wall-clock seconds for train pass, val pass, and full epoch |

> **Tip:** A large **train_acc − val_acc gap** (e.g. 0.99 train vs 0.70 val) is a sign of overfitting. Try adding dropout, augmentation, fewer unfreeze layers, or `--weight-decay`.

---

## Unfreeze layers guide

### ResNet-50

| `--unfreeze-layers` | Trainable blocks | Notes |
|---------------------|-----------------|-------|
| `0` | Classifier head only | Fastest, least accurate — good sanity check |
| `1` | Head + `layer4` | **Default** |
| `2` | + `layer3` | **Recommended — good accuracy without excessive RAM** |
| `3` | + `layer2` | More capacity, slower |
| `4` | Full backbone | Most RAM / most accurate |

### EfficientNet-B2 (9 feature blocks total)

| `--unfreeze-layers` | Notes |
|---------------------|-------|
| `1` | Classifier + last conv head block |
| `2` | + last MBConv stage — **default** |
| `5` | **Best from experiments** — strong accuracy at moderate RAM |
| `9+` | Full backbone fine-tuning |

### VGG-16 (5 conv blocks)

| `--unfreeze-layers` | Notes |
|---------------------|-------|
| `0` | Frozen features — head only |
| `1` | Last conv block — **default** |
| `2` | Last two conv blocks — recommended |
| `5` | Full backbone |

---

## Per-model quick-start

### 1) EfficientNet — recommended starting point

EfficientNet-B2 has **3× fewer parameters than ResNet-50** and typically achieves higher accuracy on fine-grained classification.

```powershell
# Default run (B2, 224 px, batch 4, 10 epochs)
python .\efficientnet_model.py

# Best accuracy run from recorded experiments (~79% val_acc at epoch 9)
python .\efficientnet_model.py train --unfreeze-layers 5 --epochs 15 --batch-size 128 --learning-rate 5e-3

# Adaptive LR — start aggressive, reduce automatically when progress stalls
python .\efficientnet_model.py train --unfreeze-layers 5 --epochs 20 --batch-size 128 --learning-rate 8e-3 --adaptive-lr

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

# Good accuracy run from experiments (~79% val_acc at epoch 8)
python .\resnet_model.py train --unfreeze-layers 10 --epochs 10 --batch-size 32 --learning-rate 5e-4 --dropout .4 --weight-decay 1e-3

# Adaptive LR — no static schedule, fully reactive
python .\resnet_model.py train --unfreeze-layers 2 --epochs 15 --batch-size 32 --learning-rate 1e-3 --adaptive-lr --lr-scheduler none
```

Predict:
```powershell
python .\resnet_model.py predict --image-path "C:\path\to\bird.jpg"
```

---

### 3) VGG-16

> VGG is large (135 M params). Keep `--batch-size 2` or `4` to stay inside 32 GB.

```powershell
# Default run
python .\vgg_model.py

# Unfreeze last two conv blocks
python .\vgg_model.py train --unfreeze-layers 2 --epochs 10 --batch-size 2

# With adaptive LR
python .\vgg_model.py train --unfreeze-layers 2 --epochs 15 --batch-size 4 --learning-rate 5e-4 --adaptive-lr
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
| `resnet_bird_classifier_u2_ep15_bs8_img224.pt` | Final-epoch checkpoint (`u`=unfreeze layers) |
| `best_resnet_bird_classifier_u2_ep15_bs8_img224.pt` | **Best val-accuracy epoch — use this for prediction** |
| `resnet_config.json` | Full config snapshot for reproducibility |
| (same pattern for vgg, efficientnet) | |

Each `.pt` checkpoint stores:
- `model_state_dict` — all model weights
- `classes` — ordered list of bird species names (output index → species name)
- `config` — full training hyper-parameters used to produce this checkpoint
- `model_name` — architecture identifier (e.g. `efficientnet_b2`, `resnet50`, `vgg16`)

### Use a trained checkpoint for prediction

```powershell
python .\resnet_model.py predict `
  --checkpoint-path ".\model_artifacts\best_resnet_bird_classifier_u2_ep15_bs8_img224.pt" `
  --image-path "C:\path\to\bird.jpg"

python .\efficientnet_model.py predict `
  --checkpoint-path ".\model_artifacts\best_efficientnet_bird_classifier.pt" `
  --image-path "C:\path\to\bird.jpg"
```

Expected output:
```
Predicted bird: Baltimore_Oriole
Confidence: 0.9341
```

---

## Memory tips

- Training streams one `.pt` file at a time — the whole dataset is never concatenated into RAM.
- Lower `--batch-size` first if you hit OOM errors (`1` or `2` usually fits any setup).
- EfficientNet-B0 (`--model-variant b0`) is the lightest backbone option.
- `--unfreeze-layers 0` trains only the new head — very low memory, surprisingly decent accuracy for a first run.
- `224×224` default is safe; higher `--image-size` multiplies RAM/VRAM usage sharply.

Memory impact ranking (highest first):
1. `--image-size` — quadratic impact on activation maps
2. `--batch-size`
3. Model family (VGG >> ResNet > EfficientNet B3 > B2 > B0)
4. `--unfreeze-layers` (more frozen = less gradient memory)
5. DataLoader `num_workers` if added later

---

## FAQ

**Q: What is `model_artifacts/`?**  
A: Your trained-model output folder: checkpoint `.pt` files and config JSON files.

**Q: Can I really use a `.pt` file directly for prediction?**  
A: Yes. Run the `predict` sub-command with `--checkpoint-path` and `--image-path`. The script rebuilds the network from the stored config and loads the weights.

**Q: Which checkpoint should I use?**  
A: Always use the `best_*.pt` checkpoint — it was saved from the epoch with the highest validation accuracy, not necessarily the last epoch.

**Q: What does a negative `delta_val_acc` mean?**  
A: The model's validation accuracy dropped compared to the previous epoch. Occasional dips are normal (stochastic noise), but consistent negative deltas suggest the LR is too high or the model is overfitting. Use `--adaptive-lr` to automatically reduce the LR when this happens.

**Q: Should I use `--adaptive-lr` together with `--lr-scheduler cosine`?**  
A: You can. They act independently — the cosine schedule decays LR smoothly in the background while the adaptive step fires an extra reduction any time val_acc drops. For the most reactive setup use `--lr-scheduler none --adaptive-lr`.

**Q: How do I get faster training without crashing memory?**  
A: Start with `--image-size 224 --batch-size 4`, then increase `--batch-size` gradually (GPU utilisation goes up with larger batches). If memory is tight, use EfficientNet-B0 or reduce `--image-size`.

**Q: What about the `torch.load` FutureWarning?**  
A: The loaders explicitly pass `weights_only=False` (with a backward-compatible fallback for older torch versions) to suppress the warning and make behaviour explicit.
