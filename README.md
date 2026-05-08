# Bird Is Da Word — Demo Guide

A multi-model bird species classifier that can identify **200 bird species** from a single image or video.
Pre-trained checkpoints are included in `model_artifacts/` so you can run predictions right away — no training required.

---

## 1. Install

```powershell
pip install -r requirements.txt
```

---

## 2. Run a Prediction

Pick any bird photo and pass it to one of the scripts below.
Each script auto-loads the best available checkpoint from `model_artifacts/`.

For all base classifier scripts, prediction supports both forms:

```powershell
# Default (auto-picks best checkpoint)
python .\efficientnet_model.py "C:\path\to\bird.jpg"

# Explicit model/checkpoint as first argument
python .\efficientnet_model.py ".\model_artifacts\best_efficientnet_bird_classifier.pt" "C:\path\to\bird.jpg"
```

If you provide two positional arguments, the first is treated as the model/checkpoint path and the second as the image path.

### EfficientNet *(recommended — best accuracy)*

```powershell
python .\efficientnet_model.py "C:\path\to\bird.jpg"
```

### ResNet

```powershell
python .\resnet_model.py "C:\path\to\bird.jpg"
```

### VGG

```powershell
python .\vgg_model.py "C:\path\to\bird.jpg"
```

### YOLO classifier

```powershell
python .\yolo_model.py "C:\path\to\bird.jpg"
```

Every script prints the same three things:

```
Predicted bird: American_Goldfinch
Confidence:     0.9732
Checkpoint:     model_artifacts\best_efficientnet_bird_classifier.pt
```

---

## 3. YOLO Detection + Classifier Pipeline

`yolo_detection_classifier.py` first runs **object detection** to find the bird in the frame,
crops the bounding box, and then hands the crop to EfficientNet for species classification.
This is useful when the bird is not centered or takes up only a small part of the frame.

### Image

```powershell
python .\yolo_detection_classifier.py "C:\path\to\bird.jpg"
```

Save an annotated copy of the result:

```powershell
python .\yolo_detection_classifier.py "C:\path\to\bird.jpg" --output-path ".\runs\annotated_bird.jpg"
```

### Video

```powershell
python .\yolo_detection_classifier.py "C:\path\to\birds.mov"
```

Video mode scans every frame, picks the single highest-confidence bird box,
classifies that crop, and prints the final species guess.

### Detector-only mode (bounding boxes, no classification)

```powershell
python .\yolo_detection_classifier.py "C:\path\to\bird.jpg" --detector-only
```

---

## 4. Use a Specific Checkpoint

All scripts accept an explicit checkpoint as the first argument:

```powershell
python .\efficientnet_model.py "C:\path\to\checkpoint.pt" "C:\path\to\bird.jpg"
python .\yolo_detection_classifier.py "C:\path\to\birds.mov" --checkpoint-path ".\model_artifacts\best_resnet_bird_classifier_u15_ep10_bs64_img224.pt"
```

---

## Files You Need

| Path | Purpose |
|------|---------|
| `model_artifacts/` | Trained checkpoints (`.pt`) and config JSONs |
| `yolo26n.pt` | YOLO detection backbone |
| `yolov8n-cls.pt` | YOLO classification backbone |
| `pt_data/` | Pre-processed tensor files (only needed to retrain) |

Safe to delete: `__pycache__/`, `runs/`, `yolo_cls_data/`

---

---

## Architecture & Model Details

### Dataset

- **200 bird species**, split into `Train/` and `Test/` folders per class.
- Images are pre-processed once by `BirdTrain.py` into per-class `.pt` tensor files stored in `pt_data/`.
  `LazyPtDataset` loads one class file at a time during training, keeping RAM usage flat regardless of dataset size.

---

### EfficientNet (`efficientnet_model.py`)

| Detail | Value |
|--------|-------|
| Backbone | EfficientNet-B2 (default) — also supports B0, B1, B3 |
| Pretrained on | ImageNet-1K |
| Parameters | ~9.1 M (B2); B0 ≈ 5.3 M → B3 ≈ 12.2 M |
| Head | `Dropout → Linear(num_classes)` |
| Fine-tuning strategy | Freeze all backbone layers, then selectively unfreeze the last N feature blocks (`--unfreeze-layers`) |
| Optimizer | AdamW with differential LR (backbone gets `lr × backbone_lr_multiplier`) |
| LR schedule | Cosine annealing (default), step decay, or none |
| Regularisation | Label smoothing (0.1), weight decay, dropout |
| Augmentation | Random horizontal/vertical flip, rotation ±15°, random erasing |
| Adaptive LR | Optional — halves all LR groups whenever `val_acc` drops vs the previous epoch |

---

### ResNet (`resnet_model.py`)

| Detail | Value |
|--------|-------|
| Backbone | ResNet-50, pretrained on ImageNet-1K |
| Fine-tuning strategy | Layer-group unfreezing — `--unfreeze-layers 1` exposes `layer4 + fc`; each additional level adds the next residual group going backwards |
| Head | `Dropout → Linear(num_classes)` replacing the original FC layer |
| Optimizer | AdamW with differential LR |
| Other | Same label smoothing, cosine scheduler, and adaptive LR options as EfficientNet |

---

### VGG (`vgg_model.py`)

| Detail | Value |
|--------|-------|
| Backbone | VGG-16, pretrained on ImageNet-1K |
| Fine-tuning strategy | Block-level unfreezing from the end of the feature extractor |
| Head | Replaces VGG's original three-layer classifier with `Dropout → Linear(num_classes)` |
| Note | Larger memory footprint than EfficientNet/ResNet; smaller batch sizes recommended |

---

### YOLO Classification (`yolo_model.py`)

| Detail | Value |
|--------|-------|
| Base | YOLOv8n-cls (`yolov8n-cls.pt`) via Ultralytics |
| Task | End-to-end image classification (no detection step) |
| Data prep | Images organised into `yolo_cls_data/train/<class>/` and `yolo_cls_data/test/<class>/` |
| Training | Delegates to the Ultralytics `model.train()` API; accuracy history is parsed from Ultralytics CSV results |

---

### YOLO Detection + Classifier Pipeline (`yolo_detection_classifier.py`)

A two-stage pipeline:

1. **Detection** — `yolo26n.pt` runs on the full image/frame and returns bounding boxes for all birds found.
2. **Classification** — each bird crop is passed to the best available base classifier (EfficientNet by default) for species identification.

In **video mode** the detector scans every frame, selects the single highest-confidence bird box across all frames, and classifies that one crop.

---

### Training a Model from Scratch

If you want to retrain, first put the dataset root path in `dataPath.txt`, then:

```powershell
# Build .pt tensor files
python .\BirdTrain.py

# Train EfficientNet (example hyper-parameters)
python .\efficientnet_model.py train --unfreeze-layers 5 --epochs 15 --batch-size 128 --learning-rate 5e-3

# Train ResNet
python .\resnet_model.py train --unfreeze-layers 10 --epochs 10 --batch-size 32 --learning-rate 5e-4 --dropout .4 --weight-decay 1e-3

# Train VGG
python .\vgg_model.py train --unfreeze-layers 2 --epochs 10 --batch-size 2

# Train YOLO classifier
python .\yolo_model.py train --image-size 224 --epochs 10 --batch-size 8
```

Each run saves:
- A `best_<model>_bird_classifier.pt` checkpoint (best validation accuracy)
- A final-epoch `.pt` checkpoint
- A `<model>_config.json` with all hyperparameters
- A `<model>_error_vs_epochs.png` training/validation error curve
