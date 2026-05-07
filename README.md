# Bird Classification From `.pt` Data

This project has two main ways to run bird predictions:

- **Base classifiers**: `efficientnet_model.py`, `resnet_model.py`, `vgg_model.py`
- **YOLO**: `yolo_model.py` and `yolo_detection_classifier.py`

---

## Install

```powershell
pip install -r requirements.txt
```

---

## Prepare data

Expected dataset layout:

```text
<dataset root>/
  Train/
    <bird_name>/
      *.jpg
  Test/
    <bird_name>/
      *.jpg
```

Put the dataset root path into `dataPath.txt`, then run:

```powershell
python .\BirdTrain.py
```

This creates class tensors in `pt_data/`.

---

## Train models

### EfficientNet

```powershell
python .\efficientnet_model.py train --unfreeze-layers 5 --epochs 15 --batch-size 128 --learning-rate 5e-3
```

### ResNet

```powershell
python .\resnet_model.py train --unfreeze-layers 10 --epochs 10 --batch-size 32 --learning-rate 5e-4 --dropout .4 --weight-decay 1e-3
```

### VGG

```powershell
python .\vgg_model.py train --unfreeze-layers 2 --epochs 10 --batch-size 2
```

### YOLO classification

```powershell
python .\yolo_model.py train --image-size 224 --epochs 10 --batch-size 8
```

Training output includes:

- `train_acc`
- `val_acc`
- `delta_val_acc`
- epoch timing
- best-checkpoint weighted / macro precision, recall, and F1

---

## Run the base models

Just pass a picture path.

### EfficientNet

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

Each script prints:

- predicted bird name
- confidence
- checkpoint used

By default, inference uses the best available checkpoint for that model if it exists.

---

## Run YOLO

## 1) Direct YOLO classification

For a single image:

```powershell
python .\yolo_model.py "C:\path\to\bird.jpg"
```

This predicts the bird species directly with the YOLO classification model.

## 2) YOLO detection + base classifier

Script: `yolo_detection_classifier.py`

This uses YOLO to find bird boxes, then sends the crop to the best EfficientNet checkpoint by default.

### If source is an image

```powershell
python .\yolo_detection_classifier.py "C:\path\to\bird.jpg"
```

Optional annotated output image:

```powershell
python .\yolo_detection_classifier.py "C:\path\to\bird.jpg" --output-path ".\runs\annotated_bird.jpg"
```

### If source is a video

```powershell
python .\yolo_detection_classifier.py "C:\path\to\birds.mov"
```

Video mode is simple:

- it scans the full video,
- finds the **single best bird box** from YOLO,
- classifies that one crop once,
- prints the final guess.

Optional annotated output of the best frame:

```powershell
python .\yolo_detection_classifier.py "C:\path\to\birds.mov" --output-path ".\runs\best_frame.jpg"
```

If you want to override the default checkpoint:

```powershell
python .\yolo_detection_classifier.py "C:\path\to\birds.mov" --checkpoint-path ".\model_artifacts\best_resnet_bird_classifier_u20_ep10_bs32_img224.pt"
```

---

## Files to keep

Keep:

- `model_artifacts/`
- `pt_data/`
- `yolo26n.pt`
- `yolov8n-cls.pt`

Safe to delete:

- `__pycache__/`
- `runs/`
- `yolo_cls_data/`

