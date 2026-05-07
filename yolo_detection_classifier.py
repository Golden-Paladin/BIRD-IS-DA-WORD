from __future__ import annotations

"""YOLO detection -> base classifier handoff for still images.

This script is intentionally image-only:
1. YOLO finds bird bounding boxes.
2. Each box is cropped (with optional padding).
3. The crop is passed to a base classifier checkpoint (EfficientNet by default).
4. The script prints bird labels and can save an annotated image.
"""

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from bird_classifier_inference import load_bird_classifier, predict_bird_from_pil

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

@dataclass(frozen=True)
class DetectionClassifierConfig:
    """Shared configuration for one detection/classification image run."""

    detector_path: str = "yolo26n.pt"
    classifier_checkpoint: str = str(Path("model_artifacts") / "best_efficientnet_bird_classifier.pt")
    detector_image_size: int = 640
    classifier_image_size: int | None = None
    conf_threshold: float = 0.25
    iou_threshold: float = 0.45
    crop_padding: float = 0.05
    min_box_size: int = 16
    device: str | None = None


@dataclass(frozen=True)
class ClassifiedDetection:
    """One detected box after the species classifier predicts a label."""

    box: tuple[int, int, int, int]
    detection_confidence: float
    label: str
    classifier_confidence: float


def normalize_cli_args(argv: list[str]) -> list[str]:
    """Allow `python yolo_detection_classifier.py image.jpg` convenience usage."""
    if not argv:
        return ["--help"]

    if argv[0] in {"-h", "--help"}:
        return argv

    if argv[0].startswith("--"):
        return argv

    maybe_path = Path(argv[0])
    if maybe_path.suffix.lower() in IMAGE_SUFFIXES:
        return ["--image-path", *argv]
    return argv


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Build the CLI for image detection/classification."""
    parser = argparse.ArgumentParser(
        description=(
            "Detect birds with YOLO, crop each bird bounding box, and classify each crop "
            "with the best EfficientNet checkpoint by default."
        )
    )
    parser.add_argument("--image-path", type=Path, required=True)
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Optional path to save an annotated image with bounding boxes and species labels",
    )
    parser.add_argument("--detector-path", type=Path, default=Path("yolo26n.pt"))
    parser.add_argument(
        "--classifier-checkpoint",
        type=Path,
        default=Path("model_artifacts") / "best_efficientnet_bird_classifier.pt",
        help="Checkpoint from efficientnet_model.py, resnet_model.py, or vgg_model.py",
    )
    parser.add_argument("--detector-image-size", type=int, default=640)
    parser.add_argument(
        "--classifier-image-size",
        type=int,
        default=None,
        help="Optional override for crop resize size. Defaults to the image_size stored in the checkpoint.",
    )
    parser.add_argument("--conf-threshold", type=float, default=0.25)
    parser.add_argument("--iou-threshold", type=float, default=0.45)
    parser.add_argument(
        "--crop-padding",
        type=float,
        default=0.05,
        help="Extra padding around the detected bird box before species classification",
    )
    parser.add_argument(
        "--min-box-size",
        type=int,
        default=16,
        help="Ignore detections smaller than this many pixels on either side",
    )
    parser.add_argument("--device", default=None, help="Optional device override, for example cuda:0")

    return parser.parse_args(normalize_cli_args(sys.argv[1:] if argv is None else argv))


def get_yolo_class() -> type:
    """Import YOLO lazily so help works even if ultralytics is not installed."""
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError(
            "ultralytics is required for yolo_detection_classifier.py. Install with: pip install ultralytics"
        ) from exc
    return YOLO


def build_config(args: argparse.Namespace) -> DetectionClassifierConfig:
    """Collect and validate CLI arguments."""
    cfg = DetectionClassifierConfig(
        detector_path=str(args.detector_path),
        classifier_checkpoint=str(args.classifier_checkpoint),
        detector_image_size=args.detector_image_size,
        classifier_image_size=args.classifier_image_size,
        conf_threshold=args.conf_threshold,
        iou_threshold=args.iou_threshold,
        crop_padding=args.crop_padding,
        min_box_size=args.min_box_size,
        device=args.device,
    )

    if cfg.detector_image_size <= 0:
        raise ValueError("detector-image-size must be greater than 0")
    if cfg.classifier_image_size is not None and cfg.classifier_image_size <= 0:
        raise ValueError("classifier-image-size must be greater than 0 when provided")
    if cfg.min_box_size < 1:
        raise ValueError("min-box-size must be at least 1")
    if cfg.crop_padding < 0:
        raise ValueError("crop-padding must be non-negative")

    return cfg


def find_bird_class_id(detector: Any) -> int | None:
    """Return the detector class ID for 'bird' if available."""
    names = getattr(detector, "names", {})
    iterable = names.items() if isinstance(names, dict) else enumerate(names)
    for class_id, name in iterable:
        if str(name).strip().lower() == "bird":
            return int(class_id)
    return None


def clamp_box(
    box: tuple[int, int, int, int],
    image_width: int,
    image_height: int,
    padding_fraction: float,
) -> tuple[int, int, int, int] | None:
    """Clamp and pad a detection box so it stays inside image bounds."""
    x1, y1, x2, y2 = box
    width = max(x2 - x1, 1)
    height = max(y2 - y1, 1)
    pad_x = int(round(width * padding_fraction))
    pad_y = int(round(height * padding_fraction))

    left = max(0, x1 - pad_x)
    top = max(0, y1 - pad_y)
    right = min(image_width, x2 + pad_x)
    bottom = min(image_height, y2 + pad_y)

    if right <= left or bottom <= top:
        return None
    return left, top, right, bottom


def extract_boxes(result: Any, min_box_size: int) -> list[tuple[tuple[int, int, int, int], float]]:
    """Extract and validate YOLO boxes as `(box, detector_confidence)` tuples."""
    if result is None or result.boxes is None or len(result.boxes) == 0:
        return []

    boxes = result.boxes
    xyxy = boxes.xyxy.cpu().tolist() if boxes.xyxy is not None else []
    confidences = boxes.conf.cpu().tolist() if boxes.conf is not None else [0.0] * len(xyxy)

    parsed: list[tuple[tuple[int, int, int, int], float]] = []
    for raw_box, det_conf in zip(xyxy, confidences):
        x1, y1, x2, y2 = [int(round(value)) for value in raw_box]
        if x2 <= x1 or y2 <= y1:
            continue
        if (x2 - x1) < min_box_size or (y2 - y1) < min_box_size:
            continue
        parsed.append(((x1, y1, x2, y2), float(det_conf)))

    return parsed


def classify_detections(
    image: Image.Image,
    boxes: list[tuple[tuple[int, int, int, int], float]],
    cfg: DetectionClassifierConfig,
    classifier: Any,
) -> list[ClassifiedDetection]:
    """Crop each detection box and classify bird species with the base model."""
    detections: list[ClassifiedDetection] = []
    image_width, image_height = image.size

    for box, det_conf in boxes:
        clamped = clamp_box(box, image_width, image_height, cfg.crop_padding)
        if clamped is None:
            continue

        crop = image.crop(clamped)
        if crop.size[0] <= 0 or crop.size[1] <= 0:
            continue

        label, cls_confidence, _ = predict_bird_from_pil(
            crop,
            classifier,
            image_size=cfg.classifier_image_size,
        )
        detections.append(
            ClassifiedDetection(
                box=box,
                detection_confidence=det_conf,
                label=label,
                classifier_confidence=cls_confidence,
            )
        )

    return detections


def annotate_image(image: Image.Image, detections: list[ClassifiedDetection]) -> Image.Image:
    """Draw detection boxes and classifier labels on the image."""
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    font = ImageFont.load_default()

    for detection in detections:
        x1, y1, x2, y2 = detection.box
        label_text = (
            f"{detection.label} | cls {detection.classifier_confidence:.2f} | det {detection.detection_confidence:.2f}"
        )
        draw.rectangle((x1, y1, x2, y2), outline=(40, 220, 80), width=3)

        text_bbox = draw.textbbox((x1, y1), label_text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        text_left = x1
        text_top = max(0, y1 - text_height - 6)

        draw.rectangle(
            (text_left, text_top, text_left + text_width + 6, text_top + text_height + 4),
            fill=(40, 220, 80),
        )
        draw.text((text_left + 3, text_top + 2), label_text, fill=(0, 0, 0), font=font)

    return annotated


def run_image(args: argparse.Namespace) -> None:
    """Run YOLO detection + base classifier handoff on one still image."""
    cfg = build_config(args)
    YOLO = get_yolo_class()

    detector = YOLO(cfg.detector_path)
    classifier = load_bird_classifier(cfg.classifier_checkpoint, device=cfg.device)

    bird_class_id = find_bird_class_id(detector)
    if bird_class_id is None:
        print("Warning: detector did not expose a 'bird' class name, so all detections will be processed.")
        predict_results = detector.predict(
            source=str(args.image_path),
            imgsz=cfg.detector_image_size,
            conf=cfg.conf_threshold,
            iou=cfg.iou_threshold,
            verbose=False,
            device=cfg.device,
        )
    else:
        print(f"Filtering detections to YOLO class 'bird' (class_id={bird_class_id}).")
        predict_results = detector.predict(
            source=str(args.image_path),
            imgsz=cfg.detector_image_size,
            conf=cfg.conf_threshold,
            iou=cfg.iou_threshold,
            verbose=False,
            device=cfg.device,
            classes=[bird_class_id],
        )

    if not predict_results:
        raise RuntimeError("YOLO predict returned no results.")

    with Image.open(args.image_path) as image_file:
        image = image_file.convert("RGB")

    raw_boxes = extract_boxes(predict_results[0], min_box_size=cfg.min_box_size)
    detections = classify_detections(image, raw_boxes, cfg, classifier)

    print(
        f"Loaded classifier: {classifier.model_name or 'unknown'} | classes={len(classifier.classes)} | "
        f"crop_size={cfg.classifier_image_size or classifier.image_size}"
    )
    print(f"Image: {args.image_path}")

    if not detections:
        print("No birds were detected in this image.")
    else:
        print(f"Detected {len(detections)} bird(s):")
        for idx, detection in enumerate(detections, start=1):
            x1, y1, x2, y2 = detection.box
            print(
                f"  [{idx}] {detection.label} - classifier_conf={detection.classifier_confidence:.4f} - "
                f"detector_conf={detection.detection_confidence:.4f} - box=({x1}, {y1}, {x2}, {y2})"
            )

    if args.output_path is not None:
        annotated = annotate_image(image, detections)
        args.output_path.parent.mkdir(parents=True, exist_ok=True)
        annotated.save(args.output_path)
        print(f"Annotated image saved to: {args.output_path}")


def main() -> None:
    """CLI entrypoint."""
    args = parse_args()
    run_image(args)


if __name__ == "__main__":
    main()
