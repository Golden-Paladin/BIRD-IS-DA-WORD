from __future__ import annotations

"""YOLO detection -> base classifier handoff for images and video.

Pass one source path:
1. If it is an image, YOLO finds boxes and classifies each crop.
2. If it is a video, YOLO scans the full video, keeps the best bird box, and classifies that one crop once.
"""

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from bird_classifier_inference import load_bird_classifier, predict_bird_from_pil, resolve_checkpoint_path

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


@dataclass(frozen=True)
class DetectionClassifierConfig:
    """Shared detector/classifier configuration."""

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
    """One detection after species classification."""

    box: tuple[int, int, int, int]
    detection_confidence: float
    label: str
    classifier_confidence: float


def normalize_cli_args(argv: list[str]) -> list[str]:
    """Keep the CLI source-first and minimal."""
    if not argv:
        return ["--help"]

    if argv[0] in {"-h", "--help"}:
        return argv
    return argv


def add_common_args(parser: argparse.ArgumentParser) -> None:
    """Attach shared detector/classifier arguments."""
    parser.add_argument("--detector-path", type=Path, default=Path("yolo26n.pt"))
    parser.add_argument(
        "--checkpoint-path",
        "--classifier-checkpoint",
        dest="classifier_checkpoint",
        type=Path,
        default=None,
        help="Optional base-model checkpoint override. Defaults to the best EfficientNet checkpoint if present.",
    )
    parser.add_argument("--detector-image-size", type=int, default=640, help=argparse.SUPPRESS)
    parser.add_argument(
        "--classifier-image-size",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--conf-threshold", type=float, default=0.25, help=argparse.SUPPRESS)
    parser.add_argument("--iou-threshold", type=float, default=0.45, help=argparse.SUPPRESS)
    parser.add_argument(
        "--crop-padding",
        type=float,
        default=0.05,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--min-box-size",
        type=int,
        default=16,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--device", default=None, help=argparse.SUPPRESS)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Build a minimal source-first CLI for YOLO inference."""
    parser = argparse.ArgumentParser(
        description=(
            "Pass one image or video path. YOLO will detect birds and the best EfficientNet checkpoint will classify them."
        )
    )
    parser.add_argument("source", help="Image path, video path, or webcam index like 0")
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Optional output image path. For videos this saves the best detected frame.",
    )
    add_common_args(parser)
    return parser.parse_args(normalize_cli_args(sys.argv[1:] if argv is None else argv))


def source_is_image(source: str) -> bool:
    """Return True when the provided source looks like an image file."""
    return Path(source).suffix.lower() in IMAGE_SUFFIXES


def get_yolo_class() -> type:
    """Import YOLO lazily so help works even if ultralytics is missing."""
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError(
            "ultralytics is required for yolo_detection_classifier.py. Install with: pip install ultralytics"
        ) from exc
    return YOLO


def get_cv2() -> Any:
    """Import OpenCV lazily so image mode can run without importing it first."""
    try:
        import cv2
    except ImportError as exc:
        raise ImportError(
            "opencv-python is required for video mode. Install with: pip install opencv-python"
        ) from exc
    return cv2


def build_config(args: argparse.Namespace) -> DetectionClassifierConfig:
    """Collect and validate shared args."""
    checkpoint_path = resolve_checkpoint_path(
        args.classifier_checkpoint,
        [Path("model_artifacts") / "best_efficientnet_bird_classifier.pt"],
    )
    cfg = DetectionClassifierConfig(
        detector_path=str(args.detector_path),
        classifier_checkpoint=str(checkpoint_path),
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
    """Return the detector class ID for 'bird' if present."""
    names = getattr(detector, "names", {})
    iterable = names.items() if isinstance(names, dict) else enumerate(names)
    for class_id, name in iterable:
        if str(name).strip().lower() == "bird":
            return int(class_id)
    return None


def clamp_box(
    box: tuple[int, int, int, int],
    width: int,
    height: int,
    padding_fraction: float,
) -> tuple[int, int, int, int] | None:
    """Clamp and pad a box so it stays inside image bounds."""
    x1, y1, x2, y2 = box
    bw = max(x2 - x1, 1)
    bh = max(y2 - y1, 1)
    pad_x = int(round(bw * padding_fraction))
    pad_y = int(round(bh * padding_fraction))

    left = max(0, x1 - pad_x)
    top = max(0, y1 - pad_y)
    right = min(width, x2 + pad_x)
    bottom = min(height, y2 + pad_y)
    if right <= left or bottom <= top:
        return None
    return left, top, right, bottom


def extract_boxes(result: Any, min_box_size: int) -> list[tuple[tuple[int, int, int, int], float]]:
    """Parse YOLO output into `(box, confidence)` pairs."""
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
    """Crop each detection and classify bird species with base model."""
    detections: list[ClassifiedDetection] = []
    width, height = image.size

    for box, det_conf in boxes:
        clamped = clamp_box(box, width, height, cfg.crop_padding)
        if clamped is None:
            continue

        crop = image.crop(clamped)
        if crop.size[0] <= 0 or crop.size[1] <= 0:
            continue

        label, cls_conf, _ = predict_bird_from_pil(crop, classifier, image_size=cfg.classifier_image_size)
        detections.append(
            ClassifiedDetection(
                box=box,
                detection_confidence=det_conf,
                label=label,
                classifier_confidence=cls_conf,
            )
        )
    return detections


def annotate_image(image: Image.Image, detections: list[ClassifiedDetection]) -> Image.Image:
    """Draw boxes + labels for image mode output."""
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    font = ImageFont.load_default()

    for det in detections:
        x1, y1, x2, y2 = det.box
        text = f"{det.label} | cls {det.classifier_confidence:.2f} | det {det.detection_confidence:.2f}"
        draw.rectangle((x1, y1, x2, y2), outline=(40, 220, 80), width=3)

        text_bbox = draw.textbbox((x1, y1), text, font=font)
        tw = text_bbox[2] - text_bbox[0]
        th = text_bbox[3] - text_bbox[1]
        tl = x1
        tt = max(0, y1 - th - 6)

        draw.rectangle((tl, tt, tl + tw + 6, tt + th + 4), fill=(40, 220, 80))
        draw.text((tl + 3, tt + 2), text, fill=(0, 0, 0), font=font)

    return annotated


def run_image(args: argparse.Namespace) -> None:
    """Detect + classify birds in one still image."""
    cfg = build_config(args)
    YOLO = get_yolo_class()

    detector = YOLO(cfg.detector_path)
    classifier = load_bird_classifier(cfg.classifier_checkpoint, device=cfg.device)

    bird_class_id = find_bird_class_id(detector)
    predict_kwargs: dict[str, Any] = {
        "source": str(args.source),
        "imgsz": cfg.detector_image_size,
        "conf": cfg.conf_threshold,
        "iou": cfg.iou_threshold,
        "verbose": False,
        "device": cfg.device,
    }
    if bird_class_id is None:
        print("Warning: detector did not expose a 'bird' class name, so all detections will be processed.")
    else:
        print(f"Filtering detections to YOLO class 'bird' (class_id={bird_class_id}).")
        predict_kwargs["classes"] = [bird_class_id]

    results = detector.predict(**predict_kwargs)
    if not results:
        raise RuntimeError("YOLO predict returned no results.")

    with Image.open(args.source) as image_file:
        image = image_file.convert("RGB")

    raw_boxes = extract_boxes(results[0], min_box_size=cfg.min_box_size)
    detections = classify_detections(image, raw_boxes, cfg, classifier)

    print(
        f"Loaded classifier: {classifier.model_name or 'unknown'} | classes={len(classifier.classes)} | "
        f"crop_size={cfg.classifier_image_size or classifier.image_size}"
    )
    print(f"Image: {args.source}")

    if not detections:
        print("No birds were detected in this image.")
    else:
        print(f"Detected {len(detections)} bird(s):")
        for idx, det in enumerate(detections, start=1):
            x1, y1, x2, y2 = det.box
            print(
                f"  [{idx}] {det.label} - classifier_conf={det.classifier_confidence:.4f} - "
                f"detector_conf={det.detection_confidence:.4f} - box=({x1}, {y1}, {x2}, {y2})"
            )

    if args.output_path is not None:
        annotated = annotate_image(image, detections)
        args.output_path.parent.mkdir(parents=True, exist_ok=True)
        annotated.save(args.output_path)
        print(f"Annotated image saved to: {args.output_path}")


def resolve_video_source(source: str) -> int | str:
    """Interpret numeric strings as webcam index, otherwise use path/URL string."""
    return int(source) if source.isdigit() else source


def run_video(args: argparse.Namespace) -> None:
    """Scan video, keep the single best bird detection, and classify once."""
    cfg = build_config(args)

    YOLO = get_yolo_class()
    cv2 = get_cv2()

    detector = YOLO(cfg.detector_path)
    classifier = load_bird_classifier(cfg.classifier_checkpoint, device=cfg.device)

    bird_class_id = find_bird_class_id(detector)
    detection_classes: list[int] | None = None
    if bird_class_id is None:
        print("Warning: detector did not expose a 'bird' class name, so all detections will be processed.")
    else:
        detection_classes = [bird_class_id]
        print(f"Filtering detections to YOLO class 'bird' (class_id={bird_class_id}).")

    print(
        f"Loaded classifier: {classifier.model_name or 'unknown'} | classes={len(classifier.classes)} | "
        f"crop_size={cfg.classifier_image_size or classifier.image_size}"
    )

    source = resolve_video_source(args.source)
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video source: {args.source}")

    fps = capture.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    best_frame_rgb: Image.Image | None = None
    best_box: tuple[int, int, int, int] | None = None
    best_conf = -1.0
    best_frame_idx = -1

    frame_idx = 0

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frame_idx += 1

            predict_kwargs: dict[str, Any] = {
                "source": frame,
                "verbose": False,
                "conf": cfg.conf_threshold,
                "iou": cfg.iou_threshold,
                "imgsz": cfg.detector_image_size,
                "device": cfg.device,
            }
            if detection_classes is not None:
                predict_kwargs["classes"] = detection_classes

            results = detector.predict(**predict_kwargs)
            result = results[0] if results else None
            for box, det_conf in extract_boxes(result, min_box_size=cfg.min_box_size):
                if det_conf <= best_conf:
                    continue
                best_conf = det_conf
                best_box = box
                best_frame_idx = frame_idx
                best_frame_rgb = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        capture.release()

    print(f"Scanned {frame_idx} frames.")
    if best_frame_rgb is None or best_box is None:
        print("No birds were detected in this video.")
        return

    clamped = clamp_box(best_box, best_frame_rgb.width, best_frame_rgb.height, cfg.crop_padding)
    if clamped is None:
        print("A best detection was found but crop bounds were invalid.")
        return

    crop_image = best_frame_rgb.crop(clamped)
    label, cls_conf, _ = predict_bird_from_pil(crop_image, classifier, image_size=cfg.classifier_image_size)
    x1, y1, x2, y2 = best_box
    print(
        f"Best detection: frame={best_frame_idx} box=({x1}, {y1}, {x2}, {y2}) "
        f"detector_conf={best_conf:.4f}"
    )
    print(f"Predicted bird: {label}")
    print(f"Classifier confidence: {cls_conf:.4f}")

    if args.output_path is not None:
        annotated = annotate_image(
            best_frame_rgb,
            [
                ClassifiedDetection(
                    box=best_box,
                    detection_confidence=best_conf,
                    label=label,
                    classifier_confidence=cls_conf,
                )
            ],
        )
        args.output_path.parent.mkdir(parents=True, exist_ok=True)
        annotated.save(args.output_path)
        print(f"Annotated best frame saved to: {args.output_path}")


def main() -> None:
    """CLI entrypoint."""
    args = parse_args()
    if source_is_image(str(args.source)):
        run_image(args)
    else:
        run_video(args)


if __name__ == "__main__":
    main()
