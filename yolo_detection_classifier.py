from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from PIL import Image, ImageDraw, ImageFont

from bird_classifier_inference import load_bird_classifier, predict_bird_from_pil

# Common image suffixes used to auto-detect when the user is passing an image
# path directly without an explicit sub-command.
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


@dataclass(frozen=True)
class DetectionClassifierConfig:
    """Shared configuration for YOLO detection -> classifier handoff.

    The detector finds bounding boxes, then each crop is passed into one of the
    base classifiers (EfficientNet by default) to predict the bird species.
    """

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
    """One YOLO detection after being classified by the base model."""

    box: tuple[int, int, int, int]
    detection_confidence: float
    detector_class_id: int
    label: str
    classifier_confidence: float


@dataclass
class TrackedBird:
    """Cached classifier output for one tracked bird in video mode."""

    label: str
    confidence: float
    last_seen_frame: int
    box: tuple[int, int, int, int]
    detection_confidence: float


def normalize_cli_args(argv: list[str]) -> list[str]:
    """Support image-first usage without forcing the user to type a sub-command.

    Examples supported:
    - python yolo_detection_classifier.py image --image-path bird.jpg
    - python yolo_detection_classifier.py --image-path bird.jpg
    - python yolo_detection_classifier.py bird.jpg
    - python yolo_detection_classifier.py video --source birds.mp4
    - python yolo_detection_classifier.py run --source birds.mp4
    """
    if not argv:
        return ["image", "--help"]
    if argv[0] in {"image", "video", "run", "-h", "--help"}:
        return argv
    if argv[0].startswith("--"):
        return ["image", *argv]

    maybe_path = Path(argv[0])
    if maybe_path.suffix.lower() in IMAGE_SUFFIXES:
        return ["image", "--image-path", *argv]
    return ["image", *argv]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Build the CLI for still-image and optional video detection/classification."""
    parser = argparse.ArgumentParser(
        description=(
            "Detect birds with YOLO, crop each bird bounding box, and classify the crop "
            "with the best EfficientNet checkpoint by default."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    image_parser = subparsers.add_parser(
        "image",
        help="Detect birds in one image, then classify each bird crop with a base model",
    )
    image_parser.add_argument("--image-path", type=Path, required=True)
    image_parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Optional path to save an annotated image with bounding boxes and species labels",
    )
    add_common_detection_arguments(image_parser)

    video_parser = subparsers.add_parser(
        "video",
        aliases=["run"],
        help="Video mode: detect birds, track them, and classify each track once",
    )
    video_parser.add_argument("--source", required=True, help="Video path or webcam index such as 0")
    video_parser.add_argument("--output-path", type=Path, default=None, help="Optional annotated video output path")
    video_parser.add_argument(
        "--track-ttl-frames",
        type=int,
        default=30,
        help="How many missed frames before a bird is considered gone and its cached label is cleared",
    )
    video_parser.add_argument("--show", action=argparse.BooleanOptionalAction, default=True)
    add_common_detection_arguments(video_parser)

    return parser.parse_args(normalize_cli_args(sys.argv[1:] if argv is None else argv))


def add_common_detection_arguments(parser: argparse.ArgumentParser) -> None:
    """Attach the detector/classifier arguments shared by image and video modes."""
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
    parser.add_argument("--device", default=None, help="Optional Ultralytics / classifier device override, for example cuda:0")


def get_yolo_class() -> type:
    """Import YOLO lazily so `--help` still works even if ultralytics is missing."""
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError(
            "ultralytics is required for yolo_detection_classifier.py. Install with: pip install ultralytics"
        ) from exc
    return YOLO


def get_cv2():
    """Import OpenCV lazily so the image-only workflow can avoid it when unused."""
    try:
        import cv2
    except ImportError as exc:
        raise ImportError(
            "opencv-python is required for video detection/classification. Install with: pip install opencv-python"
        ) from exc
    return cv2


def build_common_config(args: argparse.Namespace) -> DetectionClassifierConfig:
    """Collect and validate the shared detector/classifier arguments."""
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


def clamp_box(
    box: tuple[int, int, int, int],
    frame_width: int,
    frame_height: int,
    padding_fraction: float,
) -> tuple[int, int, int, int] | None:
    """Clamp a padded crop box so it always stays inside the image bounds."""
    x1, y1, x2, y2 = box
    width = max(x2 - x1, 1)
    height = max(y2 - y1, 1)
    pad_x = int(round(width * padding_fraction))
    pad_y = int(round(height * padding_fraction))

    left = max(0, x1 - pad_x)
    top = max(0, y1 - pad_y)
    right = min(frame_width, x2 + pad_x)
    bottom = min(frame_height, y2 + pad_y)
    if right <= left or bottom <= top:
        return None
    return left, top, right, bottom


def crop_bird_from_pil(image: Image.Image, box: tuple[int, int, int, int], padding_fraction: float) -> Image.Image | None:
    """Extract a padded crop from a PIL image for the species classifier."""
    width, height = image.size
    clamped = clamp_box(box, width, height, padding_fraction)
    if clamped is None:
        return None
    crop = image.crop(clamped)
    if crop.size[0] <= 0 or crop.size[1] <= 0:
        return None
    return crop


def crop_bird_from_frame(frame, box: tuple[int, int, int, int], padding_fraction: float) -> Image.Image | None:
    """Extract a padded RGB crop for the classifier from a BGR OpenCV frame."""
    frame_height, frame_width = frame.shape[:2]
    clamped = clamp_box(box, frame_width, frame_height, padding_fraction)
    if clamped is None:
        return None
    left, top, right, bottom = clamped
    crop = frame[top:bottom, left:right]
    if crop.size == 0:
        return None
    cv2 = get_cv2()
    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    return Image.fromarray(cast(Any, crop_rgb))


def find_bird_class_id(detector) -> int | None:
    """Return the detector class ID for COCO 'bird' if the model exposes one."""
    names = getattr(detector, "names", {})
    iterable = names.items() if isinstance(names, dict) else enumerate(names)
    for class_id, name in iterable:
        if str(name).strip().lower() == "bird":
            return int(class_id)
    return None


def resolve_detection_classes(detector) -> list[int] | None:
    """Restrict detections to the YOLO 'bird' class when available."""
    bird_class_id = find_bird_class_id(detector)
    if bird_class_id is None:
        print("Warning: detector did not expose a 'bird' class name, so all detections will be processed.")
        return None
    print(f"Filtering detections to YOLO class 'bird' (class_id={bird_class_id}).")
    return [bird_class_id]


def extract_boxes(result, detection_classes: list[int] | None, min_box_size: int) -> list[tuple[tuple[int, int, int, int], float, int]]:
    """Convert Ultralytics boxes into a clean Python structure.

    Each output tuple contains `(box, detector_confidence, detector_class_id)`.
    """
    if result is None or result.boxes is None or len(result.boxes) == 0:
        return []

    boxes = result.boxes
    xyxy = boxes.xyxy.cpu().tolist() if boxes.xyxy is not None else []
    confidences = boxes.conf.cpu().tolist() if boxes.conf is not None else [0.0] * len(xyxy)
    class_ids = boxes.cls.cpu().tolist() if boxes.cls is not None else [0] * len(xyxy)
    allowed_class_ids = set(detection_classes) if detection_classes is not None else None

    parsed: list[tuple[tuple[int, int, int, int], float, int]] = []
    for raw_box, det_conf, class_id in zip(xyxy, confidences, class_ids):
        detector_class_id = int(class_id)
        if allowed_class_ids is not None and detector_class_id not in allowed_class_ids:
            continue

        x1, y1, x2, y2 = [int(round(value)) for value in raw_box]
        if x2 <= x1 or y2 <= y1:
            continue
        if (x2 - x1) < min_box_size or (y2 - y1) < min_box_size:
            continue
        parsed.append(((x1, y1, x2, y2), float(det_conf), detector_class_id))
    return parsed


def classify_boxes_in_image(
    image: Image.Image,
    boxes: list[tuple[tuple[int, int, int, int], float, int]],
    classifier,
    cfg: DetectionClassifierConfig,
) -> list[ClassifiedDetection]:
    """Crop each YOLO box and ask the base classifier which bird it is."""
    detections: list[ClassifiedDetection] = []
    for box, det_conf, detector_class_id in boxes:
        crop_image = crop_bird_from_pil(image, box, padding_fraction=cfg.crop_padding)
        if crop_image is None:
            continue
        label, cls_confidence, _ = predict_bird_from_pil(
            crop_image,
            classifier,
            image_size=cfg.classifier_image_size,
        )
        detections.append(
            ClassifiedDetection(
                box=box,
                detection_confidence=det_conf,
                detector_class_id=detector_class_id,
                label=label,
                classifier_confidence=cls_confidence,
            )
        )
    return detections


def annotate_image(image: Image.Image, detections: list[ClassifiedDetection]) -> Image.Image:
    """Draw bird boxes and EfficientNet labels on top of the original image."""
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
    """Detect birds in one image and classify each detected bird crop."""
    cfg = build_common_config(args)
    YOLO = get_yolo_class()
    detector = YOLO(cfg.detector_path)
    classifier = load_bird_classifier(cfg.classifier_checkpoint, device=cfg.device)
    detection_classes = resolve_detection_classes(detector)

    predict_kwargs: dict[str, Any] = {
        "source": str(args.image_path),
        "imgsz": cfg.detector_image_size,
        "conf": cfg.conf_threshold,
        "iou": cfg.iou_threshold,
        "verbose": False,
    }
    if cfg.device is not None:
        predict_kwargs["device"] = cfg.device

    if detection_classes is not None:
        results = detector.predict(**predict_kwargs, classes=cast(Any, detection_classes))
    else:
        results = detector.predict(**predict_kwargs)
    if not results:
        raise RuntimeError("YOLO predict returned no results.")

    with Image.open(args.image_path) as image_file:
        image = image_file.convert("RGB")

    raw_boxes = extract_boxes(results[0], detection_classes=detection_classes, min_box_size=cfg.min_box_size)
    detections = classify_boxes_in_image(image, raw_boxes, classifier, cfg)

    print(
        f"Loaded classifier: {classifier.model_name or 'unknown'} | classes={len(classifier.classes)} | "
        f"crop_size={cfg.classifier_image_size or classifier.image_size}"
    )
    print(f"Image: {args.image_path}")
    if not detections:
        print("No birds were detected in this image.")
    else:
        print(f"Detected {len(detections)} bird(s):")
        for index, detection in enumerate(detections, start=1):
            x1, y1, x2, y2 = detection.box
            print(
                f"  [{index}] {detection.label} - classifier_conf={detection.classifier_confidence:.4f} - "
                f"detector_conf={detection.detection_confidence:.4f} - box=({x1}, {y1}, {x2}, {y2})"
            )

    if args.output_path is not None:
        annotated = annotate_image(image, detections)
        args.output_path.parent.mkdir(parents=True, exist_ok=True)
        annotated.save(args.output_path)
        print(f"Annotated image saved to: {args.output_path}")


def resolve_video_source(source: str) -> int | str:
    """Interpret numeric strings as webcam indices; otherwise treat as file paths/URLs."""
    return int(source) if source.isdigit() else source


def draw_track_overlay(frame, bird: TrackedBird, track_id: int) -> None:
    """Render the bounding box and cached species label onto the frame."""
    cv2 = get_cv2()
    x1, y1, x2, y2 = bird.box
    cv2.rectangle(frame, (x1, y1), (x2, y2), (40, 220, 80), 2)
    label = f"ID {track_id} | {bird.label} | cls {bird.confidence:.2f} | det {bird.detection_confidence:.2f}"
    text_origin = (x1, max(20, y1 - 10))
    cv2.putText(frame, label, text_origin, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (40, 220, 80), 2, cv2.LINE_AA)


def expire_missing_tracks(active_tracks: dict[int, TrackedBird], frame_index: int, ttl_frames: int) -> None:
    """Drop tracks that have not been seen recently so re-entering birds can be reclassified."""
    expired_ids = [
        track_id
        for track_id, bird in active_tracks.items()
        if frame_index - bird.last_seen_frame > ttl_frames
    ]
    for track_id in expired_ids:
        bird = active_tracks.pop(track_id)
        print(f"[frame {frame_index}] Bird track {track_id} left the scene -> cleared cached label {bird.label}")


def open_video_capture(source: int | str):
    """Open a webcam or video file and fail fast with a clear message if it cannot be read."""
    cv2 = get_cv2()
    capture = cv2.VideoCapture(int(source) if isinstance(source, int) else str(source))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video source: {source}")
    return capture


def create_video_writer(output_path: Path, fps: float, frame_width: int, frame_height: int):
    """Create an annotated-video writer using mp4v for broad compatibility."""
    cv2 = get_cv2()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = getattr(cv2, "VideoWriter_fourcc")(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (frame_width, frame_height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not create output video writer at {output_path}")
    return writer


def run_video(args: argparse.Namespace) -> None:
    """Optional video mode retained for tracked bird labeling across frames."""
    cfg = build_common_config(args)
    if args.track_ttl_frames < 0:
        raise ValueError("track-ttl-frames must be non-negative")

    YOLO = get_yolo_class()
    cv2 = get_cv2()
    detector = YOLO(cfg.detector_path)
    classifier = load_bird_classifier(cfg.classifier_checkpoint, device=cfg.device)
    detection_classes = resolve_detection_classes(detector)

    print(
        f"Loaded classifier: {classifier.model_name or 'unknown'} | classes={len(classifier.classes)} | "
        f"crop_size={cfg.classifier_image_size or classifier.image_size}"
    )

    source = resolve_video_source(args.source)
    capture = open_video_capture(source)
    fps = capture.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0
    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = None
    if args.output_path is not None:
        writer = create_video_writer(args.output_path, fps, frame_width, frame_height)

    active_tracks: dict[int, TrackedBird] = {}
    frame_index = 0
    warned_about_missing_ids = False

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frame_index += 1

            track_kwargs: dict[str, Any] = {
                "source": frame,
                "persist": True,
                "verbose": False,
                "conf": cfg.conf_threshold,
                "iou": cfg.iou_threshold,
                "imgsz": cfg.detector_image_size,
            }
            if cfg.device is not None:
                track_kwargs["device"] = cfg.device
            if detection_classes is not None:
                track_kwargs["classes"] = detection_classes

            results = detector.track(**track_kwargs)
            result = results[0] if results else None
            if result is not None and result.boxes is not None and len(result.boxes) > 0:
                boxes = result.boxes
                xyxy = boxes.xyxy.cpu().tolist() if boxes.xyxy is not None else []
                confidences = boxes.conf.cpu().tolist() if boxes.conf is not None else [0.0] * len(xyxy)
                class_ids = boxes.cls.cpu().tolist() if boxes.cls is not None else [0] * len(xyxy)
                if boxes.id is None:
                    if not warned_about_missing_ids:
                        print(
                            "Warning: tracker returned no persistent IDs, so detections without IDs will be classified each frame."
                        )
                        warned_about_missing_ids = True
                    track_ids = [-(frame_index * 1000 + idx + 1) for idx in range(len(xyxy))]
                else:
                    track_ids = [int(track_id) for track_id in boxes.id.int().cpu().tolist()]

                allowed_class_ids = set(detection_classes) if detection_classes is not None else None
                for track_id, raw_box, det_conf, class_id in zip(track_ids, xyxy, confidences, class_ids):
                    detector_class_id = int(class_id)
                    if allowed_class_ids is not None and detector_class_id not in allowed_class_ids:
                        continue

                    x1, y1, x2, y2 = [int(round(value)) for value in raw_box]
                    if x2 <= x1 or y2 <= y1:
                        continue
                    if (x2 - x1) < cfg.min_box_size or (y2 - y1) < cfg.min_box_size:
                        continue

                    current_box = (x1, y1, x2, y2)
                    tracked_bird = active_tracks.get(track_id)
                    if tracked_bird is None:
                        crop_image = crop_bird_from_frame(frame, current_box, padding_fraction=cfg.crop_padding)
                        if crop_image is None:
                            continue
                        label, cls_confidence, _ = predict_bird_from_pil(
                            crop_image,
                            classifier,
                            image_size=cfg.classifier_image_size,
                        )
                        tracked_bird = TrackedBird(
                            label=label,
                            confidence=cls_confidence,
                            last_seen_frame=frame_index,
                            box=current_box,
                            detection_confidence=float(det_conf),
                        )
                        active_tracks[track_id] = tracked_bird
                        print(
                            f"[frame {frame_index}] Bird track {track_id} entered -> {label} "
                            f"(classifier_conf={cls_confidence:.4f}, detector_conf={float(det_conf):.4f})"
                        )
                    else:
                        tracked_bird.last_seen_frame = frame_index
                        tracked_bird.box = current_box
                        tracked_bird.detection_confidence = float(det_conf)
                    draw_track_overlay(frame, tracked_bird, track_id)

            expire_missing_tracks(active_tracks, frame_index=frame_index, ttl_frames=args.track_ttl_frames)
            if writer is not None:
                writer.write(frame)
            if args.show:
                cv2.imshow("YOLO Bird Detection + Base Classifier", frame)
                key = cv2.waitKey(1) & 0xFF
                if key in {27, ord("q")}:
                    break
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        if args.show:
            cv2.destroyAllWindows()

    print(f"Processed {frame_index} frames.")
    if args.output_path is not None:
        print(f"Annotated video saved to: {args.output_path}")


def main() -> None:
    """CLI entrypoint."""
    args = parse_args()
    if args.command == "image":
        run_image(args)
    elif args.command in {"video", "run"}:
        run_video(args)


if __name__ == "__main__":
    main()
