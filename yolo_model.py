from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torchvision import transforms

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


@dataclass
class YOLOConfig:
    pt_data_dir: str = "pt_data"
    output_dir: str = "model_artifacts"
    checkpoint_name: str = "yolo_bird_classifier.pt"
    base_model: str = "yolov8n-cls.pt"
    image_size: int = 224
    batch_size: int = 4
    epochs: int = 3
    temp_data_dir: str = "yolo_cls_data"
    max_files: int | None = None


def normalize_cli_args(argv: list[str]) -> list[str]:
    if not argv:
        return ["train"]
    if argv[0] in {"train", "predict", "-h", "--help"}:
        return argv
    return ["train", *argv]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train or run inference with a YOLO bird classifier.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train YOLO classifier from generated .pt files")
    train_parser.add_argument("--pt-data-dir", type=Path, default=Path("pt_data"))
    train_parser.add_argument("--output-dir", type=Path, default=Path("model_artifacts"))
    train_parser.add_argument("--checkpoint-name", default="yolo_bird_classifier.pt")
    train_parser.add_argument("--base-model", default="yolov8n-cls.pt")
    train_parser.add_argument("--image-size", type=int, default=224)
    train_parser.add_argument("--batch-size", type=int, default=4)
    train_parser.add_argument("--epochs", type=int, default=3)
    train_parser.add_argument("--temp-data-dir", type=Path, default=Path("yolo_cls_data"))
    train_parser.add_argument("--max-files", type=int, default=None, help="Optional limit for quick smoke tests")

    predict_parser = subparsers.add_parser("predict", help="Predict bird class for one image")
    predict_parser.add_argument("--image-path", type=Path, required=True)
    predict_parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=Path("model_artifacts") / "yolo_bird_classifier.pt",
    )
    predict_parser.add_argument("--image-size", type=int, default=224)

    return parser.parse_args(normalize_cli_args(sys.argv[1:] if argv is None else argv))


def get_yolo_class() -> type:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError(
            "ultralytics is required for yolo_model.py. Install with: pip install ultralytics"
        ) from exc
    return YOLO


def denormalize_tensor(x_tensor: torch.Tensor) -> torch.Tensor:
    return (x_tensor * STD + MEAN).clamp(0.0, 1.0)


def export_pt_to_imagefolders(pt_dir: Path, temp_root: Path, max_files: int | None = None) -> None:
    if temp_root.exists():
        shutil.rmtree(temp_root)

    to_pil = transforms.ToPILImage()

    for split_name, split_folder in (("Train", "train"), ("Test", "val")):
        files = sorted(pt_dir.glob(f"*_{split_name}.pt"))
        if max_files is not None:
            files = files[:max_files]
        if not files:
            raise FileNotFoundError(f"No files found for split '{split_name}' in {pt_dir}")

        for file_path in files:
            payload = torch.load(file_path, map_location="cpu")
            class_name = str(payload["class_name"])
            class_dir = temp_root / split_folder / class_name
            class_dir.mkdir(parents=True, exist_ok=True)

            x_data: torch.Tensor = payload["X"].float()
            for idx in range(x_data.shape[0]):
                image_tensor = denormalize_tensor(x_data[idx])
                image = to_pil(image_tensor)
                image.save(class_dir / f"{file_path.stem}_{idx:05d}.jpg", quality=95)


def run_train(args: argparse.Namespace) -> None:
    YOLO = get_yolo_class()
    cfg = YOLOConfig(
        pt_data_dir=str(args.pt_data_dir),
        output_dir=str(args.output_dir),
        checkpoint_name=args.checkpoint_name,
        base_model=args.base_model,
        image_size=args.image_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        temp_data_dir=str(args.temp_data_dir),
        max_files=args.max_files,
    )

    if cfg.image_size <= 0:
        raise ValueError("image-size must be greater than 0")

    pt_dir = Path(cfg.pt_data_dir)
    out_dir = Path(cfg.output_dir)
    temp_root = Path(cfg.temp_data_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Converting .pt tensors into YOLO classification image folders...")
    export_pt_to_imagefolders(pt_dir, temp_root, cfg.max_files)

    print("Starting YOLO training...")
    model = YOLO(cfg.base_model)
    model.train(
        data=str(temp_root),
        epochs=cfg.epochs,
        imgsz=cfg.image_size,
        batch=cfg.batch_size,
        project=str(out_dir),
        name="yolo_cls_run",
        exist_ok=True,
        verbose=True,
    )

    best_path: Path | None = None
    trainer = getattr(model, "trainer", None)
    if trainer is not None:
        trainer_best = getattr(trainer, "best", None)
        if trainer_best:
            best_path = Path(str(trainer_best))
        elif hasattr(trainer, "save_dir"):
            maybe_best = Path(trainer.save_dir) / "weights" / "best.pt"
            if maybe_best.exists():
                best_path = maybe_best

    if best_path is None or not best_path.exists():
        candidates = sorted(
            Path.cwd().glob("runs/classify/**/weights/best.pt"),
            key=lambda path: path.stat().st_mtime,
        )
        if candidates:
            best_path = candidates[-1]

    if best_path is None or not best_path.exists():
        raise FileNotFoundError("YOLO training finished but best.pt could not be located.")

    final_path = out_dir / cfg.checkpoint_name
    shutil.copy2(best_path, final_path)

    config_path = out_dir / "yolo_config.json"
    config_path.write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")
    print(f"Saved checkpoint: {final_path}")
    print(f"Saved config: {config_path}")


def run_predict(args: argparse.Namespace) -> None:
    YOLO = get_yolo_class()
    model = YOLO(str(args.checkpoint_path))
    results = model.predict(source=str(args.image_path), imgsz=args.image_size, verbose=False)
    if not results:
        raise RuntimeError("YOLO predict returned no results.")

    result = results[0]
    if result.probs is None:
        raise RuntimeError("Expected classification probabilities, but got none.")

    class_id = int(result.probs.top1)
    confidence = float(result.probs.top1conf)
    class_name = str(result.names[class_id])

    print(f"Predicted bird: {class_name}")
    print(f"Confidence: {confidence:.4f}")


def main() -> None:
    args = parse_args()
    if args.command == "train":
        run_train(args)
    elif args.command == "predict":
        run_predict(args)


if __name__ == "__main__":
    main()

