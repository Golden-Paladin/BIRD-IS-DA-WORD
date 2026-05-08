from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import torch
from torchvision import transforms

from classification_metrics import compute_classification_metrics
from training_plots import save_error_curve_png

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


@dataclass
class YOLOConfig:
    """Configuration for one YOLO classification training run."""

    pt_data_dir: str = "pt_data"
    output_dir: str = "model_artifacts"
    checkpoint_name: str = "yolo_bird_classifier.pt"
    base_model: str = "yolov8n-cls.pt"
    image_size: int = 224
    batch_size: int = 4
    epochs: int = 3
    temp_data_dir: str = "yolo_cls_data"


def normalize_cli_args(argv: list[str]) -> list[str]:
    """Allow source-first prediction and train-first defaults."""
    if not argv:
        return ["train"]
    if argv[0] in {"train", "predict", "-h", "--help"}:
        return argv
    if argv[0] == "--image-path":
        return ["predict", *argv]
    if argv[0] == "--checkpoint-path":
        return ["predict", *argv]
    if len(argv) >= 2 and Path(argv[0]).suffix.lower() == ".pt":
        return ["predict", *argv]
    if Path(argv[0]).suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}:
        return ["predict", argv[0], *argv[1:]]
    return ["train", *argv]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Build CLI arguments for train and predict commands."""
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

    predict_parser = subparsers.add_parser("predict", help="Predict bird class for one image")
    predict_parser.add_argument("predict_arg1", nargs="?", type=Path)
    predict_parser.add_argument("predict_arg2", nargs="?", type=Path)
    predict_parser.add_argument("--image-path", dest="image_path_flag", type=Path, default=None, help=argparse.SUPPRESS)
    predict_parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=None,
        help="Optional checkpoint override. By default the trained YOLO checkpoint is used if present.",
    )

    args = parser.parse_args(normalize_cli_args(sys.argv[1:] if argv is None else argv))
    if args.command == "predict":
        image_suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
        args.image_path = args.image_path_flag

        if args.image_path is None:
            a1 = args.predict_arg1
            a2 = args.predict_arg2
            if a1 is not None and a2 is not None:
                if a1.suffix.lower() == ".pt" and a2.suffix.lower() in image_suffixes:
                    args.checkpoint_path = args.checkpoint_path or a1
                    args.image_path = a2
                elif a2.suffix.lower() == ".pt" and a1.suffix.lower() in image_suffixes:
                    args.checkpoint_path = args.checkpoint_path or a2
                    args.image_path = a1
                else:
                    args.checkpoint_path = args.checkpoint_path or a1
                    args.image_path = a2
            else:
                args.image_path = a1

        if args.image_path is None:
            parser.error("predict requires an image path")
    return args


def get_yolo_class() -> type:
    """Import Ultralytics YOLO lazily for faster CLI help and clearer errors."""
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError(
            "ultralytics is required for yolo_model.py. Install with: pip install ultralytics"
        ) from exc
    return YOLO


def denormalize_tensor(x_tensor: torch.Tensor) -> torch.Tensor:
    """Convert normalized tensors back to displayable RGB tensors."""
    return (x_tensor * STD + MEAN).clamp(0.0, 1.0)


def export_pt_to_imagefolders(pt_dir: Path, temp_root: Path) -> None:
    """Convert per-class PT tensors to image folders expected by YOLO classification."""
    if temp_root.exists():
        shutil.rmtree(temp_root)

    to_pil = transforms.ToPILImage()

    for split_name, split_folder in (("Train", "train"), ("Test", "val")):
        files = sorted(pt_dir.glob(f"*_{split_name}.pt"))
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


def evaluate_yolo_classifier(model: Any, val_root: Path, image_size: int) -> dict[str, float | int | str]:
    """Run a manual validation pass to compute weighted precision/recall/F1.

    Ultralytics classification training already reports accuracy-style metrics,
    but the project now standardizes on precision/recall/F1 for the best model
    across all training scripts, so we compute those explicitly here.
    """
    image_paths = sorted(path for path in val_root.glob("*/*.jpg") if path.is_file())
    if not image_paths:
        raise FileNotFoundError(f"No validation images found under {val_root}")

    names = getattr(model, "names", {})
    if isinstance(names, dict):
        class_names = [str(names[idx]) for idx in sorted(names)]
    else:
        class_names = [str(name) for name in names]
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}

    targets: list[int] = []
    predictions: list[int] = []
    for image_path in image_paths:
        class_name = image_path.parent.name
        if class_name not in class_to_idx:
            raise ValueError(f"Validation class {class_name!r} is missing from YOLO class names")
        results = model.predict(source=str(image_path), imgsz=image_size, verbose=False)
        if not results:
            raise RuntimeError(f"YOLO predict returned no results for {image_path}")
        result = results[0]
        if result.probs is None:
            raise RuntimeError(f"YOLO predict returned no classification probabilities for {image_path}")
        targets.append(class_to_idx[class_name])
        predictions.append(int(result.probs.top1))

    return compute_classification_metrics(targets, predictions, len(class_names)).to_dict()


def _pick_first_available(row: dict[str, str], candidates: list[str]) -> float | None:
    """Return the first numeric value found under the candidate column names."""
    for key in candidates:
        raw = row.get(key)
        if raw in {None, "", "nan", "NaN"}:
            continue
        try:
            return float(raw)
        except ValueError:
            continue
    return None


def _extract_yolo_acc_history(results_csv: Path) -> tuple[list[float | None], list[float | None]]:
    """Extract per-epoch train/validation accuracy histories from YOLO results.csv.

    Column names vary a bit by Ultralytics version, so we search several
    known alternatives for train/val top-1 accuracy.
    """
    if not results_csv.exists():
        return [], []

    train_candidates = [
        "train/acc",
        "train/top1_acc",
        "train/accuracy_top1",
        "train/cls_acc",
        "metrics/train_top1",
    ]
    val_candidates = [
        "metrics/accuracy_top1",
        "val/acc",
        "val/top1_acc",
        "val/accuracy_top1",
        "metrics/val_top1",
    ]

    train_history: list[float | None] = []
    val_history: list[float | None] = []
    with results_csv.open("r", encoding="utf-8", newline="") as file_obj:
        reader = csv.DictReader(file_obj)
        for row in reader:
            train_history.append(_pick_first_available(row, train_candidates))
            val_history.append(_pick_first_available(row, val_candidates))

    return train_history, val_history


def run_train(args: argparse.Namespace) -> None:
    """Train YOLO classifier from exported PT tensors and save artifacts."""
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
    )

    if cfg.image_size <= 0:
        raise ValueError("image-size must be greater than 0")

    pt_dir = Path(cfg.pt_data_dir)
    out_dir = Path(cfg.output_dir)
    temp_root = Path(cfg.temp_data_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Converting .pt tensors into YOLO classification image folders...")
    export_pt_to_imagefolders(pt_dir, temp_root)

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
    resolved_best_path = cast(Path, best_path)

    trainer_save_dir = None
    if trainer is not None and hasattr(trainer, "save_dir"):
        trainer_save_dir = Path(str(trainer.save_dir))
    elif best_path is not None:
        trainer_save_dir = best_path.parent.parent

    if trainer_save_dir is not None:
        train_acc_history, val_acc_history = _extract_yolo_acc_history(trainer_save_dir / "results.csv")
        if val_acc_history:
            plot_path = out_dir / "yolo_error_vs_epochs.png"
            save_error_curve_png(
                plot_path,
                train_acc_history=train_acc_history,
                val_acc_history=val_acc_history,
                title="YOLO Error vs Epoch",
            )
            print(f"Error curve PNG: {plot_path}")
        else:
            print("Could not extract epoch accuracy history from YOLO results.csv; skipping error curve PNG.")

    final_path = out_dir / cfg.checkpoint_name
    shutil.copy2(resolved_best_path, final_path)

    best_model = YOLO(str(resolved_best_path))
    best_metrics = evaluate_yolo_classifier(best_model, temp_root / "val", cfg.image_size)

    config_path = out_dir / "yolo_config.json"
    config_path.write_text(
        json.dumps({"config": asdict(cfg), "best_metrics": best_metrics}, indent=2),
        encoding="utf-8",
    )
    print(f"Saved checkpoint: {final_path}")
    print(
        "Best model metrics (weighted) - "
        f"precision: {best_metrics['precision_weighted']:.4f} - "
        f"recall: {best_metrics['recall_weighted']:.4f} - "
        f"f1score: {best_metrics['f1_weighted']:.4f}"
    )
    print(
        "Best model metrics (macro) - "
        f"precision: {best_metrics['precision_macro']:.4f} - "
        f"recall: {best_metrics['recall_macro']:.4f} - "
        f"f1score: {best_metrics['f1_macro']:.4f}"
    )
    print(f"Saved config: {config_path}")


def run_predict(args: argparse.Namespace) -> None:
    """Run single-image bird prediction with a YOLO classification checkpoint."""
    YOLO = get_yolo_class()
    if args.checkpoint_path is not None:
        checkpoint_path = Path(args.checkpoint_path)
    else:
        exact = Path("model_artifacts") / "yolo_bird_classifier.pt"
        if exact.exists():
            checkpoint_path = exact
        else:
            matches = sorted(Path().glob("model_artifacts/yolo_bird_classifier*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
            if not matches:
                raise FileNotFoundError("No YOLO checkpoint found. Expected model_artifacts/yolo_bird_classifier.pt")
            checkpoint_path = matches[0]
    model = YOLO(str(checkpoint_path))
    results = model.predict(source=str(args.image_path), imgsz=224, verbose=False)
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
    print(f"Checkpoint: {checkpoint_path}")


def main() -> None:
    """CLI entrypoint for YOLO training and inference."""
    args = parse_args()
    if args.command == "train":
        run_train(args)
    elif args.command == "predict":
        run_predict(args)


if __name__ == "__main__":
    main()

