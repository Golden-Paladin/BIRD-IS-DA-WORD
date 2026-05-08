from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_SPLITS = ("Train", "Test")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for PT export."""
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Create bird .pt files from Train/Test image folders."
    )
    parser.add_argument(
        "--data-path-file",
        type=Path,
        default=script_dir / "dataPath.txt",
        help="Text file that contains the dataset root path.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=script_dir / "pt_data",
        help="Folder where the generated .pt files will be saved.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=224,
        help="Resize images to image_size x image_size.",
    )
    parser.add_argument(
        "--overwrite",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Clear old files in output-dir before writing new PT files.",
    )
    return parser.parse_args()


def read_dataset_root(path_file: Path) -> Path:
    """Read dataset root path from a text file."""
    if not path_file.exists():
        raise FileNotFoundError(f"Could not find path file: {path_file}")

    dataset_root = Path(path_file.read_text(encoding="utf-8").strip()).expanduser()
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {dataset_root}")

    return dataset_root


def find_split_dir(dataset_root: Path, split_name: str) -> Path:
    """Find a split directory by name, case-insensitive."""
    for child in dataset_root.iterdir():
        if child.is_dir() and child.name.lower() == split_name.lower():
            return child
    raise FileNotFoundError(
        f"Could not find split folder '{split_name}' inside {dataset_root}"
    )


def collect_image_paths(class_dir: Path) -> list[Path]:
    """Collect all supported image files for one class folder."""
    return sorted(
        path
        for path in class_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def clear_output_dir(output_dir: Path) -> None:
    """Remove existing output content and recreate the directory."""
    if output_dir.exists():
        for child in output_dir.iterdir():
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
    output_dir.mkdir(parents=True, exist_ok=True)


def export_split_files(
    split_name: str,
    split_dir: Path,
    output_dir: Path,
    preprocess: transforms.Compose,
) -> dict[str, object]:
    """Export one split into per-class PT files and return a summary."""
    saved_files: list[str] = []
    per_class_counts: dict[str, int] = {}
    skipped_images: list[dict[str, str]] = []

    for class_dir in sorted(path for path in split_dir.iterdir() if path.is_dir()):
        image_paths = collect_image_paths(class_dir)
        image_tensors: list[torch.Tensor] = []

        for image_path in image_paths:
            try:
                with Image.open(image_path) as image:
                    image_tensors.append(preprocess(image.convert("RGB")))
            except Exception as exc:  # pragma: no cover - defensive logging path
                skipped_images.append({"path": str(image_path), "error": str(exc)})

        per_class_counts[class_dir.name] = len(image_tensors)
        if not image_tensors:
            continue

        file_name = f"{class_dir.name}_{split_name}.pt"
        file_path = output_dir / file_name
        torch.save(
            {
                "X": torch.stack(image_tensors),
                "y": [class_dir.name] * len(image_tensors),
                "class_name": class_dir.name,
                "split": split_name,
            },
            file_path,
        )
        saved_files.append(file_name)
        print(f"Saved {file_name} with {len(image_tensors)} images")

    saved_images = sum(per_class_counts.values())
    if saved_images == 0:
        raise ValueError(f"No images were processed for split '{split_name}'.")

    return {
        "split_dir": str(split_dir),
        "saved_files": saved_files,
        "saved_images": saved_images,
        "per_class_counts": per_class_counts,
        "skipped_images": skipped_images,
    }


def write_metadata(
    output_dir: Path,
    dataset_root: Path,
    image_size: int,
    summaries: dict[str, dict[str, object]],
) -> Path:
    """Write dataset conversion metadata to JSON."""
    metadata_path = output_dir / "dataset_metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "dataset_root": str(dataset_root),
                "image_size": image_size,
                "splits": summaries,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return metadata_path


def main() -> None:
    """Run PT export for Train/Test splits and write metadata."""
    args = parse_args()
    if args.image_size <= 0:
        raise ValueError("image-size must be greater than 0")

    dataset_root = read_dataset_root(args.data_path_file)
    preprocess = transforms.Compose(
        [
            transforms.Resize((args.image_size, args.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    print(f"Dataset root: {dataset_root}")
    print(f"Output directory: {args.output_dir}")

    if args.overwrite:
        clear_output_dir(args.output_dir)
    else:
        args.output_dir.mkdir(parents=True, exist_ok=True)

    summaries: dict[str, dict[str, object]] = {}
    for split_name in DEFAULT_SPLITS:
        split_dir = find_split_dir(dataset_root, split_name)
        print(f"\nProcessing {split_name} split from {split_dir}")
        summaries[split_name] = export_split_files(
            split_name=split_name,
            split_dir=split_dir,
            output_dir=args.output_dir,
            preprocess=preprocess,
        )

    metadata_path = write_metadata(
        output_dir=args.output_dir,
        dataset_root=dataset_root,
        image_size=args.image_size,
        summaries=summaries,
    )

    print("\nDone generating .pt files.")
    for split_name, summary in summaries.items():
        saved_files = summary["saved_files"]
        print(
            f"{split_name}: {summary['saved_images']} images in "
            f"{len(saved_files)} file(s)"
        )
    print(f"Metadata written to {metadata_path}")



if __name__ == "__main__":
    main()
