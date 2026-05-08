from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import cast
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset
from torchvision import models, transforms

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_SPLITS = ("Train", "Test")


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Create bird .pt files and optionally load them for training."
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
        "--limit-per-class",
        type=int,
        default=None,
        help="Optional cap used for quick smoke tests.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for the simple DataLoader setup.",
    )
    parser.add_argument(
        "--prepare-training",
        action="store_true",
        help="Load the generated .pt files back into memory and build DataLoaders (only recommended for small smoke tests).",
    )
    return parser.parse_args()


def read_dataset_root(path_file: Path) -> Path:
    if not path_file.exists():
        raise FileNotFoundError(f"Could not find path file: {path_file}")

    dataset_root = Path(path_file.read_text(encoding="utf-8").strip()).expanduser()
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {dataset_root}")

    return dataset_root


def find_split_dir(dataset_root: Path, split_name: str) -> Path:
    for child in dataset_root.iterdir():
        if child.is_dir() and child.name.lower() == split_name.lower():
            return child
    raise FileNotFoundError(
        f"Could not find split folder '{split_name}' inside {dataset_root}"
    )


def collect_image_paths(class_dir: Path, limit_per_class: int | None) -> list[Path]:
    image_paths = sorted(
        path
        for path in class_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    if limit_per_class is not None:
        image_paths = image_paths[:limit_per_class]
    return image_paths


def clear_output_dir(output_dir: Path) -> None:
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
    limit_per_class: int | None,
) -> dict[str, object]:
    saved_files: list[str] = []
    per_class_counts: dict[str, int] = {}
    skipped_images: list[dict[str, str]] = []

    for class_dir in sorted(path for path in split_dir.iterdir() if path.is_dir()):
        image_paths = collect_image_paths(class_dir, limit_per_class)
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
    limit_per_class: int | None,
    summaries: dict[str, dict[str, object]],
) -> Path:
    metadata_path = output_dir / "dataset_metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "dataset_root": str(dataset_root),
                "image_size": image_size,
                "limit_per_class": limit_per_class,
                "splits": summaries,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return metadata_path


def load_generated_data(output_dir: Path) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    list[str],
]:
    X_train_parts: list[torch.Tensor] = []
    y_train: list[str] = []
    X_test_parts: list[torch.Tensor] = []
    y_test: list[str] = []

    for root, _, files in os.walk(output_dir):
        pt_files = sorted(file_name for file_name in files if file_name.lower().endswith(".pt"))
        if not pt_files:
            continue

        print(f"Current Folder: {root}")
        for file_name in pt_files:
            file_path = Path(root) / file_name
            diction = torch.load(file_path, map_location="cpu")
            if "_test.pt" in file_name.lower():
                X_test_parts.append(diction["X"])
                y_test.extend(diction["y"])
            else:
                X_train_parts.append(diction["X"])
                y_train.extend(diction["y"])

    if not X_train_parts or not X_test_parts:
        raise ValueError("Expected both train and test .pt files in the output folder.")

    unique_classes = sorted(set(y_train) | set(y_test))
    print(unique_classes)
    print(len(unique_classes))

    str_to_int = {name: index for index, name in enumerate(unique_classes)}
    print(len(str_to_int))

    y_train_tensor = torch.tensor([str_to_int[name] for name in y_train], dtype=torch.long)
    print(len(set(y_test)))
    y_test_tensor = torch.tensor([str_to_int[name] for name in y_test], dtype=torch.long)

    X_train = torch.cat(X_train_parts, dim=0)
    X_test = torch.cat(X_test_parts, dim=0)

    print(X_train.shape, y_train_tensor.shape)
    print(X_test.shape, y_test_tensor.shape)
    print(len(unique_classes))

    return X_train, y_train_tensor, X_test, y_test_tensor, unique_classes


def build_training_objects(
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_test: torch.Tensor,
    y_test: torch.Tensor,
    num_classes: int,
    batch_size: int,
) -> None:
    resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

    for param in resnet.parameters():
        param.requires_grad = False

    num_ftrs = resnet.fc.in_features
    resnet.fc = nn.Linear(num_ftrs, num_classes)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resnet = resnet.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(resnet.fc.parameters(), lr=0.001)

    dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)

    print(f"Train dataset size: {len(dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")

    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(f"Train batches: {len(train_loader)}")
    print(f"Test batches: {len(test_loader)}")
    print(f"Model device: {device}")
    print(f"Loss: {criterion.__class__.__name__}")
    print(f"Optimizer: {optimizer.__class__.__name__}")


def main() -> None:
    args = parse_args()
    if args.image_size <= 0:
        raise ValueError("image-size must be greater than 0")
    if args.batch_size <= 0:
        raise ValueError("batch-size must be greater than 0")
    if args.limit_per_class is not None and args.limit_per_class <= 0:
        raise ValueError("limit-per-class must be greater than 0 when provided")

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

    clear_output_dir(args.output_dir)

    summaries: dict[str, dict[str, object]] = {}
    for split_name in DEFAULT_SPLITS:
        split_dir = find_split_dir(dataset_root, split_name)
        print(f"\nProcessing {split_name} split from {split_dir}")
        summaries[split_name] = export_split_files(
            split_name=split_name,
            split_dir=split_dir,
            output_dir=args.output_dir,
            preprocess=preprocess,
            limit_per_class=args.limit_per_class,
        )

    metadata_path = write_metadata(
        output_dir=args.output_dir,
        dataset_root=dataset_root,
        image_size=args.image_size,
        limit_per_class=args.limit_per_class,
        summaries=summaries,
    )

    print("\nDone generating .pt files.")
    for split_name, summary in summaries.items():
        saved_files = cast(list[str], summary["saved_files"])
        print(
            f"{split_name}: {summary['saved_images']} images in "
            f"{len(saved_files)} file(s)"
        )
    print(f"Metadata written to {metadata_path}")

    if not args.prepare_training:
        print("Use --prepare-training to load the generated .pt files and build DataLoaders.")
        print("That option loads everything into RAM, so skip it for large datasets.")
        return

    print("\nLoading generated .pt files...")
    X_train, y_train, X_test, y_test, unique_classes = load_generated_data(args.output_dir)
    build_training_objects(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        num_classes=len(unique_classes),
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
