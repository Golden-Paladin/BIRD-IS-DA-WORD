from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import models, transforms

from pt_streaming import LazyPtDataset, collect_classes, scan_pt_split

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


@dataclass
class ResNetConfig:
    pt_data_dir: str = "pt_data"
    output_dir: str = "model_artifacts"
    checkpoint_name: str = "resnet_bird_classifier.pt"
    image_size: int = 224
    batch_size: int = 4
    epochs: int = 10
    learning_rate: float = 0.0005
    freeze_backbone: bool = True
    max_files: int | None = None


def normalize_cli_args(argv: list[str]) -> list[str]:
    if not argv:
        return ["train"]
    if argv[0] in {"train", "predict", "-h", "--help"}:
        return argv
    return ["train", *argv]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train or run inference with a ResNet bird classifier.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train from generated .pt files")
    train_parser.add_argument("--pt-data-dir", type=Path, default=Path("pt_data"))
    train_parser.add_argument("--output-dir", type=Path, default=Path("model_artifacts"))
    train_parser.add_argument("--checkpoint-name", default="resnet_bird_classifier.pt")
    train_parser.add_argument("--image-size", type=int, default=224)
    train_parser.add_argument("--batch-size", type=int, default=4)
    train_parser.add_argument("--epochs", type=int, default=3)
    train_parser.add_argument("--learning-rate", type=float, default=5e-4)
    train_parser.add_argument("--freeze-backbone", action=argparse.BooleanOptionalAction, default=True)
    train_parser.add_argument("--max-files", type=int, default=None, help="Optional limit for quick smoke tests")

    predict_parser = subparsers.add_parser("predict", help="Predict bird class for one image")
    predict_parser.add_argument("--image-path", type=Path, required=True)
    predict_parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=Path("model_artifacts") / "resnet_bird_classifier.pt",
    )
    predict_parser.add_argument("--image-size", type=int, default=224)

    return parser.parse_args(normalize_cli_args(sys.argv[1:] if argv is None else argv))


def create_model(num_classes: int, freeze_backbone: bool) -> nn.Module:
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    return model


def run_train(args: argparse.Namespace) -> None:
    cfg = ResNetConfig(
        pt_data_dir=str(args.pt_data_dir),
        output_dir=str(args.output_dir),
        checkpoint_name=args.checkpoint_name,
        image_size=args.image_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        freeze_backbone=args.freeze_backbone,
        max_files=args.max_files,
    )

    if cfg.image_size <= 0:
        raise ValueError("image-size must be greater than 0")

    pt_dir = Path(cfg.pt_data_dir)
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_records = scan_pt_split(pt_dir, "Train", cfg.max_files)
    test_records = scan_pt_split(pt_dir, "Test", cfg.max_files)
    classes = collect_classes(train_records, test_records)
    class_to_idx = {name: idx for idx, name in enumerate(classes)}

    train_dataset = LazyPtDataset(train_records, class_to_idx)
    test_dataset = LazyPtDataset(test_records, class_to_idx)
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False)

    print(
        f"Streaming training data from {len(train_records)} train file(s) and "
        f"{len(test_records)} test file(s) across {len(classes)} classes."
    )
    print(f"Train images: {len(train_dataset)} | Test images: {len(test_dataset)}")

    model = create_model(num_classes=len(classes), freeze_backbone=cfg.freeze_backbone)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable, lr=cfg.learning_rate)

    for epoch in range(cfg.epochs):
        model.train()
        running_loss = 0.0
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            if x_batch.shape[-1] != cfg.image_size:
                x_batch = F.interpolate(x_batch, size=(cfg.image_size, cfg.image_size), mode="bilinear", align_corners=False)

            optimizer.zero_grad()
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x_batch, y_batch in test_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                if x_batch.shape[-1] != cfg.image_size:
                    x_batch = F.interpolate(x_batch, size=(cfg.image_size, cfg.image_size), mode="bilinear", align_corners=False)
                preds = model(x_batch).argmax(dim=1)
                correct += (preds == y_batch).sum().item()
                total += y_batch.size(0)

        avg_loss = running_loss / max(len(train_loader), 1)
        accuracy = correct / max(total, 1)
        print(f"Epoch {epoch + 1}/{cfg.epochs} - loss: {avg_loss:.4f} - val_acc: {accuracy:.4f}")

    checkpoint_path = out_dir / cfg.checkpoint_name
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "classes": classes,
            "config": asdict(cfg),
            "model_name": "resnet50",
        },
        checkpoint_path,
    )
    config_path = out_dir / "resnet_config.json"
    config_path.write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")
    print(f"Saved checkpoint: {checkpoint_path}")
    print(f"Saved config: {config_path}")


def run_predict(args: argparse.Namespace) -> None:
    checkpoint = torch.load(args.checkpoint_path, map_location="cpu")
    classes: list[str] = checkpoint["classes"]

    model = create_model(num_classes=len(classes), freeze_backbone=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    transform = transforms.Compose(
        [
            transforms.Resize((args.image_size, args.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD),
        ]
    )

    with Image.open(args.image_path) as image:
        x_tensor = transform(image.convert("RGB")).unsqueeze(0)

    with torch.no_grad():
        probs = torch.softmax(model(x_tensor), dim=1)
    top_idx = int(probs.argmax(dim=1).item())
    top_conf = float(probs[0, top_idx].item())

    print(f"Predicted bird: {classes[top_idx]}")
    print(f"Confidence: {top_conf:.4f}")


def main() -> None:
    args = parse_args()
    if args.command == "train":
        run_train(args)
    elif args.command == "predict":
        run_predict(args)


if __name__ == "__main__":
    main()

