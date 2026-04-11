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

# Each variant: (factory_fn, weights, params)
_VARIANTS: dict[str, tuple] = {
    "b0": (models.efficientnet_b0, models.EfficientNet_B0_Weights.IMAGENET1K_V1),   #  5.3M params
    "b1": (models.efficientnet_b1, models.EfficientNet_B1_Weights.IMAGENET1K_V1),   #  7.8M params
    "b2": (models.efficientnet_b2, models.EfficientNet_B2_Weights.IMAGENET1K_V1),   #  9.1M params  ← default
    "b3": (models.efficientnet_b3, models.EfficientNet_B3_Weights.IMAGENET1K_V1),   # 12.2M params
}


@dataclass
class EfficientNetConfig:
    pt_data_dir: str = "pt_data"
    output_dir: str = "model_artifacts"
    checkpoint_name: str = "efficientnet_bird_classifier.pt"
    model_variant: str = "b2"           # b0 | b1 | b2 | b3
    image_size: int = 224
    batch_size: int = 4
    epochs: int = 10
    learning_rate: float = 5e-4
    # 0=classifier only; 1=unfreeze last feature block; 2=last 2; ... 9=all features
    unfreeze_layers: int = 2
    backbone_lr_multiplier: float = 0.1
    lr_scheduler: str = "cosine"        # cosine | step | none
    label_smoothing: float = 0.1
    weight_decay: float = 1e-4
    dropout: float = 0.3
    augment: bool = True
    max_files: int | None = None


def normalize_cli_args(argv: list[str]) -> list[str]:
    if not argv:
        return ["train"]
    if argv[0] in {"train", "predict", "-h", "--help"}:
        return argv
    return ["train", *argv]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train or run inference with an EfficientNet bird classifier.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    p = subparsers.add_parser("train", help="Train from generated .pt files")
    p.add_argument("--pt-data-dir", type=Path, default=Path("pt_data"))
    p.add_argument("--output-dir", type=Path, default=Path("model_artifacts"))
    p.add_argument("--checkpoint-name", default="efficientnet_bird_classifier.pt")
    p.add_argument(
        "--model-variant", default="b2", choices=list(_VARIANTS),
        help="EfficientNet variant: b0 (~5M params), b1 (~8M), b2 (~9M, default), b3 (~12M)",
    )
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--learning-rate", type=float, default=5e-4)
    p.add_argument(
        "--unfreeze-layers", type=int, default=2,
        help="Feature blocks to unfreeze from the top (0=classifier only, 1=last block, 2=last 2, ...)",
    )
    p.add_argument("--backbone-lr-multiplier", type=float, default=0.1)
    p.add_argument("--lr-scheduler", default="cosine", choices=["cosine", "step", "none"])
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--dropout", type=float, default=0.3, help="Dropout before the classifier linear layer")
    p.add_argument("--augment", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--max-files", type=int, default=None, help="Optional cap for quick smoke tests")

    pred = subparsers.add_parser("predict", help="Predict bird class for one image")
    pred.add_argument("--image-path", type=Path, required=True)
    pred.add_argument(
        "--checkpoint-path", type=Path,
        default=Path("model_artifacts") / "efficientnet_bird_classifier.pt",
    )
    pred.add_argument("--image-size", type=int, default=224)

    return parser.parse_args(normalize_cli_args(sys.argv[1:] if argv is None else argv))


def create_model(num_classes: int, variant: str, unfreeze_layers: int, dropout: float) -> nn.Module:
    if variant not in _VARIANTS:
        raise ValueError(f"Unknown variant '{variant}'. Choose from: {list(_VARIANTS)}")
    model_fn, weights = _VARIANTS[variant]
    model = model_fn(weights=weights)

    # Freeze everything first
    for param in model.parameters():
        param.requires_grad = False

    # EfficientNet-Bx has model.features as a Sequential of N blocks.
    # Unfreeze the last `unfreeze_layers` blocks (counting from the end).
    feature_blocks = list(model.features.children())
    for i in range(min(unfreeze_layers, len(feature_blocks))):
        for param in feature_blocks[-(i + 1)].parameters():
            param.requires_grad = True

    # Replace classifier head (always trainable)
    in_features = model.classifier[-1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout, inplace=True),
        nn.Linear(in_features, num_classes),
    )
    return model


def _build_aug_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(p=0.1),
        transforms.RandomRotation(degrees=15),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.2)),
    ])


def run_train(args: argparse.Namespace) -> None:
    cfg = EfficientNetConfig(
        pt_data_dir=str(args.pt_data_dir),
        output_dir=str(args.output_dir),
        checkpoint_name=args.checkpoint_name,
        model_variant=args.model_variant,
        image_size=args.image_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        unfreeze_layers=args.unfreeze_layers,
        backbone_lr_multiplier=args.backbone_lr_multiplier,
        lr_scheduler=args.lr_scheduler,
        label_smoothing=args.label_smoothing,
        weight_decay=args.weight_decay,
        dropout=args.dropout,
        augment=args.augment,
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

    aug = _build_aug_transform() if cfg.augment else None
    train_dataset = LazyPtDataset(train_records, class_to_idx, transform=aug)
    test_dataset = LazyPtDataset(test_records, class_to_idx)
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False)

    print(
        f"EfficientNet-{cfg.model_variant.upper()} | "
        f"{len(train_records)} train files / {len(test_records)} test files / "
        f"{len(classes)} classes | train={len(train_dataset)} test={len(test_dataset)}"
    )

    model = create_model(
        num_classes=len(classes),
        variant=cfg.model_variant,
        unfreeze_layers=cfg.unfreeze_layers,
        dropout=cfg.dropout,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    total_p = sum(p.numel() for p in model.parameters())
    train_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {total_p:,} total | {train_p:,} trainable | device: {device}")

    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)

    # Differential learning rates: backbone much lower than new head
    backbone_params = [
        p for n, p in model.named_parameters()
        if p.requires_grad and not n.startswith("classifier")
    ]
    head_params = [
        p for n, p in model.named_parameters()
        if p.requires_grad and n.startswith("classifier")
    ]
    param_groups: list[dict] = []
    if backbone_params:
        param_groups.append({"params": backbone_params, "lr": cfg.learning_rate * cfg.backbone_lr_multiplier})
    if head_params:
        param_groups.append({"params": head_params, "lr": cfg.learning_rate})
    optimizer = torch.optim.AdamW(param_groups, weight_decay=cfg.weight_decay)

    scheduler = None
    if cfg.lr_scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    elif cfg.lr_scheduler == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(1, cfg.epochs // 3), gamma=0.5)

    best_acc = 0.0
    best_checkpoint_path = out_dir / f"best_{cfg.checkpoint_name}"
    model_name = f"efficientnet_{cfg.model_variant}"

    for epoch in range(cfg.epochs):
        model.train()
        running_loss = 0.0
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            if x_batch.shape[-1] != cfg.image_size:
                x_batch = F.interpolate(x_batch, size=(cfg.image_size, cfg.image_size), mode="bilinear", align_corners=False)
            optimizer.zero_grad()
            loss = criterion(model(x_batch), y_batch)
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

        if accuracy > best_acc:
            best_acc = accuracy
            torch.save(
                {"model_state_dict": model.state_dict(), "classes": classes, "config": asdict(cfg), "model_name": model_name},
                best_checkpoint_path,
            )

        if scheduler is not None:
            scheduler.step()

        print(f"Epoch {epoch + 1}/{cfg.epochs} - loss: {avg_loss:.4f} - val_acc: {accuracy:.4f} [best: {best_acc:.4f}]")

    checkpoint_path = out_dir / cfg.checkpoint_name
    torch.save(
        {"model_state_dict": model.state_dict(), "classes": classes, "config": asdict(cfg), "model_name": model_name},
        checkpoint_path,
    )
    config_path = out_dir / "efficientnet_config.json"
    config_path.write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")
    print(f"\nBest val_acc: {best_acc:.4f}  →  {best_checkpoint_path}")
    print(f"Final checkpoint: {checkpoint_path}")
    print(f"Config: {config_path}")


def run_predict(args: argparse.Namespace) -> None:
    checkpoint = torch.load(args.checkpoint_path, map_location="cpu")
    classes: list[str] = checkpoint["classes"]
    cfg_dict = checkpoint.get("config", {})
    variant = str(cfg_dict.get("model_variant", "b2"))
    dropout = float(cfg_dict.get("dropout", 0.3))

    model = create_model(num_classes=len(classes), variant=variant, unfreeze_layers=0, dropout=dropout)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])

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

