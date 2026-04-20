from __future__ import annotations

import argparse
import json
import sys
import time                          # wall-clock timing for each epoch
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import models, transforms

from pt_streaming import LazyPtDataset, collect_classes, scan_pt_split

# ImageNet normalisation constants — VGG-16 pretrained weights require these.
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]


@dataclass
class VGGConfig:
    """All hyper-parameters and paths for one training run.

    Written to JSON in model_artifacts/ so every checkpoint is fully reproducible.
    """
    pt_data_dir: str = "pt_data"
    output_dir: str = "model_artifacts"
    checkpoint_name: str = "vgg_bird_classifier.pt"
    image_size: int = 224
    batch_size: int = 4
    epochs: int = 10
    learning_rate: float = 5e-4
    # 0=all features frozen; 1=unfreeze last conv block; 2=last 2 blocks; ... 5=all features
    unfreeze_layers: int = 1
    backbone_lr_multiplier: float = 0.1
    lr_scheduler: str = "cosine"   # cosine | step | none
    label_smoothing: float = 0.1
    weight_decay: float = 1e-4
    dropout: float = 0.5           # VGG already has dropout in its classifier; default higher than ResNet
    augment: bool = True
    max_files: int | None = None
    # --- adaptive LR ---
    # When True, the learning rate of all parameter groups is scaled down by
    # adaptive_lr_factor whenever val_acc drops relative to the previous epoch.
    adaptive_lr: bool = False
    adaptive_lr_factor: float = 0.5


def normalize_cli_args(argv: list[str]) -> list[str]:
    """Silently prepend 'train' when the user omits the sub-command.

    Allows `python vgg_model.py --epochs 10` instead of the full
    `python vgg_model.py train --epochs 10`.
    """
    if not argv:
        return ["train"]
    if argv[0] in {"train", "predict", "-h", "--help"}:
        return argv
    return ["train", *argv]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Build the full CLI with 'train' and 'predict' sub-commands."""
    parser = argparse.ArgumentParser(description="Train or run inference with a VGG bird classifier.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # ── train sub-command ────────────────────────────────────────────────────
    p = subparsers.add_parser("train", help="Train from generated .pt files")
    p.add_argument("--pt-data-dir", type=Path, default=Path("pt_data"))
    p.add_argument("--output-dir", type=Path, default=Path("model_artifacts"))
    p.add_argument("--checkpoint-name", default="vgg_bird_classifier.pt")
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--learning-rate", type=float, default=5e-4)
    p.add_argument(
        "--unfreeze-layers", type=int, default=1,
        help="VGG conv blocks to unfreeze from top (0=all frozen, 1=last block, 2=last 2, ..., 5=all features)",
    )
    p.add_argument("--backbone-lr-multiplier", type=float, default=0.1,
                   help="Backbone LR = head LR × this value (prevents catastrophic forgetting of pretrained features)")
    p.add_argument("--lr-scheduler", default="cosine", choices=["cosine", "step", "none"],
                   help="cosine: smooth decay; step: halve every 1/3 epochs; none: fixed LR")
    p.add_argument("--label-smoothing", type=float, default=0.1,
                   help="Cross-entropy label smoothing — reduces overconfidence on 200 classes")
    p.add_argument("--weight-decay", type=float, default=1e-4,
                   help="AdamW L2 regularisation coefficient")
    p.add_argument("--dropout", type=float, default=0.5, help="Dropout rate in the classifier (default 0.5)")
    p.add_argument("--augment", action=argparse.BooleanOptionalAction, default=True,
                   help="Random flip / rotate / erase applied every training epoch")
    p.add_argument("--max-files", type=int, default=None, help="Optional cap for quick smoke tests")
    # adaptive LR flags
    p.add_argument(
        "--adaptive-lr", action=argparse.BooleanOptionalAction, default=False,
        help=(
            "Enable adaptive LR reduction. "
            "When val_acc falls vs the previous epoch every LR group is "
            "multiplied by --adaptive-lr-factor."
        ),
    )
    p.add_argument(
        "--adaptive-lr-factor", type=float, default=0.5,
        help="Factor applied to all LRs when delta_val_acc < 0 (default 0.5 = halve LR)",
    )

    # ── predict sub-command ──────────────────────────────────────────────────
    pred = subparsers.add_parser("predict", help="Predict bird class for one image")
    pred.add_argument("--image-path", type=Path, required=True)
    pred.add_argument(
        "--checkpoint-path", type=Path,
        default=Path("model_artifacts") / "vgg_bird_classifier.pt",
    )
    pred.add_argument("--image-size", type=int, default=224)

    return parser.parse_args(normalize_cli_args(sys.argv[1:] if argv is None else argv))


def create_model(num_classes: int, unfreeze_layers: int, dropout: float) -> nn.Module:
    """Build a VGG-16 model fine-tuned for bird classification.

    Strategy:
    1. Load the ImageNet-pretrained VGG-16.
    2. Freeze all parameters.
    3. Find the MaxPool2d boundary markers to identify the 5 conv blocks.
    4. Unfreeze the last N blocks so those layers are fine-tuned.
    5. Always unfreeze the full classifier head.
    6. Adjust dropout in the existing classifier and replace the output layer.

    Args:
        num_classes:     Number of bird species to distinguish.
        unfreeze_layers: Number of conv blocks to unfreeze from the end.
        dropout:         Dropout probability applied in the classifier.
    """
    model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)

    # Freeze all params first
    for param in model.parameters():
        param.requires_grad = False

    # VGG16 features has MaxPool2d at indices 4, 9, 16, 23, 30 (5 conv blocks).
    # We identify block boundaries by the pool positions so we can unfreeze
    # whole blocks at a time rather than individual layers.
    pool_indices = [i for i, layer in enumerate(model.features) if isinstance(layer, nn.MaxPool2d)]
    if unfreeze_layers > 0:
        if unfreeze_layers >= len(pool_indices):
            start_idx = 0  # unfreeze all features
        else:
            # Start right after the (N+1)-th-from-last pool layer to get N blocks
            start_idx = pool_indices[-(unfreeze_layers + 1)] + 1
        for i in range(start_idx, len(model.features)):
            for param in model.features[i].parameters():
                param.requires_grad = True

    # Always unfreeze the full classifier head (contains the final Linear we replace)
    for param in model.classifier.parameters():
        param.requires_grad = True

    # Update the existing Dropout layers in the pretrained classifier
    # so they use the user-specified dropout probability
    for module in model.classifier.modules():
        if isinstance(module, nn.Dropout):
            module.p = dropout

    # Swap the final Linear(4096 → 1000 ImageNet) for Linear(4096 → num_classes)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model


def _build_aug_transform() -> transforms.Compose:
    """On-the-fly augmentation applied only during training.

    These transforms act on tensors already in [0,1] float format produced
    by LazyPtDataset, so no ToTensor call is needed here.
    """
    return transforms.Compose([
        transforms.RandomHorizontalFlip(),           # birds appear left and right equally
        transforms.RandomVerticalFlip(p=0.1),        # rare, adds variety
        transforms.RandomRotation(degrees=15),        # slight tilt tolerance
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.2)),  # simulates occlusion
    ])


def run_train(args: argparse.Namespace) -> None:
    """Full training loop for VGG-16.

    Reads .pt class files via LazyPtDataset (one class at a time in RAM to
    stay memory-efficient), trains with differential learning rates, optional
    LR scheduling, and optional adaptive LR when validation accuracy drops.

    Note: VGG-16 has 135 M parameters — keep batch_size low (2–4) to stay
    within 32 GB RAM.
    """
    cfg = VGGConfig(
        pt_data_dir=str(args.pt_data_dir),
        output_dir=str(args.output_dir),
        checkpoint_name=args.checkpoint_name,
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
        adaptive_lr=args.adaptive_lr,
        adaptive_lr_factor=args.adaptive_lr_factor,
    )

    if cfg.image_size <= 0:
        raise ValueError("image-size must be greater than 0")

    pt_dir  = Path(cfg.pt_data_dir)
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Discover .pt split files and build a unified, sorted class list
    train_records = scan_pt_split(pt_dir, "Train", cfg.max_files)
    test_records  = scan_pt_split(pt_dir, "Test",  cfg.max_files)
    classes       = collect_classes(train_records, test_records)
    class_to_idx  = {name: idx for idx, name in enumerate(classes)}

    aug           = _build_aug_transform() if cfg.augment else None
    train_dataset = LazyPtDataset(train_records, class_to_idx, transform=aug)
    test_dataset  = LazyPtDataset(test_records,  class_to_idx)
    train_loader  = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
    test_loader   = DataLoader(test_dataset,  batch_size=cfg.batch_size, shuffle=False)

    print(
        f"VGG-16 | {len(train_records)} train files / {len(test_records)} test files / "
        f"{len(classes)} classes | train={len(train_dataset)} test={len(test_dataset)}"
    )

    model  = create_model(num_classes=len(classes), unfreeze_layers=cfg.unfreeze_layers, dropout=cfg.dropout)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = model.to(device)

    total_p = sum(p.numel() for p in model.parameters())
    train_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {total_p:,} total | {train_p:,} trainable | device: {device}")
    if cfg.adaptive_lr:
        print(f"Adaptive LR: ON  (factor={cfg.adaptive_lr_factor} applied when delta_val_acc < 0)")
    else:
        print("Adaptive LR: OFF (fixed schedule)")

    # Label smoothing distributes a small probability mass to wrong classes,
    # which prevents the model from becoming over-confident.
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)

    # Differential learning rates: conv feature layers get a much lower LR
    # so we don't destroy the rich pretrained representations they hold.
    backbone_params = [p for n, p in model.named_parameters() if p.requires_grad and n.startswith("features")]
    head_params     = [p for n, p in model.named_parameters() if p.requires_grad and n.startswith("classifier")]
    param_groups: list[dict] = []
    if backbone_params:
        param_groups.append({"params": backbone_params, "lr": cfg.learning_rate * cfg.backbone_lr_multiplier})
    if head_params:
        param_groups.append({"params": head_params, "lr": cfg.learning_rate})
    optimizer = torch.optim.AdamW(param_groups, weight_decay=cfg.weight_decay)

    scheduler = None
    if cfg.lr_scheduler == "cosine":
        # Cosine annealing: smoothly decays LR from initial value toward ~0
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    elif cfg.lr_scheduler == "step":
        # Step decay: halve LR every (epochs // 3) steps
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(1, cfg.epochs // 3), gamma=0.5)

    best_acc             = 0.0
    prev_acc             = None   # previous epoch val_acc for computing delta
    best_checkpoint_path = out_dir / f"best_{cfg.checkpoint_name}"

    for epoch in range(cfg.epochs):
        epoch_start = time.perf_counter()

        # ── Training phase ───────────────────────────────────────────────────
        model.train()
        running_loss  = 0.0
        train_correct = 0
        train_total   = 0
        train_start   = time.perf_counter()

        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            # Resize on-the-fly if stored tensor resolution ≠ target image size
            if x_batch.shape[-1] != cfg.image_size:
                x_batch = F.interpolate(x_batch, size=(cfg.image_size, cfg.image_size),
                                        mode="bilinear", align_corners=False)
            optimizer.zero_grad()
            logits = model(x_batch)
            loss   = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            running_loss  += loss.item()
            # Count correct train predictions so we can report training accuracy
            train_correct += (logits.argmax(dim=1) == y_batch).sum().item()
            train_total   += y_batch.size(0)

        train_seconds = time.perf_counter() - train_start

        # ── Validation phase ─────────────────────────────────────────────────
        model.eval()
        correct   = 0
        total     = 0
        val_start = time.perf_counter()
        with torch.no_grad():
            for x_batch, y_batch in test_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                if x_batch.shape[-1] != cfg.image_size:
                    x_batch = F.interpolate(x_batch, size=(cfg.image_size, cfg.image_size),
                                            mode="bilinear", align_corners=False)
                preds    = model(x_batch).argmax(dim=1)
                correct += (preds == y_batch).sum().item()
                total   += y_batch.size(0)
        val_seconds = time.perf_counter() - val_start

        avg_loss  = running_loss / max(len(train_loader), 1)
        train_acc = train_correct / max(train_total, 1)
        val_acc   = correct / max(total, 1)

        # delta_val_acc: positive = improvement, negative = regression
        delta_str = "N/A"
        if prev_acc is not None:
            delta     = val_acc - prev_acc
            delta_str = f"{delta:+.4f}"
            # ── Adaptive LR ─────────────────────────────────────────────────
            # When validation accuracy drops, scale all LR groups down.
            # Smaller steps can help the model navigate out of a plateau or
            # prevent overshooting a good minimum it has found.
            if cfg.adaptive_lr and delta < 0:
                for pg in optimizer.param_groups:
                    pg["lr"] *= cfg.adaptive_lr_factor
                current_lrs = [f"{pg['lr']:.2e}" for pg in optimizer.param_groups]
                print(f"  [adaptive LR] val_acc dropped → LR scaled ×{cfg.adaptive_lr_factor}: {current_lrs}")
        prev_acc = val_acc

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(
                {"model_state_dict": model.state_dict(), "classes": classes,
                 "config": asdict(cfg), "model_name": "vgg16"},
                best_checkpoint_path,
            )

        if scheduler is not None:
            scheduler.step()

        epoch_seconds = time.perf_counter() - epoch_start
        print(
            f"Epoch {epoch + 1}/{cfg.epochs} - loss: {avg_loss:.4f} "
            f"- train_acc: {train_acc:.4f} - val_acc: {val_acc:.4f} "
            f"- delta_val_acc: {delta_str} [best: {best_acc:.4f}] "
            f"- time: train={train_seconds:.1f}s val={val_seconds:.1f}s total={epoch_seconds:.1f}s"
        )

    # Save the final-epoch checkpoint (distinct from the best-epoch checkpoint above)
    checkpoint_path = out_dir / cfg.checkpoint_name
    torch.save(
        {"model_state_dict": model.state_dict(), "classes": classes,
         "config": asdict(cfg), "model_name": "vgg16"},
        checkpoint_path,
    )
    config_path = out_dir / "vgg_config.json"
    config_path.write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")
    print(f"\nBest val_acc: {best_acc:.4f}  →  {best_checkpoint_path}")
    print(f"Final checkpoint: {checkpoint_path}")
    print(f"Config: {config_path}")


def run_predict(args: argparse.Namespace) -> None:
    """Load a VGG checkpoint and classify a single bird image.

    The checkpoint stores the class list and config so no extra flags are
    needed beyond the image path.
    """
    checkpoint = torch.load(args.checkpoint_path, map_location="cpu")
    classes: list[str] = checkpoint["classes"]
    cfg_dict = checkpoint.get("config", {})
    dropout  = float(cfg_dict.get("dropout", 0.5))

    # Reconstruct model and load saved weights
    model = create_model(num_classes=len(classes), unfreeze_layers=0, dropout=dropout)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])

    with Image.open(args.image_path) as image:
        x_tensor = transform(image.convert("RGB")).unsqueeze(0)  # add batch dim

    with torch.no_grad():
        probs   = torch.softmax(model(x_tensor), dim=1)
    top_idx  = int(probs.argmax(dim=1).item())
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

