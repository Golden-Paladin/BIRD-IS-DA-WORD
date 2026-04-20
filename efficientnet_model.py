from __future__ import annotations

import argparse
import json
import sys
import time                          # used to measure per-epoch wall-clock time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import models, transforms

from pt_streaming import LazyPtDataset, collect_classes, scan_pt_split

# ImageNet normalisation constants — required because the pretrained backbone
# was trained on images normalised with these exact mean/std values.
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

# Each variant maps a short name to (factory_function, pretrained_weights).
# B2 is the default because it gives the best accuracy/memory trade-off for
# this dataset (~9 M params, competes with ResNet-50 at 23 M).
_VARIANTS: dict[str, tuple] = {
    "b0": (models.efficientnet_b0, models.EfficientNet_B0_Weights.IMAGENET1K_V1),   #  5.3M params
    "b1": (models.efficientnet_b1, models.EfficientNet_B1_Weights.IMAGENET1K_V1),   #  7.8M params
    "b2": (models.efficientnet_b2, models.EfficientNet_B2_Weights.IMAGENET1K_V1),   #  9.1M params  ← default
    "b3": (models.efficientnet_b3, models.EfficientNet_B3_Weights.IMAGENET1K_V1),   # 12.2M params
}


@dataclass
class EfficientNetConfig:
    """All hyper-parameters and paths for one training run.

    Stored as JSON in model_artifacts/ so every checkpoint is self-documenting.
    """
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
    # --- adaptive LR ---
    # When True, the learning rate is reduced by adaptive_lr_factor any time
    # val_acc drops relative to the previous epoch (delta_val_acc < 0).
    adaptive_lr: bool = False
    adaptive_lr_factor: float = 0.5    # multiply all LR groups by this on a bad epoch


def normalize_cli_args(argv: list[str]) -> list[str]:
    """Allow calling the script without an explicit sub-command.

    If the user writes `python efficientnet_model.py --epochs 5` we
    transparently prepend 'train' so argparse is happy.
    """
    if not argv:
        return ["train"]
    if argv[0] in {"train", "predict", "-h", "--help"}:
        return argv
    return ["train", *argv]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Build the CLI parser for both 'train' and 'predict' sub-commands."""
    parser = argparse.ArgumentParser(description="Train or run inference with an EfficientNet bird classifier.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # ── train sub-command ────────────────────────────────────────────────────
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
    p.add_argument("--backbone-lr-multiplier", type=float, default=0.1,
                   help="Backbone LR = head LR × this value (keeps pretrained features from being overwritten)")
    p.add_argument("--lr-scheduler", default="cosine", choices=["cosine", "step", "none"],
                   help="LR schedule: cosine annealing, step decay every 1/3 of epochs, or none")
    p.add_argument("--label-smoothing", type=float, default=0.1,
                   help="Cross-entropy label smoothing — prevents over-confidence on 200 classes")
    p.add_argument("--weight-decay", type=float, default=1e-4,
                   help="AdamW L2 regularisation coefficient")
    p.add_argument("--dropout", type=float, default=0.3, help="Dropout before the classifier linear layer")
    p.add_argument("--augment", action=argparse.BooleanOptionalAction, default=True,
                   help="Random flip / rotate / erase applied every epoch")
    p.add_argument("--max-files", type=int, default=None, help="Optional cap for quick smoke tests")
    # adaptive LR flags
    p.add_argument(
        "--adaptive-lr", action=argparse.BooleanOptionalAction, default=False,
        help=(
            "Enable adaptive learning-rate reduction. "
            "When val_acc drops vs the previous epoch the LR of every "
            "parameter group is multiplied by --adaptive-lr-factor."
        ),
    )
    p.add_argument(
        "--adaptive-lr-factor", type=float, default=0.5,
        help="Multiplicative factor applied to all LRs when val_acc delta is negative (default 0.5 = halve LR)",
    )

    # ── predict sub-command ──────────────────────────────────────────────────
    pred = subparsers.add_parser("predict", help="Predict bird class for one image")
    pred.add_argument("--image-path", type=Path, required=True)
    pred.add_argument(
        "--checkpoint-path", type=Path,
        default=Path("model_artifacts") / "efficientnet_bird_classifier.pt",
    )
    pred.add_argument("--image-size", type=int, default=224)

    return parser.parse_args(normalize_cli_args(sys.argv[1:] if argv is None else argv))


def create_model(num_classes: int, variant: str, unfreeze_layers: int, dropout: float) -> nn.Module:
    """Build an EfficientNet-Bx with a custom classification head.

    Strategy:
    1. Load the pretrained ImageNet weights.
    2. Freeze *all* parameters so the backbone acts as a pure feature extractor.
    3. Selectively unfreeze the last `unfreeze_layers` feature blocks for
       fine-tuning (higher number = more capacity, more RAM, longer training).
    4. Replace the classifier head with Dropout + Linear sized for our bird classes.

    Args:
        num_classes:     Number of bird species to classify.
        variant:         One of 'b0', 'b1', 'b2', 'b3'.
        unfreeze_layers: How many feature blocks to unfreeze from the end.
        dropout:         Dropout probability before the final linear layer.
    """
    if variant not in _VARIANTS:
        raise ValueError(f"Unknown variant '{variant}'. Choose from: {list(_VARIANTS)}")
    model_fn, weights = _VARIANTS[variant]
    model = model_fn(weights=weights)

    # Freeze everything first — we only want to train selected layers
    for param in model.parameters():
        param.requires_grad = False

    # EfficientNet-Bx has model.features as a Sequential of N blocks.
    # Unfreeze the last `unfreeze_layers` blocks (counting from the end).
    # E.g. unfreeze_layers=5 means the 5 deepest feature blocks are trainable.
    feature_blocks = list(model.features.children())
    for i in range(min(unfreeze_layers, len(feature_blocks))):
        for param in feature_blocks[-(i + 1)].parameters():
            param.requires_grad = True

    # Replace classifier head (always trainable — these are brand-new weights)
    in_features = model.classifier[-1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout, inplace=True),
        nn.Linear(in_features, num_classes),
    )
    return model


def _build_aug_transform() -> transforms.Compose:
    """On-the-fly augmentation applied during training (not at inference time).

    Applied *after* the tensor is already in [0,1] float format by
    LazyPtDataset, so we only use pixel-level transforms (no ToTensor).
    """
    return transforms.Compose([
        transforms.RandomHorizontalFlip(),           # birds face left/right equally
        transforms.RandomVerticalFlip(p=0.1),        # rare but adds robustness
        transforms.RandomRotation(degrees=15),        # slight tilt invariance
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.2)),  # occlusion simulation
    ])


def run_train(args: argparse.Namespace) -> None:
    """Full training loop for EfficientNet.

    Reads pre-processed .pt class files, builds data loaders, constructs the
    model, runs the training loop, and saves checkpoints + config JSON.
    """
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
        adaptive_lr=args.adaptive_lr,
        adaptive_lr_factor=args.adaptive_lr_factor,
    )

    if cfg.image_size <= 0:
        raise ValueError("image-size must be greater than 0")

    pt_dir = Path(cfg.pt_data_dir)
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Discover all .pt split files and derive the unified class list
    train_records = scan_pt_split(pt_dir, "Train", cfg.max_files)
    test_records  = scan_pt_split(pt_dir, "Test",  cfg.max_files)
    classes       = collect_classes(train_records, test_records)
    class_to_idx  = {name: idx for idx, name in enumerate(classes)}

    # LazyPtDataset loads one class file at a time — keeps RAM usage flat
    aug           = _build_aug_transform() if cfg.augment else None
    train_dataset = LazyPtDataset(train_records, class_to_idx, transform=aug)
    test_dataset  = LazyPtDataset(test_records,  class_to_idx)
    train_loader  = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
    test_loader   = DataLoader(test_dataset,  batch_size=cfg.batch_size, shuffle=False)

    print(
        f"EfficientNet-{cfg.model_variant.upper()} | "
        f"{len(train_records)} train files / {len(test_records)} test files / "
        f"{len(classes)} classes | train={len(train_dataset)} test={len(test_dataset)}"
    )

    model  = create_model(
        num_classes=len(classes),
        variant=cfg.model_variant,
        unfreeze_layers=cfg.unfreeze_layers,
        dropout=cfg.dropout,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = model.to(device)

    total_p = sum(p.numel() for p in model.parameters())
    train_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {total_p:,} total | {train_p:,} trainable | device: {device}")
    if cfg.adaptive_lr:
        print(f"Adaptive LR: ON  (factor={cfg.adaptive_lr_factor} applied when delta_val_acc < 0)")
    else:
        print("Adaptive LR: OFF (fixed schedule)")

    # Label smoothing prevents the model from becoming overconfident — useful
    # because we have 200 very similar-looking classes.
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)

    # Differential learning rates: backbone (pretrained) gets a much smaller LR
    # than the new head so we don't destroy the pretrained representations.
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

    # Static LR schedule (runs alongside or instead of adaptive LR)
    scheduler = None
    if cfg.lr_scheduler == "cosine":
        # Smoothly decays LR from initial to ~0 over all epochs
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    elif cfg.lr_scheduler == "step":
        # Halves LR every (epochs // 3) steps — more abrupt than cosine
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(1, cfg.epochs // 3), gamma=0.5)

    best_acc  = 0.0
    prev_acc  = None   # used to compute delta_val_acc each epoch
    best_checkpoint_path = out_dir / f"best_{cfg.checkpoint_name}"
    model_name = f"efficientnet_{cfg.model_variant}"

    for epoch in range(cfg.epochs):
        epoch_start = time.perf_counter()

        # ── Training phase ───────────────────────────────────────────────────
        model.train()
        running_loss   = 0.0
        train_correct  = 0
        train_total    = 0
        train_start    = time.perf_counter()

        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            # Resize on-the-fly if the stored tensor size differs from the target
            if x_batch.shape[-1] != cfg.image_size:
                x_batch = F.interpolate(x_batch, size=(cfg.image_size, cfg.image_size),
                                        mode="bilinear", align_corners=False)
            optimizer.zero_grad()
            logits = model(x_batch)
            loss   = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            running_loss  += loss.item()
            # Accumulate training-set correct predictions for train accuracy
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

        avg_loss   = running_loss / max(len(train_loader), 1)
        train_acc  = train_correct / max(train_total, 1)
        val_acc    = correct / max(total, 1)

        # Delta is the change in validation accuracy vs the previous epoch.
        # A negative delta means the model got worse — used by adaptive LR.
        delta_str = "N/A"
        if prev_acc is not None:
            delta     = val_acc - prev_acc
            delta_str = f"{delta:+.4f}"
            # ── Adaptive LR ─────────────────────────────────────────────────
            # If enabled and val_acc has gone down, scale all LR groups down.
            # This lets the model try smaller gradient steps to escape a local
            # plateau without manually tuning a schedule.
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
                 "config": asdict(cfg), "model_name": model_name},
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

    # Save final-epoch checkpoint (may not be the best epoch)
    checkpoint_path = out_dir / cfg.checkpoint_name
    torch.save(
        {"model_state_dict": model.state_dict(), "classes": classes,
         "config": asdict(cfg), "model_name": model_name},
        checkpoint_path,
    )
    config_path = out_dir / "efficientnet_config.json"
    config_path.write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")
    print(f"\nBest val_acc: {best_acc:.4f}  →  {best_checkpoint_path}")
    print(f"Final checkpoint: {checkpoint_path}")
    print(f"Config: {config_path}")


def run_predict(args: argparse.Namespace) -> None:
    """Load a saved checkpoint and predict the bird species in one image.

    The checkpoint embeds all config (variant, dropout, class list) so you
    don't need to pass any extra flags — just the image path.
    """
    checkpoint = torch.load(args.checkpoint_path, map_location="cpu")
    classes: list[str] = checkpoint["classes"]
    cfg_dict = checkpoint.get("config", {})
    variant  = str(cfg_dict.get("model_variant", "b2"))
    dropout  = float(cfg_dict.get("dropout", 0.3))

    # Rebuild network architecture then load saved weights
    model = create_model(num_classes=len(classes), variant=variant, unfreeze_layers=0, dropout=dropout)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])

    with Image.open(args.image_path) as image:
        x_tensor = transform(image.convert("RGB")).unsqueeze(0)  # add batch dimension

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

