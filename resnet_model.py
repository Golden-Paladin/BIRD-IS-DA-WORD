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

# ImageNet normalisation constants — the pretrained backbone expects these.
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]


@dataclass
class ResNetConfig:
    """All hyper-parameters and paths for one training run.

    Serialised to JSON alongside every checkpoint so results are reproducible.
    """
    pt_data_dir: str = "pt_data"
    output_dir: str = "model_artifacts"
    checkpoint_name: str = "resnet_bird_classifier.pt"
    image_size: int = 224
    batch_size: int = 4
    epochs: int = 10
    learning_rate: float = 5e-4
    # 0=all frozen; 1=unfreeze layer4+fc; 2=+layer3; 3=+layer2; 4=all backbone
    unfreeze_layers: int = 1
    # backbone unfrozen layers use lr * this multiplier (lower = more careful fine-tuning)
    backbone_lr_multiplier: float = 0.1
    lr_scheduler: str = "cosine"   # cosine | step | none
    label_smoothing: float = 0.1
    weight_decay: float = 1e-4
    dropout: float = 0.3
    augment: bool = True
    max_files: int | None = None
    # --- adaptive LR ---
    # When True, all LR groups are multiplied by adaptive_lr_factor whenever
    # val_acc decreases compared to the previous epoch (delta_val_acc < 0).
    adaptive_lr: bool = False
    adaptive_lr_factor: float = 0.5   # how aggressively to reduce LR on a bad epoch


def normalize_cli_args(argv: list[str]) -> list[str]:
    """Silently prepend 'train' when the user omits the sub-command.

    Allows `python resnet_model.py --epochs 5` in addition to the
    canonical `python resnet_model.py train --epochs 5`.
    """
    if not argv:
        return ["train"]
    if argv[0] in {"train", "predict", "-h", "--help"}:
        return argv
    return ["train", *argv]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Build the full CLI with 'train' and 'predict' sub-commands."""
    parser = argparse.ArgumentParser(description="Train or run inference with a ResNet bird classifier.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # ── train sub-command ────────────────────────────────────────────────────
    p = subparsers.add_parser("train", help="Train from generated .pt files")
    p.add_argument("--pt-data-dir", type=Path, default=Path("pt_data"))
    p.add_argument("--output-dir", type=Path, default=Path("model_artifacts"))
    p.add_argument("--checkpoint-name", default="resnet_bird_classifier.pt")
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--learning-rate", type=float, default=5e-4)
    p.add_argument(
        "--unfreeze-layers", type=int, default=1,
        help="ResNet blocks to unfreeze from top (0=all frozen, 1=layer4+fc, 2=+layer3, 3=+layer2, 4=all backbone)",
    )
    p.add_argument("--backbone-lr-multiplier", type=float, default=0.1,
                   help="LR multiplier applied to unfrozen backbone layers (default 0.1 = 10x lower than head)")
    p.add_argument("--lr-scheduler", default="cosine", choices=["cosine", "step", "none"],
                   help="cosine: smooth decay; step: halve every 1/3 epochs; none: fixed LR")
    p.add_argument("--label-smoothing", type=float, default=0.1,
                   help="Softens targets — prevents overconfidence on 200 nearly-identical classes")
    p.add_argument("--weight-decay", type=float, default=1e-4,
                   help="AdamW L2 weight penalty")
    p.add_argument("--dropout", type=float, default=0.3, help="Dropout rate before final FC layer")
    p.add_argument("--augment", action=argparse.BooleanOptionalAction, default=True,
                   help="Enable on-the-fly random augmentation (flip, rotate, erase)")
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
        help="Factor by which all LRs are scaled when delta_val_acc < 0 (default 0.5 = halve)",
    )

    # ── predict sub-command ──────────────────────────────────────────────────
    pred = subparsers.add_parser("predict", help="Predict bird class for one image")
    pred.add_argument("--image-path", type=Path, required=True)
    pred.add_argument(
        "--checkpoint-path", type=Path,
        default=Path("model_artifacts") / "resnet_bird_classifier.pt",
    )
    pred.add_argument("--image-size", type=int, default=None,
                      help="Optional override. Defaults to training image_size from checkpoint.")

    return parser.parse_args(normalize_cli_args(sys.argv[1:] if argv is None else argv))

class SEBlock(nn.Module):
    def __init__ (self, channels, r=16):
        super(SEBlock, self).__init__()
        # Squeeze Layer to Capture Global Channel
        # This will be responsible as the 'whole' of the image
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        # The excitation will take each small part of the image
        # and check how much it matters in the context of global
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // r, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.squeeze(x).view(b, c)

        y = self.excitation(y).view(b, c, 1, 1)

        return x * y.expand_as(x)

def create_model(num_classes: int, unfreeze_layers: int, dropout: float) -> nn.Module:
    """Build a ResNet-50 fine-tuned for bird classification.

    Strategy:
    1. Load ImageNet-pretrained ResNet-50.
    2. Freeze all parameters.
    3. Unfreeze the last N residual-layer groups for fine-tuning.
    4. Replace the fully-connected head with Dropout + Linear.

    Args:
        num_classes:     Number of bird species.
        unfreeze_layers: 0=all frozen, 1=layer4, 2=layer4+layer3, 3=+layer2, 4=all.
        dropout:         Dropout probability before the final linear layer.
    """
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

    # Freeze everything first
    for param in model.parameters():
        param.requires_grad = False

    # Selectively unfreeze the last N residual blocks (counts from layer4 backwards)
    resnet_layers = [model.layer1, model.layer2, model.layer3, model.layer4]
    for i in range(min(unfreeze_layers, len(resnet_layers))):
        for param in resnet_layers[-(i + 1)].parameters():
            param.requires_grad = True

    # Replace FC head with Dropout + Linear (always trainable as new params)
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(num_features, num_classes),
    )
    return model


def create_model_with_attention(num_classes: int, unfreeze_layers: int, dropout: float) -> nn.Module:
    """Build a ResNet-50 fine-tuned for bird classification.

    Strategy:
    1. Load ImageNet-pretrained ResNet-50.
    2. Freeze all parameters.
    3. Unfreeze the last N residual-layer groups for fine-tuning.
    4. Replace the fully-connected head with Dropout + Linear.

    Args:
        num_classes:     Number of bird species.
        unfreeze_layers: 0=all frozen, 1=layer4, 2=layer4+layer3, 3=+layer2, 4=all.
        dropout:         Dropout probability before the final linear layer.
    """
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

    # Freeze everything first
    for param in model.parameters():
        param.requires_grad = False

    # Selectively unfreeze the last N residual blocks (counts from layer4 backwards)
    resnet_layers = [model.layer1, model.layer2, model.layer3, model.layer4]
    resnet_layer_sizes = [256, 512, 1024, 2048]

    for s, l in zip(resnet_layer_sizes, resnet_layers):
        layer_container = getattr(model, l)
        for i in range(len(layer_container)):
            original_block = layer_container[i]

            layer_container[i] = nn.Sequential(
                original_block,
                SEBlock(channels=s)
            )

    for i in range(min(unfreeze_layers, len(resnet_layers))):
        for param in resnet_layers[-(i + 1)].parameters():
            param.requires_grad = True

    # Replace FC head with Dropout + Linear (always trainable as new params)
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(num_features, num_classes),
    )
    return model

def _build_aug_transform() -> transforms.Compose:
    """Return the augmentation pipeline used during training only.

    These transforms are applied to tensors already in [0,1] float format,
    so no ToTensor step is needed here.
    """
    return transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(p=0.1),
        transforms.RandomRotation(degrees=15),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.2)),
    ])


def _safe_torch_load(path: Path) -> dict:
    """Load a .pt checkpoint while staying compatible with older torch versions.

    PyTorch >= 2.0 requires `weights_only` to be explicit; older versions
    don't accept that argument at all, so we try both.
    """
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _checkpoint_name_with_traits(cfg: ResNetConfig) -> str:
    """Build a descriptive filename that encodes the key hyper-parameters.

    Example: resnet_bird_classifier_u10_ep10_bs32_img224.pt
    Makes it easy to tell checkpoints apart without opening them.
    """
    src    = Path(cfg.checkpoint_name)
    stem   = src.stem
    suffix = src.suffix or ".pt"
    return f"{stem}_u{cfg.unfreeze_layers}_ep{cfg.epochs}_bs{cfg.batch_size}_img{cfg.image_size}{suffix}"


def run_train(args: argparse.Namespace) -> None:
    """Full training loop for ResNet-50.

    Loads .pt class files via LazyPtDataset (one class at a time in RAM),
    trains with differential learning rates, optional LR scheduling, and
    optional adaptive LR reduction when validation accuracy drops.
    """
    cfg = ResNetConfig(
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

    # Embed key hyper-parameters in the checkpoint filename for easy identification
    cfg.checkpoint_name = _checkpoint_name_with_traits(cfg)

    pt_dir  = Path(cfg.pt_data_dir)
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

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
        f"ResNet-50 | {len(train_records)} train files / {len(test_records)} test files / "
        f"{len(classes)} classes | train={len(train_dataset)} test={len(test_dataset)}"
    )
    print(
        f"Config: image_size={cfg.image_size}, batch_size={cfg.batch_size}, "
        f"unfreeze_layers={cfg.unfreeze_layers}, epochs={cfg.epochs}, lr={cfg.learning_rate}"
    )

    model  = create_model_with_attention(num_classes=len(classes), unfreeze_layers=cfg.unfreeze_layers, dropout=cfg.dropout)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = model.to(device)

    total_p = sum(p.numel() for p in model.parameters())
    train_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {total_p:,} total | {train_p:,} trainable | device: {device}")
    if cfg.adaptive_lr:
        print(f"Adaptive LR: ON  (factor={cfg.adaptive_lr_factor} applied when delta_val_acc < 0)")
    else:
        print("Adaptive LR: OFF (fixed schedule)")

    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)

    # Differential learning rates: backbone gets a much smaller lr to avoid forgetting
    backbone_params = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith("fc")]
    head_params     = [p for n, p in model.named_parameters() if p.requires_grad and n.startswith("fc")]
    param_groups: list[dict] = []
    if backbone_params:
        param_groups.append({"params": backbone_params, "lr": cfg.learning_rate * cfg.backbone_lr_multiplier})
    if head_params:
        param_groups.append({"params": head_params, "lr": cfg.learning_rate})
    optimizer = torch.optim.AdamW(param_groups, weight_decay=cfg.weight_decay)

    scheduler = None
    if cfg.lr_scheduler == "cosine":
        # CosineAnnealingLR: smoothly reduces LR from initial value to ~0
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    elif cfg.lr_scheduler == "step":
        # StepLR: halves every (epochs // 3) epochs — more abrupt step-down
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(1, cfg.epochs // 3), gamma=0.5)

    best_acc             = 0.0
    prev_acc             = None   # track previous epoch val_acc for delta computation
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
            # Resize on-the-fly if .pt tensor resolution differs from target
            if x_batch.shape[-1] != cfg.image_size:
                x_batch = F.interpolate(x_batch, size=(cfg.image_size, cfg.image_size),
                                        mode="bilinear", align_corners=False)
            optimizer.zero_grad()
            logits = model(x_batch)
            loss   = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            running_loss  += loss.item()
            # Count correct training predictions for reporting train accuracy
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

        # delta_val_acc: positive = improving, negative = degrading
        delta_str = "N/A"
        if prev_acc is not None:
            delta     = val_acc - prev_acc
            delta_str = f"{delta:+.4f}"
            # ── Adaptive LR ─────────────────────────────────────────────────
            # Reduce all LR groups when validation accuracy has dropped.
            # Helps the model escape plateaus by taking smaller gradient steps.
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
                 "config": asdict(cfg), "model_name": "resnet50"},
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

    checkpoint_path = out_dir / cfg.checkpoint_name
    torch.save(
        {"model_state_dict": model.state_dict(), "classes": classes,
         "config": asdict(cfg), "model_name": "resnet50"},
        checkpoint_path,
    )
    config_path = out_dir / "resnet_config.json"
    config_path.write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")
    print(f"\nBest val_acc: {best_acc:.4f}  →  {best_checkpoint_path}")
    print(f"Final checkpoint: {checkpoint_path}")
    print(f"Config: {config_path}")


def run_predict(args: argparse.Namespace) -> None:
    """Load a saved ResNet checkpoint and classify a single bird image.

    All necessary config (dropout, image size, class list) is embedded inside
    the checkpoint so no extra flags are required.
    """
    checkpoint = _safe_torch_load(args.checkpoint_path)
    classes: list[str] = checkpoint["classes"]
    cfg_dict   = checkpoint.get("config", {})
    dropout    = float(cfg_dict.get("dropout", 0.3))
    image_size = int(args.image_size or cfg_dict.get("image_size", 224))

    # Reconstruct model architecture and load saved weights
    model = create_model_with_attention(num_classes=len(classes), unfreeze_layers=0, dropout=dropout)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
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

