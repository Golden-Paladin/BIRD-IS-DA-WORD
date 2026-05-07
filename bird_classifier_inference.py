from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import torch
from PIL import Image
from torchvision import transforms
# All three torchvision backbones were fine-tuned from ImageNet checkpoints, so
# inference must use the same mean/std normalization the training scripts expect.
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
def safe_torch_load(path: Path) -> dict:
    """Load a checkpoint while staying compatible with older torch versions."""
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")
@dataclass
class LoadedBirdClassifier:
    """Fully reconstructed classifier checkpoint ready for inference."""
    model: torch.nn.Module
    classes: list[str]
    image_size: int
    device: torch.device
    model_name: str
def _build_model_from_checkpoint(checkpoint: dict) -> tuple[torch.nn.Module, list[str], int, str]:
    """Recreate the correct model architecture based on checkpoint metadata."""
    classes: list[str] = list(checkpoint["classes"])
    cfg_dict = checkpoint.get("config", {})
    model_name = str(checkpoint.get("model_name", ""))
    if model_name.startswith("efficientnet") or "model_variant" in cfg_dict:
        from efficientnet_model import create_model as create_efficientnet_model
        variant = str(cfg_dict.get("model_variant", "b2"))
        dropout = float(cfg_dict.get("dropout", 0.3))
        image_size = int(cfg_dict.get("image_size", 224))
        model = create_efficientnet_model(
            num_classes=len(classes),
            variant=variant,
            unfreeze_layers=0,
            dropout=dropout,
        )
    elif model_name.startswith("resnet"):
        from resnet_model import create_model_with_attention
        dropout = float(cfg_dict.get("dropout", 0.3))
        image_size = int(cfg_dict.get("image_size", 224))
        model = create_model_with_attention(
            num_classes=len(classes),
            unfreeze_layers=0,
            dropout=dropout,
        )
    elif model_name.startswith("vgg"):
        from vgg_model import create_model as create_vgg_model
        dropout = float(cfg_dict.get("dropout", 0.5))
        image_size = int(cfg_dict.get("image_size", 224))
        model = create_vgg_model(
            num_classes=len(classes),
            unfreeze_layers=0,
            dropout=dropout,
        )
    else:
        raise ValueError(
            "Unsupported checkpoint architecture. Expected an EfficientNet, ResNet, or VGG checkpoint, "
            f"but got model_name={model_name!r}."
        )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, classes, image_size, model_name
def load_bird_classifier(
    checkpoint_path: str | Path,
    device: str | torch.device | None = None,
) -> LoadedBirdClassifier:
    """Load a saved base-model checkpoint for inference from images or crops."""
    checkpoint_path = Path(checkpoint_path)
    checkpoint = safe_torch_load(checkpoint_path)
    model, classes, image_size, model_name = _build_model_from_checkpoint(checkpoint)
    resolved_device = torch.device(device) if device is not None else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    model = model.to(resolved_device)
    model.eval()
    return LoadedBirdClassifier(
        model=model,
        classes=classes,
        image_size=image_size,
        device=resolved_device,
        model_name=model_name,
    )
def build_inference_transform(image_size: int) -> transforms.Compose:
    """Create the normalization pipeline shared by all base classifiers."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])
def predict_bird_from_pil(
    image: Image.Image,
    classifier: LoadedBirdClassifier,
    image_size: int | None = None,
) -> tuple[str, float, int]:
    """Classify one RGB image/crop and return label, confidence, and class index."""
    target_size = int(image_size or classifier.image_size)
    transform = build_inference_transform(target_size)
    x_tensor = transform(image.convert("RGB")).unsqueeze(0).to(classifier.device)
    with torch.no_grad():
        probabilities = torch.softmax(classifier.model(x_tensor), dim=1)
    top_index = int(probabilities.argmax(dim=1).item())
    top_confidence = float(probabilities[0, top_index].item())
    return classifier.classes[top_index], top_confidence, top_index
