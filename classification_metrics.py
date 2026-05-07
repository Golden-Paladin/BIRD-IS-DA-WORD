from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

import torch


@dataclass(frozen=True)
class ClassificationMetrics:
    """Summary metrics for single-label multi-class classification.

    The project has many bird species and potentially uneven class support, so we
    compute weighted averages by default. Weighted metrics account for class
    imbalance by weighting each class contribution by its validation support.
    """

    accuracy: float
    precision_weighted: float
    recall_weighted: float
    f1_weighted: float
    precision_macro: float
    recall_macro: float
    f1_macro: float
    support: int
    averaging: str = "weighted"

    def to_dict(self) -> dict[str, float | int | str]:
        """Return a JSON/checkpoint-friendly representation."""
        return asdict(self)


def compute_classification_metrics(
    targets: Sequence[int],
    predictions: Sequence[int],
    num_classes: int,
) -> ClassificationMetrics:
    """Compute accuracy, weighted precision/recall/F1, and macro variants.

    Args:
        targets: Ground-truth class indices.
        predictions: Predicted class indices.
        num_classes: Total number of classes in the problem.

    Returns:
        A ClassificationMetrics dataclass containing aggregate metrics.

    Raises:
        ValueError: If the targets/predictions lengths differ or num_classes < 1.
    """
    if num_classes < 1:
        raise ValueError("num_classes must be at least 1")
    if len(targets) != len(predictions):
        raise ValueError("targets and predictions must have the same length")

    total_samples = len(targets)
    if total_samples == 0:
        return ClassificationMetrics(
            accuracy=0.0,
            precision_weighted=0.0,
            recall_weighted=0.0,
            f1_weighted=0.0,
            precision_macro=0.0,
            recall_macro=0.0,
            f1_macro=0.0,
            support=0,
        )

    confusion = torch.zeros((num_classes, num_classes), dtype=torch.float64)
    for truth, pred in zip(targets, predictions):
        if truth < 0 or truth >= num_classes:
            raise ValueError(f"target index {truth} is outside [0, {num_classes})")
        if pred < 0 or pred >= num_classes:
            raise ValueError(f"prediction index {pred} is outside [0, {num_classes})")
        confusion[truth, pred] += 1.0

    true_positives = confusion.diag()
    support_per_class = confusion.sum(dim=1)
    predicted_per_class = confusion.sum(dim=0)

    precision_per_class = torch.where(
        predicted_per_class > 0,
        true_positives / predicted_per_class,
        torch.zeros_like(true_positives),
    )
    recall_per_class = torch.where(
        support_per_class > 0,
        true_positives / support_per_class,
        torch.zeros_like(true_positives),
    )
    f1_per_class = torch.where(
        (precision_per_class + recall_per_class) > 0,
        2.0 * precision_per_class * recall_per_class / (precision_per_class + recall_per_class),
        torch.zeros_like(true_positives),
    )

    weights = support_per_class / max(float(support_per_class.sum().item()), 1.0)
    accuracy = float(true_positives.sum().item() / total_samples)

    return ClassificationMetrics(
        accuracy=accuracy,
        precision_weighted=float((precision_per_class * weights).sum().item()),
        recall_weighted=float((recall_per_class * weights).sum().item()),
        f1_weighted=float((f1_per_class * weights).sum().item()),
        precision_macro=float(precision_per_class.mean().item()),
        recall_macro=float(recall_per_class.mean().item()),
        f1_macro=float(f1_per_class.mean().item()),
        support=total_samples,
    )

