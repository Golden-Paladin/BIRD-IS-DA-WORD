from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib

# Use a non-interactive backend so plot generation works in headless environments.
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _to_error_points(acc_values: Iterable[float | None]) -> list[float | None]:
    """Convert accuracy values into error values (error = 1 - accuracy)."""
    points: list[float | None] = []
    for value in acc_values:
        if value is None:
            points.append(None)
        else:
            points.append(1.0 - float(value))
    return points


def save_error_curve_png(
    output_path: Path,
    train_acc_history: list[float | None],
    val_acc_history: list[float | None],
    title: str,
) -> Path:
    """Save a PNG plot of training/validation error versus epoch.

    Args:
        output_path: Full PNG path to write.
        train_acc_history: Per-epoch training accuracy.
        val_acc_history: Per-epoch validation accuracy.
        title: Chart title.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    train_error = _to_error_points(train_acc_history)
    val_error = _to_error_points(val_acc_history)
    max_len = max(len(train_error), len(val_error))
    epochs = list(range(1, max_len + 1))

    plt.figure(figsize=(9, 5))
    if train_error:
        plt.plot(epochs[: len(train_error)], train_error, label="train_error", marker="o")
    if val_error:
        plt.plot(epochs[: len(val_error)], val_error, label="val_error", marker="o")

    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Error (1 - accuracy)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=140)
    plt.close()
    return output_path

