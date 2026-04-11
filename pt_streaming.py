from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class PtFileRecord:
    path: Path
    class_name: str
    num_samples: int


def _infer_class_name(file_path: Path, split_name: str) -> str:
    suffix = f"_{split_name}"
    stem = file_path.stem
    if stem.endswith(suffix):
        return stem[: -len(suffix)]
    return stem


def scan_pt_split(pt_dir: Path, split_name: str, max_files: int | None = None) -> list[PtFileRecord]:
    files = sorted(pt_dir.glob(f"*_{split_name}.pt"))
    if max_files is not None:
        files = files[:max_files]

    if not files:
        raise FileNotFoundError(f"No files found for split '{split_name}' in {pt_dir}")

    records: list[PtFileRecord] = []
    for file_path in files:
        payload = torch.load(file_path, map_location="cpu")
        x_data = payload["X"]
        if x_data.ndim != 4:
            raise ValueError(f"Expected a 4D tensor in {file_path}, but got shape {tuple(x_data.shape)}")

        num_samples = int(x_data.shape[0])
        if num_samples == 0:
            continue

        class_name = str(payload.get("class_name") or _infer_class_name(file_path, split_name))
        records.append(PtFileRecord(path=file_path, class_name=class_name, num_samples=num_samples))

    if not records:
        raise ValueError(f"No usable samples were found for split '{split_name}' in {pt_dir}")

    return records


def collect_classes(*record_groups: list[PtFileRecord]) -> list[str]:
    return sorted({record.class_name for group in record_groups for record in group})


class LazyPtDataset(Dataset[tuple[torch.Tensor, int]]):
    def __init__(self, records: list[PtFileRecord], class_to_idx: dict[str, int]) -> None:
        if not records:
            raise ValueError("records cannot be empty")

        self.records = records
        self.class_to_idx = class_to_idx
        self._cumulative_sizes: list[int] = []

        running_total = 0
        for record in records:
            running_total += record.num_samples
            self._cumulative_sizes.append(running_total)

        self._cached_path: Path | None = None
        self._cached_tensor: torch.Tensor | None = None

    def __len__(self) -> int:
        return self._cumulative_sizes[-1]

    def _load_tensor(self, file_path: Path) -> torch.Tensor:
        if self._cached_path != file_path or self._cached_tensor is None:
            payload = torch.load(file_path, map_location="cpu")
            self._cached_tensor = payload["X"].float()
            self._cached_path = file_path
        cached_tensor = self._cached_tensor
        if cached_tensor is None:
            raise RuntimeError(f"Failed to cache tensor data from {file_path}")
        return cached_tensor

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError("index out of range")

        record_idx = bisect_right(self._cumulative_sizes, index)
        record = self.records[record_idx]
        previous_total = 0 if record_idx == 0 else self._cumulative_sizes[record_idx - 1]
        sample_idx = index - previous_total

        x_tensor = self._load_tensor(record.path)[sample_idx]
        y_tensor = self.class_to_idx[record.class_name]
        return x_tensor, y_tensor
