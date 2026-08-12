"""Deterministic synthetic and caller-supplied tensor datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Any


class SyntheticClassificationDataset:
    def __init__(
        self,
        *,
        length: int,
        input_dim: int,
        num_classes: int,
        seed: int,
        torch: Any,
    ) -> None:
        self._length = length
        self._input_dim = input_dim
        self._num_classes = num_classes
        self._seed = seed
        self._torch = torch

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, index: int) -> tuple[Any, Any]:
        generator = self._torch.Generator().manual_seed(self._seed + int(index))
        features = self._torch.randn(self._input_dim, generator=generator)
        label = self._torch.tensor(
            (int(index) * 104729 + self._seed) % self._num_classes,
            dtype=self._torch.long,
        )
        return features, label


class TensorDirectoryDataset:
    """Load explicit ``.pt`` samples without imposing an image dependency."""

    def __init__(self, root: Path, *, torch: Any) -> None:
        self._root = Path(root)
        self._torch = torch
        if not self._root.is_dir():
            raise FileNotFoundError(f"data root does not exist: {self._root}")
        self._samples = tuple(sorted(self._root.rglob("*.pt")))
        if not self._samples:
            raise ValueError(f"data root contains no .pt samples: {self._root}")

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> tuple[Any, Any]:
        sample = self._torch.load(
            self._samples[index],
            map_location="cpu",
            weights_only=True,
        )
        if isinstance(sample, dict):
            features, label = sample["input"], sample["target"]
        else:
            features, label = sample
        return features, self._torch.as_tensor(label, dtype=self._torch.long)


def build_dataset(config: Any, *, torch: Any, world_size: int) -> Any:
    if config.synthetic:
        length = max(
            config.steps * config.batch_size_per_rank * world_size,
            config.batch_size_per_rank * world_size,
        )
        return SyntheticClassificationDataset(
            length=length,
            input_dim=config.input_dim,
            num_classes=config.num_classes,
            seed=config.seed,
            torch=torch,
        )
    return TensorDirectoryDataset(config.data_root, torch=torch)
