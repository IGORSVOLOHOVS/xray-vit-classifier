from pathlib import Path
from typing import Any

import torch
from PIL import Image, ImageFile
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from transformers import ViTImageProcessor

# Allow PIL to load truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True


class ViTTransform:
    """Wrapper for ViT image processing as a torch transform."""

    def __init__(self, processor: ViTImageProcessor) -> None:
        self.processor = processor

    def __call__(self, image: Image.Image) -> torch.Tensor:
        if image.mode != "RGB":
            image = image.convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt")
        # Cast to Tensor to satisfy type checkers as return_tensors="pt" returns torch.Tensor
        tensor = inputs["pixel_values"].squeeze(0)
        if not isinstance(tensor, torch.Tensor):
            raise TypeError("Expected torch.Tensor from processor")
        return tensor


class DatasetManager:
    """Manages dataset loading and split creation."""

    def __init__(self, data_root: str | Path, processor: ViTImageProcessor) -> None:
        self.data_root = Path(data_root)
        self.processor = processor
        self.transform = ViTTransform(processor)
        self._classes: list[str] = []

    def load_splits(
        self, batch_size: int = 32
    ) -> tuple[DataLoader[Any], DataLoader[Any], DataLoader[Any], dict[str, int]]:
        """Loads train, val, and test splits and returns DataLoaders."""
        train_path = self.data_root / "train"
        val_path = self.data_root / "val"
        test_path = self.data_root / "test"

        train_dataset = ImageFolder(root=str(train_path), transform=self.transform)
        val_dataset = ImageFolder(root=str(val_path), transform=self.transform)
        test_dataset = ImageFolder(root=str(test_path), transform=self.transform)

        self._classes = train_dataset.classes

        loaders = (
            DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
            DataLoader(val_dataset, batch_size=batch_size, shuffle=False),
            DataLoader(test_dataset, batch_size=batch_size, shuffle=False),
        )

        stats = {
            "Training": len(train_dataset),
            "Validation": len(val_dataset),
            "Test": len(test_dataset),
        }

        return (*loaders, stats)

    @property
    def classes(self) -> list[str]:
        return self._classes
