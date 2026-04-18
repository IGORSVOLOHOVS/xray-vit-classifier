from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import torch
from PIL import Image, ImageFile
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from transformers import ViTImageProcessor

# Allow PIL to load truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True


@runtime_checkable
class DataProvider(Protocol):
    """Protocol for data loading and management."""

    def load_splits(
        self, batch_size: int = 32
    ) -> tuple[DataLoader[Any], DataLoader[Any], DataLoader[Any], dict[str, int]]: ...

    @property
    def classes(self) -> list[str]: ...


class ViTTransform:
    """Wrapper for ViT image processing as a torch transform.

    This ensures that image processing logic remains consistent across
    training and inference.
    """

    def __init__(self, processor: ViTImageProcessor) -> None:
        self.processor = processor

    def __call__(self, image: Image.Image) -> torch.Tensor:
        """Transforms a PIL image into a torch tensor."""
        if image.mode != "RGB":
            image = image.convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt")
        # Extract pixel values tensor
        tensor = inputs["pixel_values"].squeeze(0)
        if not isinstance(tensor, torch.Tensor):
            raise TypeError("Expected torch.Tensor from processor")
        return tensor


class DatasetManager:
    """Manages dataset loading and split creation.

    Implements the DataProvider protocol to decouple the engine from
    the specific filesystem structure.
    """

    def __init__(self, data_root: str | Path, processor: ViTImageProcessor) -> None:
        self.data_root = Path(data_root)
        self.processor = processor
        self.transform = ViTTransform(processor)
        self._classes: list[str] = []

    def load_splits(
        self, batch_size: int = 32
    ) -> tuple[DataLoader[Any], DataLoader[Any], DataLoader[Any], dict[str, int]]:
        """Loads train, val, and test splits and returns DataLoaders.

        Args:
            batch_size: Number of samples per batch.

        Returns:
            A tuple containing (train_loader, val_loader, test_loader, stats_dict).
        """
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
        """Returns the list of class names."""
        return self._classes
