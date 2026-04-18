from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch
from xray_classifier.data import DatasetManager, ViTTransform
from xray_classifier.models import ModelWrapper


def test_vit_transform_rgb() -> None:
    """Verifies that the transform correctly handles image conversion."""
    processor = MagicMock()
    processor.return_value = {"pixel_values": torch.zeros((1, 3, 224, 224))}
    transform = ViTTransform(processor)

    from PIL import Image

    img = Image.new("L", (100, 100))  # Grayscale
    result = transform(img)

    assert isinstance(result, torch.Tensor)
    assert result.shape == (3, 224, 224)


def test_model_wrapper_initialization() -> None:
    """Tests that the model wrapper initializes with correct head size."""
    # We use a tiny model to speed up initialization if possible,
    # but here we just check if it loads.
    model_name = "google/vit-base-patch16-224"
    wrapper = ModelWrapper(model_name, num_labels=2)

    assert wrapper.num_labels == 2
    assert wrapper.model.classifier.out_features == 2


def test_dataset_manager_paths() -> None:
    """Verifies path handling in DatasetManager."""
    manager = DatasetManager("test_data", MagicMock())
    assert manager.data_root == Path("test_data")
