from pathlib import Path
from unittest.mock import MagicMock, patch

import torch
from transformers import ViTConfig, ViTForImageClassification

from xray_classifier.data import DatasetManager, ViTTransform
from xray_classifier.models import ModelWrapper


def tiny_vit() -> ViTForImageClassification:
    """The real model class, built from a small config: there are no weights to fetch."""
    config = ViTConfig(
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=64,
        image_size=32,
        patch_size=16,
        num_labels=5,
    )
    return ViTForImageClassification(config)


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
    """The wrapper replaces the head with a two-way one and trains only that head."""
    model_name = "google/vit-base-patch16-224"
    with (
        patch("xray_classifier.models.ViTImageProcessor.from_pretrained") as processor,
        patch(
            "xray_classifier.models.ViTForImageClassification.from_pretrained",
            return_value=tiny_vit(),
        ) as model,
    ):
        wrapper = ModelWrapper(model_name, num_labels=2)

    processor.assert_called_once_with(model_name)
    model.assert_called_once_with(model_name, output_attentions=True)
    assert wrapper.num_labels == 2
    assert wrapper.model.classifier.out_features == 2
    trainable = {name for name, value in wrapper.model.named_parameters() if value.requires_grad}
    assert trainable == {"classifier.weight", "classifier.bias"}


def test_dataset_manager_paths() -> None:
    """Verifies path handling in DatasetManager."""
    manager = DatasetManager("test_data", MagicMock())
    assert manager.data_root == Path("test_data")
