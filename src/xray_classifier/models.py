from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import torch
from torch import nn
from transformers import ViTForImageClassification, ViTImageProcessor


@runtime_checkable
class ImageClassifier(Protocol):
    """Protocol for image classification models."""

    def to(self, device: torch.device) -> "ImageClassifier": ...
    def save(self, path: str | Path) -> None: ...
    def train(self, mode: bool = True) -> "ImageClassifier": ...
    def eval(self) -> "ImageClassifier": ...
    def __call__(self, x: torch.Tensor) -> Any: ...


class ModelWrapper:
    """Wraps ViT model for training and inference.

    Following Clean Architecture, this class implements the ImageClassifier protocol
    and isolates the transformers dependency from the core engine.
    """

    def __init__(self, model_name: str, num_labels: int = 2) -> None:
        self.model_name = model_name
        self.num_labels = num_labels

        # Load pre-trained components
        self.processor = ViTImageProcessor.from_pretrained(model_name)
        self.model: ViTForImageClassification = ViTForImageClassification.from_pretrained(
            model_name, output_attentions=True
        )

        # Re-initialize the classifier head for binary classification
        in_features = getattr(self.model.classifier, "in_features", 768)
        self.model.classifier = nn.Linear(int(in_features), num_labels)

        # Freeze base layers by default
        self._freeze_base_layers()

    def _freeze_base_layers(self) -> None:
        """Freezes all parameters except the classifier head."""
        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.model.classifier.parameters():
            param.requires_grad = True

    def to(self, device: torch.device) -> "ModelWrapper":
        """Moves the model to the specified device."""
        self.model.to(device)  # type: ignore[arg-type]
        return self

    def save(self, path: str | Path) -> None:
        """Saves the model state dict."""
        torch.save(self.model.state_dict(), path)

    def train(self, mode: bool = True) -> "ModelWrapper":
        """Sets the model to training mode."""
        self.model.train(mode)
        return self

    def eval(self) -> "ModelWrapper":
        """Sets the model to evaluation mode."""
        self.model.eval()  # type: ignore[no-untyped-call]
        return self

    def __call__(self, x: torch.Tensor) -> Any:
        """Forward pass."""
        return self.model(x)
