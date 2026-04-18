from pathlib import Path

import torch
from torch import nn
from transformers import ViTForImageClassification, ViTImageProcessor


class ModelWrapper:
    """Wraps ViT model for training and inference."""

    def __init__(self, model_name: str, num_labels: int = 2) -> None:
        self.model_name = model_name
        self.num_labels = num_labels

        # Load pre-trained components
        self.processor = ViTImageProcessor.from_pretrained(model_name)
        self.model = ViTForImageClassification.from_pretrained(model_name, output_attentions=True)

        # Re-initialize the classifier head for binary classification
        self.model.classifier = nn.Linear(self.model.classifier.in_features, num_labels)

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
        self.model.to(device)
        return self

    def save(self, path: str | Path) -> None:
        """Saves the model state dict."""
        torch.save(self.model.state_dict(), path)

    def train(self) -> None:
        self.model.train()

    def eval(self) -> None:
        self.model.eval()

    def __call__(self, *args: any, **kwargs: any) -> any:
        return self.model(*args, **kwargs)
