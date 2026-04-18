from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from PIL import Image

from .data import DataProvider
from .models import ImageClassifier
from .ui import UIHandler


class AttentionVisualizer:
    """Handles ViT attention map visualization and interpretation.

    Following Clean Architecture, this class focuses on the 'explanation' logic
    independent of the CLI or training process.
    """

    def __init__(
        self,
        model: ImageClassifier,
        data_provider: DataProvider,
        ui: UIHandler,
        output_dir: str | Path = "attention_maps",
    ) -> None:
        self.model = model
        self.data_provider = data_provider
        self.ui = ui
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def run_error_analysis(self, test_loader: Any, count: int = 10) -> None:
        """Finds incorrect predictions and generates attention maps.

        Args:
            test_loader: DataLoader for the test set.
            count: Number of errors to visualize.
        """
        self.model.eval()
        indices: list[int] = []
        all_preds: list[int] = []
        all_labels: list[int] = []

        self.ui.log("Scanning dataset for incorrect predictions...")

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.logits, 1)

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        indices = [i for i, (p, t) in enumerate(zip(all_preds, all_labels)) if p != t]

        if not indices:
            self.ui.log("[bold red]No errors found to analyze.")
            return

        limit = min(count, len(indices))
        self.ui.log(f"Visualizing top {limit} errors...")

        for i in indices[:limit]:
            self._visualize_sample(i, test_loader.dataset, all_preds[i], all_labels[i])

    def _visualize_sample(self, index: int, dataset: Any, pred: int, true_label: int) -> None:
        """Generates and saves attention map for a single sample."""
        img_tensor, _ = dataset[index]
        img_path, _ = dataset.samples[index]
        image = Image.open(img_path).convert("RGB")

        img_batch = img_tensor.unsqueeze(0).to(self.device)
        with torch.no_grad():
            outputs = self.model(img_batch)
            attentions = outputs.attentions

        # Average last layer attention across heads
        last_layer_att = attentions[-1][0]
        avg_att = torch.mean(last_layer_att, dim=0)

        # Extract attention from CLS token to patches (exclude CLS itself at index 0)
        cls_att = avg_att[0, 1:]

        # Reshape to grid (assuming 14x14 patches for 224x224 input)
        att_grid = cls_att.reshape(14, 14).cpu().numpy()
        att_grid = (att_grid - att_grid.min()) / (att_grid.max() - att_grid.min())

        # Plotting
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        axes[0].imshow(image)
        axes[0].set_title(
            f"Original (T: {self.data_provider.classes[true_label]}, P: {self.data_provider.classes[pred]})"
        )
        axes[0].axis("off")

        # Interpolate heatmap
        scaled_att = (
            F.interpolate(
                torch.tensor(att_grid).unsqueeze(0).unsqueeze(0),
                size=(image.height, image.width),
                mode="bicubic",
                align_corners=False,
            )
            .squeeze()
            .numpy()
        )

        axes[1].imshow(image)
        heatmap = axes[1].imshow(scaled_att, cmap="jet", alpha=0.5)
        axes[1].set_title("Attention Heatmap")
        axes[1].axis("off")

        plt.colorbar(heatmap, ax=axes[1], fraction=0.046, pad=0.04)

        save_path = self.output_dir / f"error_{index}_t{true_label}_p{pred}.png"
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
