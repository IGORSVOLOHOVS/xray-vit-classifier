from pathlib import Path
from typing import Any

import torch
from PIL import Image, ImageDraw, ImageFont
from sklearn.metrics import classification_report
from torch import nn
from torch.utils.data import DataLoader

from .data import DatasetManager
from .models import ImageClassifier, ModelWrapper
from .ui import UIHandler


class TrainingEngine:
    """Orchestrates the training, validation, and evaluation pipeline.

    Utilizes Clean Architecture protocols to remain decoupled from specific
    model or data loading implementations.
    """

    def __init__(
        self,
        model: ImageClassifier,
        ui: UIHandler,
        output_dir: str | Path = "prediction_samples",
    ) -> None:
        """Initializes the engine.

        Args:
            model: An object implementing the ImageClassifier protocol.
            ui: Terminal UI handler.
            output_dir: Directory where prediction samples will be saved.
        """
        self.model = model
        self.ui = ui
        self.output_dir = Path(output_dir)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model.to(self.device)
        self._ensure_output_dir()

    def _ensure_output_dir(self) -> None:
        """Creates or cleans the output directory."""
        if self.output_dir.exists():
            import shutil

            shutil.rmtree(self.output_dir)
        self.output_dir.mkdir(parents=True)

    def train_epoch(
        self,
        loader: DataLoader[Any],
        epoch: int,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
    ) -> float:
        """Trains the model for one epoch.

        Args:
            loader: Training data loader.
            epoch: Current epoch index.
            optimizer: Optimization algorithm.
            criterion: Loss function.

        Returns:
            The average loss for the last 20 batches or overall.
        """
        self.model.train()
        running_loss = 0.0
        last_loss = 0.0

        with self.ui.get_progress_bar() as progress:
            task = progress.add_task(f"Training Epoch {epoch}", total=len(loader))

            for i, (inputs, labels) in enumerate(loader):
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs.logits, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                progress.update(task, advance=1)

                if i % 20 == 19:
                    last_loss = running_loss / 20.0
                    running_loss = 0.0

        return last_loss

    def validate(self, loader: DataLoader[Any], criterion: nn.Module) -> float:
        """Validates the model.

        Args:
            loader: Validation data loader.
            criterion: Loss function.

        Returns:
            The average validation loss.
        """
        self.model.eval()
        running_vloss = 0.0
        with torch.no_grad():
            for vinputs, vlabels in loader:
                vinputs, vlabels = vinputs.to(self.device), vlabels.to(self.device)
                voutputs = self.model(vinputs)
                vloss = criterion(voutputs.logits, vlabels)
                running_vloss += vloss.item()
        return running_vloss / len(loader) if len(loader) > 0 else 0.0

    def evaluate_and_sample(self, loader: DataLoader[Any], classes: list[str]) -> dict[str, Any]:
        """Evaluates the model and saves visual samples.

        Args:
            loader: Test data loader.
            classes: List of class names.

        Returns:
            A classification report dictionary.
        """
        self.model.eval()
        all_preds: list[int] = []
        all_labels: list[int] = []
        correct_samples: list[dict[str, Any]] = []
        incorrect_samples: list[dict[str, Any]] = []

        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(loader):
                images_dev = images.to(self.device)
                outputs = self.model(images_dev)
                _, predicted = torch.max(outputs.logits, 1)

                preds_cpu = predicted.cpu().numpy()
                labels_cpu = labels.numpy()

                all_preds.extend(preds_cpu)
                all_labels.extend(labels_cpu)

                # Collect samples
                for i in range(len(preds_cpu)):
                    if len(correct_samples) >= 3 and len(incorrect_samples) >= 3:
                        break

                    dataset: Any = loader.dataset
                    batch_size = loader.batch_size if loader.batch_size is not None else 1
                    img_path, _ = dataset.samples[batch_idx * batch_size + i]

                    sample = {
                        "path": img_path,
                        "true": classes[labels_cpu[i]],
                        "pred": classes[preds_cpu[i]],
                    }

                    if len(correct_samples) < 3 and preds_cpu[i] == labels_cpu[i]:
                        correct_samples.append(sample)
                    elif len(incorrect_samples) < 3 and preds_cpu[i] != labels_cpu[i]:
                        incorrect_samples.append(sample)

        # Save samples
        self._save_images(correct_samples, "correct")
        self._save_images(incorrect_samples, "incorrect")

        report: dict[str, Any] = classification_report(
            all_labels, all_preds, target_names=classes, output_dict=True
        )
        return report

    def _save_images(self, samples: list[dict[str, Any]], prefix: str) -> None:
        """Saves annotated images for inspection."""
        for i, sample in enumerate(samples):
            img = Image.open(sample["path"]).convert("RGB")
            draw = ImageDraw.Draw(img)
            text = f"True: {sample['true']}\nPred: {sample['pred']}"
            font: Any
            try:
                font = ImageFont.truetype("arial.ttf", 20)
            except Exception:
                font = ImageFont.load_default()

            draw.text((10, 10), text, fill="red" if prefix == "incorrect" else "green", font=font)
            img.save(self.output_dir / f"{prefix}_{i}.png")


def run_pipeline() -> None:
    """Default execution pipeline."""
    ui = UIHandler()

    # Hardware Check
    cuda_available = torch.cuda.is_available()
    device_name = "cuda" if cuda_available else "cpu"
    ui.print_panel(
        f"CUDA: {'[green]ON[/]' if cuda_available else '[red]OFF[/]'} | Device: [blue]{device_name}[/]",
        title="Environment",
    )

    # Load Model
    model_name = "google/vit-base-patch16-224"
    with ui.console.status("[bold green]Initializing model..."):
        model_wrapper = ModelWrapper(model_name)

    # Load Data
    data_manager = DatasetManager("Bone_Fracture_Binary_Classification", model_wrapper.processor)
    train_loader, val_loader, test_loader, stats = data_manager.load_splits()
    ui.print_stats_table("Dataset Statistics", stats)

    # Setup Training
    engine = TrainingEngine(model_wrapper, ui)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model_wrapper.model.classifier.parameters(), lr=0.001)

    # Execute Pipeline
    for epoch in range(1):
        avg_loss = engine.train_epoch(train_loader, epoch, optimizer, criterion)
        avg_vloss = engine.validate(val_loader, criterion)
        ui.log(f"Epoch {epoch} | Train Loss: {avg_loss:.4f} | Val Loss: {avg_vloss:.4f}")

    # Evaluation
    ui.log("Evaluating model on test set...")
    report = engine.evaluate_and_sample(test_loader, data_manager.classes)
    ui.print_classification_report("Final Metrics", report)
