from pathlib import Path
from typing import Any

import torch
from PIL import Image, ImageDraw, ImageFont
from sklearn.metrics import classification_report
from torch import nn
from torch.utils.data import DataLoader

from .data import DatasetManager
from .models import ModelWrapper
from .ui import UIHandler


class TrainingEngine:
    """Orchestrates the training, validation, and evaluation pipeline."""

    def __init__(
        self, model_wrapper: ModelWrapper, ui: UIHandler, output_dir: str = "prediction_samples"
    ) -> None:
        self.wrapper = model_wrapper
        self.ui = ui
        self.output_dir = Path(output_dir)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.wrapper.to(self.device)
        self._ensure_output_dir()

    def _ensure_output_dir(self) -> None:
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
        self.wrapper.train()
        running_loss = 0.0
        last_loss = 0.0

        with self.ui.get_progress_bar() as progress:
            task = progress.add_task(f"Training Epoch {epoch}", total=len(loader))

            for i, (inputs, labels) in enumerate(loader):
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                optimizer.zero_grad()
                outputs = self.wrapper(inputs)
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
        self.wrapper.eval()
        running_vloss = 0.0
        with torch.no_grad():
            for i, (vinputs, vlabels) in enumerate(loader):
                vinputs, vlabels = vinputs.to(self.device), vlabels.to(self.device)
                voutputs = self.wrapper(vinputs)
                vloss = criterion(voutputs.logits, vlabels)
                running_vloss += vloss.item()
        return running_vloss / len(loader) if len(loader) > 0 else 0.0

    def evaluate_and_sample(self, loader: DataLoader[Any], classes: list[str]) -> dict[str, Any]:
        self.wrapper.eval()
        all_preds = []
        all_labels = []
        correct_samples: list[dict[str, Any]] = []
        incorrect_samples: list[dict[str, Any]] = []

        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(loader):
                images_dev = images.to(self.device)
                outputs = self.wrapper(images_dev)
                _, predicted = torch.max(outputs.logits, 1)

                preds_cpu = predicted.cpu().numpy()
                labels_cpu = labels.numpy()

                all_preds.extend(preds_cpu)
                all_labels.extend(labels_cpu)

                # Collect samples
                for i in range(len(preds_cpu)):
                    dataset: Any = loader.dataset
                    img_path, _ = dataset.samples[batch_idx * loader.batch_size + i]

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
        )  # type: ignore
        return report

    def _save_images(self, samples: list[dict[str, Any]], prefix: str) -> None:
        for i, sample in enumerate(samples):
            img = Image.open(sample["path"]).convert("RGB")
            draw = ImageDraw.Draw(img)
            text = f"True: {sample['true']}\nPred: {sample['pred']}"
            try:
                font = ImageFont.truetype("arial.ttf", 20)
            except Exception:
                font = ImageFont.load_default()

            draw.text((10, 10), text, fill="red" if prefix == "incorrect" else "green", font=font)
            img.save(self.output_dir / f"{prefix}_{i}.png")


def run_pipeline() -> None:
    ui = UIHandler()

    # 1. Hardware Check
    cuda_available = torch.cuda.is_available()
    device_name = "cuda" if cuda_available else "cpu"
    cuda_status = f"[bold {'green' if cuda_available else 'red'}]{cuda_available}[/]"
    ui.print_panel(
        f"CUDA Available: {cuda_status} | Device: [bold blue]{device_name}[/]",
        title="Hardware Check",
    )

    # 2. Load Model
    model_name = "google/vit-base-patch16-224"
    with ui.console.status("[bold green]Loading model and processor..."):
        model_wrapper = ModelWrapper(model_name)

    # 3. Load Data
    data_manager = DatasetManager("Bone_Fracture_Binary_Classification", model_wrapper.processor)
    train_loader, val_loader, test_loader, stats = data_manager.load_splits()
    ui.print_stats_table("Dataset Statistics", stats)

    # 4. Setup Training
    engine = TrainingEngine(model_wrapper, ui)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model_wrapper.model.classifier.parameters(), lr=0.001)

    # 5. Execute Pipeline
    best_vloss = 1e6
    for epoch in range(1):  # Single epoch as requested
        avg_loss = engine.train_epoch(train_loader, epoch, optimizer, criterion)
        avg_vloss = engine.validate(val_loader, criterion)

        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            model_wrapper.save(f"classifier_{epoch}.pt")

        ui.log(f"Epoch {epoch} finished. Train Loss: {avg_loss:.4f} | Val Loss: {avg_vloss:.4f}")

    # 6. Evaluation
    ui.log("Calculating metrics and generating samples...")
    report = engine.evaluate_and_sample(test_loader, data_manager.classes)
    ui.print_classification_report("Classification Report", report)

    # 7. Final Check
    accuracy = report["accuracy"]
    if accuracy > 0.8:
        ui.print_panel(
            f"[bold green]SUCCESS:[/] Model accuracy is {accuracy:.2%} (Target > 80%)",
            title="Final Result",
            style="green",
        )
    else:
        ui.print_panel(
            f"[bold red]FAILURE:[/] Model accuracy is {accuracy:.2%} (Target > 80%)",
            title="Final Result",
            style="red",
        )
