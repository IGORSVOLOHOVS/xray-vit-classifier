"""Script for visualizing ViT attention maps on misclassified samples."""

import torch
from xray_classifier.models import ModelWrapper
from xray_classifier.data import DatasetManager
from xray_classifier.ui import UIHandler
from xray_classifier.interpretability import AttentionVisualizer

# Configuration
MODEL_NAME = "google/vit-base-patch16-224"
WEIGHTS_PATH = "classifier_0.pt"
DATA_ROOT = "Bone_Fracture_Binary_Classification"


def main() -> None:
    """Main execution block for visualization."""
    ui = UIHandler()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load Model
    with ui.console.status("[bold green]Loading model and weights..."):
        model_wrapper = ModelWrapper(MODEL_NAME)
        try:
            model_wrapper.model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device, weights_only=True))
        except Exception as e:
            ui.log(f"[bold red]Could not load weights from {WEIGHTS_PATH}: {e}")
            return
        model_wrapper.to(device)

    # 2. Load Data
    data_manager = DatasetManager(DATA_ROOT, model_wrapper.processor)
    _, _, test_loader, _ = data_manager.load_splits(batch_size=1)

    # 3. Visualize
    visualizer = AttentionVisualizer(model_wrapper, data_manager, ui)
    visualizer.run_error_analysis(test_loader, count=10)


if __name__ == "__main__":
    main()
