import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path

from xray_classifier.models import ModelWrapper
from xray_classifier.data import DatasetManager
from xray_classifier.ui import UIHandler

# 1. Configuration
MODEL_NAME = "google/vit-base-patch16-224"
WEIGHTS_PATH = "classifier_0.pt"
DATA_ROOT = "Bone_Fracture_Binary_Classification"
OUTPUT_DIR = Path("attention_maps")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    ui = UIHandler()
    OUTPUT_DIR.mkdir(exist_ok=True)

    # 2. Load Model and Weights
    with ui.console.status("[bold green]Loading model and weights..."):
        model_wrapper = ModelWrapper(MODEL_NAME)
        model_wrapper.model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=DEVICE))
        model_wrapper.to(DEVICE)
        model_wrapper.eval()
    
    model = model_wrapper.model # For variable name compliance if needed

    # 3. Load Data
    data_manager = DatasetManager(DATA_ROOT, model_wrapper.processor)
    _, _, test_loader, _ = data_manager.load_splits(batch_size=1)
    test_dataset = test_loader.dataset

    # 4. Find Incorrect Predictions
    labels_predicted = []
    labels_true = []
    
    ui.log("Scanning test set for incorrect predictions...")
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(DEVICE)
            outputs = model_wrapper(inputs)
            _, predicted = torch.max(outputs.logits, 1)
            
            labels_predicted.append(predicted.item())
            labels_true.append(labels.item())

    # Calculate indices of incorrect predictions
    indices = [i for i, (p, t) in enumerate(zip(labels_predicted, labels_true)) if p != t]
    
    if not indices:
        ui.log("[bold red]No incorrect predictions found in the test set!")
        return

    ui.log(f"Found {len(indices)} incorrect predictions. Visualizing first {min(10, len(indices))}...")

    # 5. Visualize Attention Maps
    for i in indices[:10]:
        # Fill requested variables
        img_tensor, label = test_dataset[i]
        img_path, _ = test_dataset.samples[i]
        image = Image.open(img_path).convert("RGB")
        
        # Prediction and Attention extraction
        img_batch = img_tensor.unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            outputs = model_wrapper(img_batch)
            attentions = outputs.attentions # List of (batch, heads, seq, seq)

        # Process last layer attention
        # Shape: (1, 12, 197, 197)
        last_layer_att = attentions[-1][0] # (12, 197, 197)
        
        # Average heads
        avg_att = torch.mean(last_layer_att, dim=0) # (197, 197)
        
        # Attention from CLS token to image patches
        cls_att = avg_att[0, 1:] # (196,)
        
        # Reshape to grid
        att_grid = cls_att.reshape(14, 14).cpu().numpy()
        
        # Normalize for visualization
        att_grid = (att_grid - att_grid.min()) / (att_grid.max() - att_grid.min())

        # Create Visualization
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        # Original Image
        axes[0].imshow(image)
        axes[0].set_title(f"Original (True: {data_manager.classes[label]}, Pred: {data_manager.classes[labels_predicted[i]]})")
        axes[0].axis("off")
        
        # Heatmap Overlay
        axes[1].imshow(image)
        # Interpolate heatmap to original image size
        scaled_att = F.interpolate(
            torch.tensor(att_grid).unsqueeze(0).unsqueeze(0),
            size=(image.height, image.width),
            mode="bicubic",
            align_corners=False
        ).squeeze().numpy()
        
        heatmap = axes[1].imshow(scaled_att, cmap="jet", alpha=0.5)
        axes[1].set_title("Attention Heatmap (Last Layer)")
        axes[1].axis("off")
        
        plt.colorbar(heatmap, ax=axes[1], fraction=0.046, pad=0.04)
        
        # Save
        save_path = OUTPUT_DIR / f"error_{i}_true_{label}_pred_{labels_predicted[i]}.png"
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
        
    ui.log(f"[bold green]Attention maps saved to '{OUTPUT_DIR}/'")

if __name__ == "__main__":
    main()
