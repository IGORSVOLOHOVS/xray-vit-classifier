import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets import load_dataset
from transformers import ViTForImageClassification, ViTImageProcessor
from PIL import ImageFile, Image, ImageDraw, ImageFont
from sklearn.metrics import classification_report
import os
import shutil
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

# 0. Initialize Rich Console
console = Console()

# Allow PIL to load truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

# 1. Load Model and Processor
model_name = "google/vit-base-patch16-224"
with console.status("[bold green]Loading model and processor..."):
    processor = ViTImageProcessor.from_pretrained(model_name)
    model = ViTForImageClassification.from_pretrained(
        model_name, 
        output_attentions=True
    )

# 2. Define Wrapper for Processor
def vit_transform(image):
    if image.mode != "RGB":
        image = image.convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    return inputs["pixel_values"].squeeze(0)

# 3. Load Datasets
data_dir = "Bone_Fracture_Binary_Classification"
train_path = os.path.join(data_dir, "train")
val_path = os.path.join(data_dir, "val")
test_path = os.path.join(data_dir, "test")

from torchvision.datasets import ImageFolder

train_dataset = ImageFolder(root=train_path, transform=vit_transform)
val_dataset = ImageFolder(root=val_path, transform=vit_transform)
test_dataset = ImageFolder(root=test_path, transform=vit_transform)

# 4. Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

stats_table = Table(title="Dataset Statistics")
stats_table.add_column("Split", style="cyan")
stats_table.add_column("Count", style="magenta")
stats_table.add_row("Training", str(len(train_dataset)))
stats_table.add_row("Validation", str(len(val_dataset)))
stats_table.add_row("Test", str(len(test_dataset)))
console.print(stats_table)

# 5. Modify Model last layer
model.classifier = nn.Linear(model.classifier.in_features, 2)

# 6. Freeze base layers
for param in model.parameters():
    param.requires_grad = False
for param in model.classifier.parameters():
    param.requires_grad = True

# Device setup
cuda_available = torch.cuda.is_available()
device = torch.device("cuda" if cuda_available else "cpu")
model.to(device)

cuda_color = "green" if cuda_available else "red"
console.print(Panel(f"CUDA Available: [bold {cuda_color}]{cuda_available}[/] | Device: [bold blue]{device}[/]", title="Hardware Check"))

# 7. Setup Training Hyperparameters
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.classifier.parameters(), lr=0.001)
EPOCHS = 1
best_vloss = 1e5

# 8. Training Loop
def train_one_epoch(epoch_index):
    running_loss = 0.0
    last_loss = 0.0
    model.train()
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console
    ) as progress:
        task = progress.add_task(f"Training Epoch {epoch_index}", total=len(train_loader))
        
        for batch_index, data in enumerate(train_loader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            progress.update(task, advance=1)
            
            if batch_index % 20 == 19:
                last_loss = running_loss / 20.0
                # console.log(f"Batch {batch_index}: loss {last_loss:.4f}")
                running_loss = 0.0

    return last_loss

output_dir = "prediction_samples"
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)
os.makedirs(output_dir)

for epoch in range(EPOCHS):
    avg_loss = train_one_epoch(epoch)

    # Validation
    model.eval()
    running_vloss = 0.0
    with torch.no_grad():
        for i, vdata in enumerate(val_loader):
            vinputs, vlabels = vdata
            vinputs, vlabels = vinputs.to(device), vlabels.to(device)
            voutputs = model(vinputs)
            vloss = criterion(voutputs.logits, vlabels)
            running_vloss += vloss

    avg_vloss = running_vloss / (i + 1)

    if avg_vloss < best_vloss:
        best_vloss = avg_vloss
        model_path = f"classifier_{epoch}.pt"
        torch.save(model.state_dict(), model_path)

    console.print(f"[bold green]Epoch {epoch} finished.[/] Train Loss: {avg_loss:.4f} | Val Loss: {avg_vloss:.4f}")

# 9. Evaluation & Sample Generation
console.print("\n[bold cyan]Calculating metrics and generating samples...[/]")
labels_predicted = []
labels_true = []
correct_samples = []
incorrect_samples = []

model.eval()
with torch.no_grad():
    # We'll use a small batch size for sampling if needed, or just iterate once
    for batch_idx, (images, labels) in enumerate(test_loader):
        images_dev = images.to(device)
        outputs = model(images_dev)
        _, predicted = torch.max(outputs.logits, 1)
        
        preds_cpu = predicted.cpu().numpy()
        labels_cpu = labels.numpy()
        
        labels_predicted.extend(preds_cpu)
        labels_true.extend(labels_cpu)
        
        # Collect samples for visualization
        for i in range(len(preds_cpu)):
            if len(correct_samples) < 3 and preds_cpu[i] == labels_cpu[i]:
                # Get original image path
                idx_in_dataset = batch_idx * test_loader.batch_size + i
                img_path, _ = test_dataset.samples[idx_in_dataset]
                correct_samples.append({
                    "path": img_path,
                    "true": test_dataset.classes[labels_cpu[i]],
                    "pred": test_dataset.classes[preds_cpu[i]]
                })
            elif len(incorrect_samples) < 3 and preds_cpu[i] != labels_cpu[i]:
                idx_in_dataset = batch_idx * test_loader.batch_size + i
                img_path, _ = test_dataset.samples[idx_in_dataset]
                incorrect_samples.append({
                    "path": img_path,
                    "true": test_dataset.classes[labels_cpu[i]],
                    "pred": test_dataset.classes[preds_cpu[i]]
                })

# Save samples
def save_sample_image(sample, prefix, index):
    img = Image.open(sample["path"]).convert("RGB")
    draw = ImageDraw.Draw(img)
    text = f"True: {sample['true']}\nPred: {sample['pred']}"
    # Use default font if custom isn't available
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except:
        font = ImageFont.load_default()
    
    draw.text((10, 10), text, fill="red" if prefix == "incorrect" else "green", font=font)
    img.save(os.path.join(output_dir, f"{prefix}_{index}.png"))

for i, s in enumerate(correct_samples):
    save_sample_image(s, "correct", i)
for i, s in enumerate(incorrect_samples):
    save_sample_image(s, "incorrect", i)

console.print(f"[bold green]Generated 3 correct and 3 incorrect sample images in '{output_dir}/'[/]")

# 10. Final Report
report = classification_report(
    labels_true, labels_predicted, target_names=test_dataset.classes, output_dict=True
)

report_table = Table(title="Classification Report")
report_table.add_column("Class", style="cyan")
report_table.add_column("Precision", style="magenta")
report_table.add_column("Recall", style="magenta")
report_table.add_column("F1-Score", style="magenta")
report_table.add_column("Support", style="magenta")

for label, metrics in report.items():
    if isinstance(metrics, dict):
        report_table.add_row(
            label,
            f"{metrics['precision']:.2f}",
            f"{metrics['recall']:.2f}",
            f"{metrics['f1-score']:.2f}",
            str(int(metrics['support']))
        )
    else:
        # accuracy
        report_table.add_row("accuracy", "", "", f"{metrics:.2f}", "")

console.print(report_table)
accuracy = report['accuracy']
if accuracy > 0.8:
    console.print(Panel(f"[bold green]SUCCESS:[/] Model accuracy is {accuracy:.2%} (Target > 80%)", expand=False))
else:
    console.print(Panel(f"[bold red]FAILURE:[/] Model accuracy is {accuracy:.2%} (Target > 80%)", expand=False))
