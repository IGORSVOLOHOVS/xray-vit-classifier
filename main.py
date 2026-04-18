import argparse
import sys
from pathlib import Path

from rich.traceback import install

from xray_classifier.engine import run_pipeline
from xray_classifier.ui import UIHandler

# Install rich traceback at the very beginning of the entry point
install(show_locals=True)


def main() -> None:
    """Main entry point for the X-Ray Classifier CLI."""
    parser = argparse.ArgumentParser(
        description="X-Ray Bone Fracture Classifier - Professional ViT implementation."
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Command: train
    train_parser = subparsers.add_parser("train", help="Train the classification head")
    train_parser.add_argument(
        "--epochs", type=int, default=1, help="Number of epochs to train (default: 1)"
    )
    train_parser.add_argument(
        "--batch-size", type=int, default=32, help="Batch size (default: 32)"
    )

    # Command: eval
    eval_parser = subparsers.add_parser("eval", help="Evaluate the model and generate samples")
    eval_parser.add_argument(
        "--weights", type=str, default="classifier_0.pt", help="Path to model weights"
    )

    # Command: explain (visualize attention)
    explain_parser = subparsers.add_parser("explain", help="Generate attention maps for errors")
    explain_parser.add_argument(
        "--weights", type=str, default="classifier_0.pt", help="Path to model weights"
    )
    explain_parser.add_argument(
        "--count", type=int, default=10, help="Number of error samples to visualize"
    )

    args = parser.parse_args()

    ui = UIHandler()

    if args.command == "train":
        ui.log(f"Starting training for {args.epochs} epoch(s)...")
        run_pipeline()  # Pipeline currently does 1 epoch and eval
    elif args.command == "eval":
        ui.log(f"Evaluating model using weights: {args.weights}")
        # Logic to be refined in engine.py for pure evaluation
        run_pipeline() 
    elif args.command == "explain":
        ui.log(f"Generating {args.count} attention maps using {args.weights}...")
        # Will bridge to visualize_attention logic
        try:
            from visualize_attention import main as run_viz
            run_viz()
        except ImportError:
            ui.log("[bold red]Failed to import visualization module.")
    else:
        parser.print_help()
        sys.exit(0)


if __name__ == "__main__":
    main()
