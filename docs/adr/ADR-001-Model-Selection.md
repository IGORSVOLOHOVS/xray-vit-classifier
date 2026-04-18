# ADR 001: Selection of Vision Transformer (ViT) for X-Ray Classification

*Status: Accepted*

## Context
We need a high-performance backbone for binary bone fracture detection. Medical imaging requires high spatial resolution and the ability to capture subtle textural features.

## Options Considered
1.  **ResNet-50**: Traditional CNN, efficient but may lack global context.
2.  **Vision Transformer (ViT-base)**: Uses self-attention to capture global relationships between patches.
3.  **Swin Transformer**: Hierarchical ViT, better for multi-scale features but more complex.

## Decision
We chose **google/vit-base-patch16-224**.

## Rationale (ISO 25010)
-   **Functional Suitability**: ViT attention maps allow for superior error analysis (Explainability), which is critical for medical trust.
-   **Performance Efficiency**: While computationally heavier than ResNet, ViT shows higher F1-scores in local benchmarks (83%).
-   **Maintainability**: Large pre-trained model support in `transformers` library simplifies the implementation.

## Consequences
-   Requires GPU for efficient training/inference.
-   Model size (~340MB) must be handled carefully in CI/CD.
