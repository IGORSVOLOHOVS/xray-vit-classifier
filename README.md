# 🩻 X-Ray Bone Fracture Classifier (ViT)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-yellow.svg)](https://opensource.org/licenses/Apache-2.0)
[![Code Style: Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Checked with mypy](https://img.shields.io/badge/checked_with-mypy-21c0ad.svg)](http://mypy-lang.org/)

A professional, high-performance binary classifier for detecting bone fractures in X-ray images using **Vision Transformers (ViT)**. Designed with **Clean Architecture** principles and optimized for **ISO 25010** quality attributes.

![Hero Image](prediction_samples/correct_0.png)

## 📌 Features

-   **SOTA Architecture**: Fine-tuned `google/vit-base-patch16-224`.
-   **Clean Architecture**: Strict separation between Functional Core (logic) and Imperative Shell (CLI/UI).
-   **Terminal UX**: Premium console interface powered by `rich` with hardware-aware logs and progress bars.
-   **Interpretability**: Built-in attention map visualization for error analysis.
-   **Quality Enforced**: Validated with `mypy --strict`, `ruff`, and `bandit`.

## 🏗 Architecture

The project follows a **Functional Core / Imperative Shell** pattern to maximize testability and maintainability (ISO 25010).

```mermaid
graph TD
    subgraph Imperative_Shell [Imperative Shell (Adapters)]
        CLI[main.py CLI]
        UI[UIHandler]
    end

    subgraph Functional_Core [Functional Core (Domain)]
        Engine[TrainingEngine]
        Interpret[AttentionVisualizer]
    end

    subgraph Infrastructure [Infrastructure]
        Model[ModelWrapper / ViT]
        Data[DatasetManager]
    end

    CLI --> Engine
    CLI --> Interpret
    Engine --> Model
    Engine --> Data
    Interpret --> Model
    Engine --> UI
```

## 🚀 Installation

```bash
# Clone the repository
git clone https://github.com/username/xray-vit-classifier.git
cd xray-vit-classifier

# Install using Makefile
make install
```

## tool usage (Professional CLI)

```bash
# Train the model (1 epoch by default)
python main.py train

# Evaluate and generate prediction samples
python main.py eval --weights xray_vit_classifier_v0.1.0.pt

# Analyze errors with Attention Heatmaps
python main.py explain --count 10
```

## 📊 Performance Metrics

| Metric | Fractured | Not Fractured | Macro Avg |
| :--- | :---: | :---: | :---: |
| **Precision** | 0.84 | 0.82 | 0.83 |
| **Recall** | 0.79 | 0.86 | 0.83 |
| **F1-Score** | 0.81 | 0.84 | 0.83 |

**Global Accuracy: 83%** (Target: >80%)

## 🔍 Interpretability (Error Analysis)

Our analysis shows that errors often occur due to **misaligned attention** on medical hardware or orientation labels.

| Attention Map Analysis | Description |
| :---: | :--- |
| ![Error Analysis](attention_maps/error_1_t0_p1.png) | Model distracted by metal implants rather than the fracture site. |

## 🛠 Development

Refer to [CONTRIBUTING.md](CONTRIBUTING.md) for local setup and coding standards.

```bash
make lint   # Run Ruff and Mypy --strict
make test   # Run unit tests
```

---
*Developed with precision for medical CV applications.*
