# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-04-18

### Added
- Initial release of the high-performance ViT-base binary classifier for bone fracture detection.
- Deep Learning engine with support for half-precision (FP16) and GPU acceleration.
- Rich CLI interface for training and inference feedback.
- Model interpretability module with Attention Map visualization.
- Comprehensive test suite for model and data components.
- Modern Python project structure with `pyproject.toml`, Ruff, Mypy, and Pre-commit.
- Professional architecture documentation and ADRs.
- GitHub Release assets including pre-trained weights (`xray_vit_classifier_v0.1.0.pt`).

### Fixed
- Image resolution mismatch in pipeline (dynamically adjusted to model config).
- Input size resolution based on ViT model requirements.
