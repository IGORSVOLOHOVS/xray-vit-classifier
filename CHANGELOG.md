# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-08-05
### Added
- `benchmarks/test_preprocessing_performance.py`: the image transform measured
  at three input sizes, so a performance regression shows up as a number.
- `docs/quality-iso25010.md`: all eight ISO/IEC 25010 characteristics assessed
  against measured evidence, including the known gaps.
- Downloadable release artefacts: a wheel and an sdist with checksums, verified
  by installing and importing the wheel before it ships.

### Fixed
- `scripts/collect_quality_metrics.py` read only `.github/workflows/ci.yml`, so
  this repository scored zero for portability while actually testing on a
  platform. It now reads every workflow.

### Changed
- CI watches `release`, `dev` and `test`; `main` is gone.

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
