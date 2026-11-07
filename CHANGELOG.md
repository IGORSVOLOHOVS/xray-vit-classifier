# Changelog

## v1.1.0 (2026-11-07)

### Commits since v1.0.0

- c64314c docs(docs): name revision 4.0.0 and its exceptions in the README
- 2fd5a1c ci(ci): add the five lab workflows and the issue and pull request templates
- 6f3eaa6 chore(core): add the AI environment, its hooks and the lab-ai-env skill
- 751ecd6 docs(docs): add the README, licence, rules and manual
- 51fd833 build(build): add recipe.ps1 and the test suites it runs
- 923bc6a refactor(core): point the quality metrics script at the lab's workflows
- 8db1bde build(build): pin the toolchain, and decide line endings and ignores
- e967db8 chore: update the pre-push check
- 15d71d5 chore: stop the secret scan running on a weekly schedule
- d352d31 chore: add the pre-push check that mirrors CI
- 2bf0873 chore: add the pre-push check that mirrors CI

### Issues

- #1 Pin the toolchain, and decide line endings and ignores in xray-vit-classifier (open)
- #2 Bring the sources of xray-vit-classifier under the pinned tools (open)
- #3 Add recipe.ps1 and its test suites to xray-vit-classifier (open)
- #4 Add the README, licence, rules and manual to xray-vit-classifier (open)
- #5 Add the AI environment to xray-vit-classifier (open)
- #6 Add the gates and templates to xray-vit-classifier (open)
- #7 State the revision 4.0.0 compliance line in xray-vit-classifier (open)

### Pull requests

- #15 build(build): pin the toolchain, and decide line endings and ignores (closed)
- #16 refactor(core): point the quality metrics script at the lab's workflows (closed)
- #17 build(build): add recipe.ps1 and the test suites it runs (closed)
- #18 docs(docs): add the README, licence, rules and manual (closed)
- #19 chore(core): add the AI environment, its hooks and the lab-ai-env skill (closed)
- #20 ci(ci): add the five lab workflows and the issue and pull request templates (closed)
- #21 docs(docs): name revision 4.0.0 and its exceptions in the README (closed)

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
