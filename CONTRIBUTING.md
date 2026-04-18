# Contributing to X-Ray ViT Classifier

Thank you for your interest in contributing to this project! This document outlines the standards and workflows we follow to maintain code quality and architectural integrity.

## Development Principles

1.  **Clean Architecture**: Keep the business logic (Functional Core) isolated from external frameworks or IO (Imperative Shell).
2.  **Strict Typing**: All Python code must have type hints and pass `mypy --strict`.
3.  **Semantic Commits**: We use Conventional Commits (`type(scope): Description.`).
4.  **Ruff is the Law**: All code must be formatted and linted by Ruff.

## Getting Started

1.  Clone the repository.
2.  Install dependencies:
    ```bash
    pip install -e .
    ```
3.  Install pre-commit hooks:
    ```bash
    pip install pre-commit
    pre-commit install
    ```

## Development Workflow

-   Use `make lint` to run static analysis.
-   Use `make test` to run the test suite.
-   Ensure all PRs include relevant tests.

## Commit Guidelines

Every commit must follow the semantic pattern:
-   `feat(scope): ...`
-   `fix(scope): ...`
-   `docs(scope): ...`
-   `refactor(scope): ...`

Example: `feat(core): Add Protocol for model engine.`

---
*Follow the [ISO 25010](https://iso25000.com/index.php/en/iso-25000-standards/iso-25010) quality attributes when designing new features.*
