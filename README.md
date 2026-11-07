# xray-vit-classifier

A console tool that fine-tunes a Vision Transformer to detect bone fractures in X-ray images. It takes the pretrained `google/vit-base-patch16-224`, freezes the backbone, trains a new two-way head on a folder of labelled radiographs, and reports precision and recall on the test split. A second command draws attention maps over the images the model got wrong, to show what it was looking at.

The dataset and the weights are not in this repository. The first real run downloads the backbone from Hugging Face. The tests never do: they run offline and build a small model from a config.

## Use

```powershell
.\recipe.ps1 run train
.\recipe.ps1 run explain
```

`docs/install.md` says what to install and where the dataset goes. `docs/manual.md` describes the commands, including which options they do not read yet. `.\recipe.ps1 test` runs the tests, mypy and ruff.

## Results

One run recorded before this repository came to the lab, on the test split:

| Metric | Fractured | Not fractured | Macro average |
| --- | --- | --- | --- |
| Precision | 0.84 | 0.82 | 0.83 |
| Recall | 0.79 | 0.86 | 0.83 |
| F1 | 0.81 | 0.84 | 0.83 |

Accuracy was 83%. `prediction_samples/` and `attention_maps/` hold the images from that run. Nothing here re-runs it, because the dataset is not in the repository.

## What is here

- `main.py` is the command line, and `visualize_attention.py` the attention-map run behind `explain`.
- `src/xray_classifier/` is the package: `models.py` wraps the ViT, `data.py` loads the image folders, `engine.py` trains and evaluates, `interpretability.py` draws the maps, `ui.py` is the console output.
- `tests/` holds the tests. `benchmarks/` measures the image transform with `pytest benchmarks --benchmark-only`, on a stand-in processor so that it measures this project's code and loads no weights. The numbers from before the lab are on the tag `import/dev`.
- `docs/` holds the architecture notes, the model decision in `docs/adr/`, the ISO/IEC 25010 assessment in `docs/quality-iso25010.md`, and `docs/branching.md`.
- `scripts/` holds the helpers kept from before the lab. `collect_quality_metrics.py` produces the numbers of the assessment and `build_release_artifact.py` a wheel with checksums. `profile_application.py` came from a template and imports a package this repository does not have, so it does not run.

## Licence

Apache 2.0, see `LICENSE`.
