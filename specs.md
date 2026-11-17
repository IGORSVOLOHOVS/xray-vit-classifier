# Specs

A Python console tool that takes the pretrained google/vit-base-patch16-224 Vision Transformer, trains a new two-way head to sort X-ray images into fractured and not fractured, and draws attention maps over the images it gets wrong.

Every `##` section below is one feature. `Status` is one of To Do, In Progress, In Review, Done. Features are mirrored as Feature issues in Jira project KAN; leave the `Jira` line out of a new feature and the sync adds it.

Epic: KAN-34

## Command line with train, eval and explain
Status: Done
Jira: KAN-35

`main.py` parses three subcommands with argparse and is started through `.\recipe.ps1 run <command>`, which puts `src` on `PYTHONPATH`. `train` and `eval` call `run_pipeline()` in `src/xray_classifier/engine.py`, and `explain` imports and runs `visualize_attention.main()`. The commands print through `UIHandler` in `src/xray_classifier/ui.py`: a CUDA/device panel, a dataset statistics table, a training progress bar, a classification report table, log lines through a rich logging handler, and rich tracebacks.

## Load the radiograph dataset into train, val and test splits
Status: Done
Jira: KAN-36

`DatasetManager` in `src/xray_classifier/data.py` reads the train, val and test subfolders of a data root as torchvision `ImageFolder` sets and returns three `DataLoader`s (train shuffled, batch size 32 by default) plus a count per split. Both callers pass the root folder name Bone_Fracture_Binary_Classification, which is not in the repository; `docs/install.md` tells the user to put the dataset there. `ViTTransform` converts each image to RGB and runs it through the ViT image processor, and truncated images are allowed to load.

## Train a two-way head on a frozen ViT backbone
Status: Done
Jira: KAN-37

`ModelWrapper` in `src/xray_classifier/models.py` loads the pretrained processor and model with attentions enabled, replaces the classifier with a new `nn.Linear` to two labels, and freezes every parameter except that head. `run_pipeline` in `src/xray_classifier/engine.py` trains it for one epoch with Adam (lr 0.001) over the head's parameters and cross-entropy loss, through `TrainingEngine.train_epoch` and `TrainingEngine.validate`, on CUDA when available. It logs the train loss (the mean of the last full block of 20 batches) and the mean validation loss.

## Test-set metrics and annotated prediction samples
Status: Done
Jira: KAN-38

`TrainingEngine.evaluate_and_sample` in `src/xray_classifier/engine.py` runs the model over the test split and returns a scikit-learn classification report (precision, recall, F1, support, accuracy), which `run_pipeline` prints as a table. The `TrainingEngine` constructor empties `prediction_samples/`, and `evaluate_and_sample` writes up to three correct and three incorrect test images there, each marked with the true and predicted class.

## Attention maps for misclassified test images
Status: Done
Jira: KAN-39

`AttentionVisualizer.run_error_analysis` in `src/xray_classifier/interpretability.py` finds the test images the model got wrong and, for up to 10 of them, averages the last layer's attention over the heads, takes the CLS-to-patch row, reshapes it to a 14x14 grid, upsamples it bicubically and saves the original image next to the heatmap overlay as a PNG in `attention_maps/`. `visualize_attention.py` feeds it the test split at batch size 1 and loads a full model state dict from classifier_0.pt in the working directory, logging and stopping when that file cannot be loaded.

## Offline checks behind recipe.ps1 test
Status: Done
Jira: KAN-40

`.\recipe.ps1 test` fails when `$Version` and the `pyproject.toml` version differ, points `core.hooksPath` at `.claude/hooks/git`, sets `HF_HUB_OFFLINE`, then runs pytest with coverage, mypy in strict mode, ruff check and ruff format; `.gitea/workflows/test.yaml` runs this verb. `tests/conftest.py` sets `HF_HUB_OFFLINE` before anything imports the hub, and `tests/test_no_weights_are_downloaded.py` asserts the hub is offline for the run. `tests/test_core.py` covers the transform, the head swap and freezing on a small `ViTConfig` model, and the dataset path, and `tests/test_manual_list.py` validates `tests/manual/tests.json`.

## Honour the command-line options
Status: To Do
Jira: KAN-41

`docs/manual.md` lists four options that are accepted and then ignored: `--epochs` (`run_pipeline` loops `range(1)`), `--batch-size` (`load_splits()` is called with its default of 32), and `--weights` and `--count` (`visualize_attention.py` hardcodes `WEIGHTS_PATH` to classifier_0.pt and passes `count=10`). Done when `train --epochs 2 --batch-size 8` trains two epochs at batch size 8, `explain --weights other.pt --count 3` loads other.pt and writes at most three maps, and the option status table in `docs/manual.md` is removed.

## Save the trained model and evaluate from a weights file
Status: To Do
Jira: KAN-42

No command writes a weights file: `ModelWrapper.save` in `src/xray_classifier/models.py` is never called, so `explain` depends on a classifier_0.pt made outside this program. The `eval` branch in `main.py` carries the comment "Logic to be refined in engine.py for pure evaluation" and re-runs the whole training pipeline. Done when `train` writes its state dict with `ModelWrapper.save` to classifier_0.pt, the file `explain` loads by default, and `eval --weights <file>` loads that file and only runs `evaluate_and_sample`, with no training step.

## End-to-end pipeline test on a fixture dataset
Status: To Do
Jira: KAN-43

`docs/quality-iso25010.md` records as a known gap that no test trains on a fixture dataset, and `pyproject.toml` sets the coverage floor to `--cov-fail-under=9` because that is what the suite measures; `TrainingEngine` and `AttentionVisualizer` have no tests. Done when a test builds a tiny train/val/test image folder under pytest's `tmp_path`, runs `train_epoch`, `validate` and `evaluate_and_sample` on the small `ViTConfig` model offline, asserts the report keys and the written sample files, and the coverage floor is raised to what the suite then measures.

## Make the profiling script profile this project
Status: To Do
Jira: KAN-44

`scripts/profile_application.py` came from a template and imports `quality_template.core`, a package this repository does not have, so it fails on start, as the README says. Done when it profiles this project's own code offline (for example `ViTTransform` over generated images, with a stand-in processor like the one in `benchmarks/test_preprocessing_performance.py`) and runs to completion, or it is deleted and the README line about it updated.
