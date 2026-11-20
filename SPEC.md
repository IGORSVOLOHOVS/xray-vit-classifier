# Software Requirements Specification — xray-vit-classifier

Reverse-engineered from the code on branch `xvc-60/docs/spec-md` at commit
`75d38b5` (2026-11-17), one commit ahead of `test` (`42206b3`). Only behaviour
that the code implements is stated as a requirement. Claims from documents that
the code does not back, and tasks proposed but not done, are listed separately
in [3.11](#311-declared-but-not-implemented).

The task list of this file is synced with Jira ticket KAN-60. Each subsection
from 3.1 to 3.10 is a task that is done, and each row of 3.11 is a task still to
do. When this file was merged, KAN-60 held ten of them: 3.1 to 3.5 and 3.8 as
done, and the first four rows of 3.11 as open, under the same titles.

This file merges two specifications. One is the task-per-section `SPEC.md`
added on this branch on 2026-11-17. The other is a requirements specification
written on 2026-11-18 from branch `dev`, whose tip `69fa999` is tagged
`import/dev`. The code under `main.py`, `visualize_attention.py`, `src/` and
`benchmarks/` is the same on `dev` and on this branch. The tooling, the tests
and the documents differ (`git diff --stat origin/dev HEAD`). Where the two
specifications disagreed, this tree decided.

<!-- jira-sync:start -->
Jira: KAN-60

## Tasks

This list is one ticket in Jira project KAN, [KAN-60](https://igorsvolohovs.atlassian.net/browse/KAN-60). Each line is a checkbox there: tick it in Jira and the sync ticks it here; add a checkbox there and it is appended here. Done items are the features described in section 3; open items are what the project declares but the code does not do yet.

- [x] Command line with train, eval and explain (section 3.1) <!-- spec-command-line-with-train-eval-and-explain -->
- [x] Load the radiograph dataset into train, val and test splits (section 3.2) <!-- spec-load-the-radiograph-dataset-into-train-val-and-tes -->
- [x] Train a two-way head on a frozen ViT backbone (section 3.3) <!-- spec-train-a-two-way-head-on-a-frozen-vit-backbone -->
- [x] Test-set metrics and annotated prediction samples (section 3.4) <!-- spec-test-set-metrics-and-annotated-prediction-samples -->
- [x] Attention maps for misclassified test images (section 3.5) <!-- spec-attention-maps-for-misclassified-test-images -->
- [x] Console output (section 3.6) <!-- spec-console-output -->
- [x] Recipe verbs (section 3.7) <!-- spec-recipe-verbs -->
- [x] Offline checks behind recipe.ps1 test (section 3.8) <!-- spec-offline-checks-behind-recipe-ps1-test -->
- [x] Pull-request, release and maintenance gates (section 3.9) <!-- spec-pull-request-release-and-maintenance-gates -->
- [x] Helper scripts (section 3.10) <!-- spec-helper-scripts -->
- [ ] Honour the command-line options (declared, not implemented) <!-- spec-honour-the-command-line-options -->
- [ ] Save the trained model and evaluate from a weights file (declared, not implemented) <!-- spec-save-the-trained-model-and-evaluate-from-a-weights -->
- [ ] End-to-end pipeline test on a fixture dataset (declared, not implemented) <!-- spec-end-to-end-pipeline-test-on-a-fixture-dataset -->
- [ ] Make the profiling script profile this project (declared, not implemented) <!-- spec-make-the-profiling-script-profile-this-project -->
- [ ] Pre-trained weights `xray_vit_classifier_v0.1.0.pt` as a release asset (declared, not implemented) <!-- spec-pre-trained-weights-xrayvitclassifierv0-1-0-pt-as- -->
- [ ] Half precision (FP16) (declared, not implemented) <!-- spec-half-precision-fp16 -->
- [ ] GPU memory monitoring in the UI (declared, not implemented) <!-- spec-gpu-memory-monitoring-in-the-ui -->
- [ ] Backbone swap without changing core logic (declared, not implemented) <!-- spec-backbone-swap-without-changing-core-logic -->
- [ ] Console entry point in the wheel (declared, not implemented) <!-- spec-console-entry-point-in-the-wheel -->
- [ ] Accuracy target > 80 % (declared, not implemented) <!-- spec-accuracy-target-80 -->
- [ ] Scheduled secret scan of the full git history (declared, not implemented) <!-- spec-scheduled-secret-scan-of-the-full-git-history -->
- [ ] Pinned model revision hash (declared, not implemented) <!-- spec-pinned-model-revision-hash -->
- [ ] A committed `.env.example` listing the variable names (declared, not implemented) <!-- spec-a-committed-env-example-listing-the-variable-names -->

<!-- jira-sync:end -->

## 1. Purpose and scope

The system trains and evaluates a two-class image classifier for bone
fractures in X-ray images (`pyproject.toml:17`). It is built on the pretrained
Vision Transformer `google/vit-base-patch16-224`
(`src/xray_classifier/engine.py:203`, `visualize_attention.py:11`), and it
produces attention heatmaps for misclassified test images
(`src/xray_classifier/interpretability.py:35-121`).

In scope (implemented):

- A command-line program with `train`, `eval` and `explain` subcommands (`main.py`), started through `.\recipe.ps1 run`.
- Loading a local image dataset split into `train`, `val` and `test` folders.
- Training a new linear head on a frozen ViT backbone for one epoch, with
  validation loss and a test-set classification report.
- Saving annotated correct and incorrect test predictions as images.
- Attention heatmaps for misclassified test images, from a weights file.
- Repository tooling: the `recipe.ps1` verbs, an offline test suite, five
  Gitea workflows (pull-request gate, `test` build, release gate with a
  manual-test round, release, weekly sweep), git `commit-msg` and `pre-push`
  hooks, Claude Code hooks, a wheel and sdist build with checksums, ISO 25010
  metric collection and a wiki generator.

Out of scope (absent from the code at this commit):

- Saving trained weights from any CLI command.
- Classifying a single image supplied by the user, or any inference service, API or GUI.
- Data download, data augmentation, hyperparameter search, multi-epoch training.
- A release asset carrying the wheel. `release.yaml` attaches an installer
  `.exe` only when `dist\` holds one (`.gitea/workflows/release.yaml:296-301`),
  and `recipe.ps1 build` makes only a wheel (`recipe.ps1:48-51`).
- Any clinical or regulatory function. The code makes no such claim.

## 2. Context

| Element | Description | Source |
| --- | --- | --- |
| Operator | A person who runs `.\recipe.ps1 run <command>` in a terminal. `docs/quality-iso25010.md:87-88` assumes a user "comfortable with a terminal". Who the users actually are is not stated | `README.md:9-12`, `docs/manual.md:3-7`, `docs/quality-iso25010.md:87-88` |
| Maintainer | A person who runs the `recipe.ps1` verbs and proposes changes into `test` by pull request, under the rules in `CLAUDE.md` | `recipe.ps1`, `CONTRIBUTING.md:1-12`, `CLAUDE.md` |
| Tester | A person who runs the manual tests listed in `tests/manual/tests.json` through `tests/manual/serve.py` before a release. The list is empty | `.gitea/workflows/release-gate.yaml:204-219`, `tests/manual/tests.json:3` |
| Hugging Face Hub | Source of the pretrained processor and model `google/vit-base-patch16-224` (downloaded by `from_pretrained` when not cached) | `src/xray_classifier/models.py:32-35`, `docs/install.md:9` |
| Local dataset | Folder `Bone_Fracture_Binary_Classification/` in the working directory, which `recipe.ps1` sets to the repository root; not in the repository | `engine.py:208`, `visualize_attention.py:13`, `recipe.ps1:45`, `.gitignore:30`, `docs/install.md:11` |
| Weights file | `classifier_0.pt` in the working directory, read by `explain`; no command writes it, and `*.pt` is git-ignored | `visualize_attention.py:12,24-27`, `.gitignore:27` |
| Gitea server and runner | Runs `.gitea/workflows/*.yaml` on a runner labelled `windows` and serves the API those workflows call (reviews, comments, commit statuses, releases, packages, issues, branches, wiki), authenticated with the `LAB_BOT_TOKEN` secret | `.gitea/workflows/*.yaml`, `README.md:42` |
| GitHub | Public copy of the repository: the `origin` remote of the checkout this file was written in. The tree has no `.github/` directory, so no workflow of this tree runs there | `git remote -v`, `git ls-files` |
| Lab share | Folder named by the environment variable `LAB_SHARE`, into which `recipe.ps1 publish` copies the wheel | `recipe.ps1:66-75` |
| Jira | Ticket KAN-60 mirrors the task list of this file | this file |
| Runtime | CPython `>=3.13,<3.14`, run through `uv`. Every workflow runs on Windows with `shell: powershell`. CUDA GPU optional; on Windows the locked `torch` is the CPU build | `pyproject.toml:21,23-24`, `docs/install.md:3,7`, `.gitea/workflows/test.yaml:9-16`, `engine.py:38` |

## 3. Functional requirements

### 3.1 Command line with train, eval and explain

| ID | Requirement | Source | Verification |
| --- | --- | --- | --- |
| FR-1 | The system shall accept the subcommands `train`, `eval` and `explain`, with the arguments listed in [§4.1](#41-command-line-mainpy). | `main.py:16-43` | no test |
| FR-2 | The system shall print the help text and exit with code 0 when no subcommand is given. | `main.py:63-65` | no test |
| FR-3 | The system shall run the full training-and-evaluation pipeline (`run_pipeline`) for both `train` and `eval`. | `main.py:47-53` | no test |
| FR-4 | The system shall run `visualize_attention.main()` for `explain`, and log "Failed to import visualization module." if that import raises `ImportError`. | `main.py:54-62` | no test |
| FR-5 | The system shall install rich tracebacks with local variables at start-up. | `main.py:5,11`; `src/xray_classifier/ui.py:22` | no test |

The command line is started through `.\recipe.ps1 run <command>`, which puts
`src` on `PYTHONPATH` (FR-35).

### 3.2 Load the radiograph dataset into train, val and test splits

| ID | Requirement | Source | Verification |
| --- | --- | --- | --- |
| FR-6 | The system shall load `<root>/train`, `<root>/val` and `<root>/test` as torchvision `ImageFolder` datasets. Both callers pass the root `Bone_Fracture_Binary_Classification`, a path relative to the working directory. | `src/xray_classifier/data.py:72-78`; `engine.py:208`; `visualize_attention.py:13,34` | no test |
| FR-7 | The system shall return three `DataLoader`s with a configurable batch size (default 32), shuffling only the training loader. | `data.py:61-86` | no test |
| FR-8 | The system shall return the sample count per split under the keys `Training`, `Validation`, `Test`. | `data.py:88-94` | no test |
| FR-9 | The system shall take the class names from the training split's `ImageFolder.classes`. | `data.py:80,96-99` | no test |
| FR-10 | The system shall convert every non-RGB image to RGB, pass it through the ViT image processor and return `pixel_values` without the batch dimension, raising `TypeError` if the result is not a tensor. | `data.py:36-45` | `test_vit_transform_rgb`; benchmarks `test_transform_scales_with_image_size`, `test_transform_on_rgb_input` |
| FR-11 | The system shall allow PIL to load truncated image files. | `data.py:10-11` | no test |
| FR-12 | The system shall store the dataset root as a `pathlib.Path`. | `data.py:56` | `test_dataset_manager_paths` |

### 3.3 Train a two-way head on a frozen ViT backbone

| ID | Requirement | Source | Verification |
| --- | --- | --- | --- |
| FR-13 | The system shall load the pretrained `ViTImageProcessor` and `ViTForImageClassification` for a given model id, with attention outputs enabled. | `src/xray_classifier/models.py:32-35` | `test_model_wrapper_initialization` (asserts the `from_pretrained` call arguments, with both calls patched) |
| FR-14 | The system shall replace the classifier head with `nn.Linear(in_features, num_labels)`, with `num_labels` defaulting to 2 and `in_features` falling back to 768. | `models.py:27,37-39` | `test_model_wrapper_initialization` (2 outputs; the 768 fall-back is not tested) |
| FR-15 | The system shall freeze all model parameters except those of the new head. | `models.py:41-49` | `test_model_wrapper_initialization` (only `classifier.weight` and `classifier.bias` are trainable) |
| FR-16 | The system shall be able to save the model's state dict to a path (`ModelWrapper.save`); no code path calls it. | `models.py:56-58` | no test |
| FR-17 | The system shall run on CUDA when `torch.cuda.is_available()` is true and on the CPU otherwise. | `engine.py:38`; `interpretability.py:33`; `visualize_attention.py:19` | no test |
| FR-18 | The system shall display an "Environment" panel stating CUDA ON/OFF and the device type (`cuda` or `cpu`, not the GPU model) before loading the model. | `engine.py:193-200` | no test |
| FR-19 | The system shall display a "Dataset Statistics" table with the per-split counts. | `engine.py:209-210`; `ui.py:38-45` | no test |
| FR-20 | The system shall train for exactly one epoch with Adam (learning rate 0.001) over the head parameters and cross-entropy loss, using batch size 32. | `engine.py:208-219`; `data.py:62` | no test |
| FR-21 | The system shall show a progress bar during training and return as training loss the mean of the most recent completed block of 20 batches, or 0.0 if fewer than 20 batches ran. | `engine.py:69-92` | no test |
| FR-22 | The system shall compute the validation loss as the mean batch loss without gradients, returning 0.0 for an empty loader. | `engine.py:94-112` | no test |
| FR-23 | The system shall log `Epoch <n> \| Train Loss: <x.xxxx> \| Val Loss: <x.xxxx>` after each epoch. | `engine.py:221` | no test |

### 3.4 Test-set metrics and annotated prediction samples

| ID | Requirement | Source | Verification |
| --- | --- | --- | --- |
| FR-24 | The system shall delete and recreate the sample output directory (default `prediction_samples`) when `TrainingEngine` is constructed. | `engine.py:26,41-49` | no test |
| FR-25 | The system shall predict every test image by arg-max over the logits and compute a scikit-learn `classification_report` as a dictionary with the class names as target names. | `engine.py:130-169` | no test |
| FR-26 | The system shall save the first three correct and the first three incorrect test predictions as `correct_<i>.png` and `incorrect_<i>.png`, with "True: … / Pred: …" drawn at (10, 10) in green or red, using Arial 20 or PIL's default font. | `engine.py:142-186` | no test |
| FR-27 | The system shall print a "Final Metrics" table with precision, recall and F1 to two decimals, integer support per class and average, and an accuracy row. | `engine.py:224-226`; `ui.py:47-69` | no test |

### 3.5 Attention maps for misclassified test images

| ID | Requirement | Source | Verification |
| --- | --- | --- | --- |
| FR-28 | The system shall load the state dict from `classifier_0.pt` with `weights_only=True` onto the selected device, and on any failure log "Could not load weights from classifier_0.pt: <error>" and stop without writing any file. | `visualize_attention.py:12,22-31` | no test |
| FR-29 | The system shall load the test split with batch size 1 for attention analysis. | `visualize_attention.py:34-35` | no test |
| FR-30 | The system shall predict every test image, collect the indices where prediction and label differ, and log "No errors found to analyze." when there are none. | `interpretability.py:42-62` | no test |
| FR-31 | The system shall visualise the first `min(count, errors)` misclassified images, with `count` fixed at 10 by the caller. | `interpretability.py:64-68`; `visualize_attention.py:39` | no test |
| FR-32 | The system shall build each heatmap from the last layer's attention averaged over heads, take the CLS-to-patch row reshaped to 14 × 14, min-max normalise it and upsample it bicubically to the image size. | `interpretability.py:76-110` | no test |
| FR-33 | The system shall save a two-panel figure (original with true/predicted class title; overlay with `jet` colormap at alpha 0.5 and a colour bar) as `attention_maps/error_<index>_t<true>_p<pred>.png`, creating the directory if missing. | `interpretability.py:26,31-32,92-121` | no test |

### 3.6 Console output

| ID | Requirement | Source | Verification |
| --- | --- | --- | --- |
| FR-34 | The system shall route log messages through Python `logging` at level INFO with a `RichHandler`. The handler is bound to the console of the first `UIHandler` created, because `logging.basicConfig` without `force` does nothing once handlers exist; `main.py:45` and `engine.py:191` / `visualize_attention.py:18` each create their own `UIHandler` and console. | `ui.py:19-32,81-83` | no test |

### 3.7 Recipe verbs

| ID | Requirement | Source | Verification |
| --- | --- | --- | --- |
| FR-35 | `.\recipe.ps1 run <args>` shall set `PYTHONPATH` to `<repository>\src` and run `uv run --frozen python main.py <args>` from the repository root, returning its exit code. | `recipe.ps1:45,60-65` | manual run (§8) |
| FR-36 | `recipe.ps1` shall accept exactly one verb out of `build`, `install`, `uninstall`, `clean`, `run`, `publish`, `info` and `test`, and pass further arguments on only for `run`. | `recipe.ps1:1-7,31-32,63` | no test |
| FR-37 | Before any verb, `recipe.ps1` shall set each `NAME=value` line of `.claude/.env`, when that file exists, as an environment variable, leaving variables that are already set unchanged. | `recipe.ps1:34-43` | no test |
| FR-38 | `build` shall build a wheel of `src/xray_classifier` with `uv build --wheel --out-dir dist` and return uv's exit code. | `recipe.ps1:19,48-51`; `pyproject.toml:54-55` | workflow run (`.gitea/workflows/test.yaml:59-60`) |
| FR-39 | `clean` shall delete `dist`, `build`, `out` and `.coverage` where present and exit 0. | `recipe.ps1:54-59` | workflow run (`test.yaml:111-120`) |
| FR-40 | `info` shall print the name, project code, version, summary and provided packages from the recipe's metadata and exit 0. | `recipe.ps1:11-19,76-83` | no test |
| FR-41 | `install` and `uninstall` shall print that there is nothing to do and exit 0. The recipe defines no `$App`: the project ships as a library. | `recipe.ps1:18,52-53` | no test |
| FR-42 | `publish` shall exit 1 when `LAB_SHARE` is not set; otherwise it shall build the wheel and copy `dist\*.whl` to `$LAB_SHARE\needs\<name>\v<version>`. | `recipe.ps1:66-75` | no test |

### 3.8 Offline checks behind recipe.ps1 test

| ID | Requirement | Source | Verification |
| --- | --- | --- | --- |
| FR-43 | `test` shall read the version from the `[project]` table of `pyproject.toml` and exit 1 with "Version mismatch: recipe.ps1 says <a>, pyproject.toml says <b>." when it differs from `$Version`. | `recipe.ps1:13,84-96`; `pyproject.toml:15-16` | the verb itself |
| FR-44 | `test` shall set git's `core.hooksPath` to `.claude/hooks/git`, exiting 1 if git refuses. | `recipe.ps1:97-99` | the verb itself |
| FR-45 | `test` shall set `HF_HUB_OFFLINE=1` and then run pytest, mypy, `ruff check .` and `ruff format .` (in place, not `--check`) through `uv run --frozen`, in that order, exiting 1 at the first that fails and 0 when all pass. | `recipe.ps1:100-112` | the verb itself |
| FR-46 | pytest shall collect `tests/` only, with `src` and `.` on the import path, and measure branch coverage over `.`, `src/xray_classifier` and `scripts`, failing below 9 %. | `pyproject.toml:103-116` | the verb itself |
| FR-47 | The test suite shall run without reaching the Hugging Face Hub: `tests/conftest.py` sets `HF_HUB_OFFLINE=1` before anything imports `huggingface_hub`, and a test asserts `constants.HF_HUB_OFFLINE is True`. | `tests/conftest.py:1-6`; `tests/test_no_weights_are_downloaded.py:3-7` | `test_the_hub_is_offline_for_the_whole_run` |
| FR-48 | `tests/test_core.py` shall check the transform (grayscale in, `(3, 224, 224)` tensor out, processor mocked), the head swap and the freezing on a small `ViTConfig` model returned by a patched `from_pretrained`, and the stored dataset root. | `tests/test_core.py:11-63` | `test_vit_transform_rgb`, `test_model_wrapper_initialization`, `test_dataset_manager_paths` |
| FR-49 | `tests/test_manual_list.py` shall reject a `tests/manual/tests.json` that starts with a BOM or is not `version` 1 with a `tests` list, and any entry whose `id` is illegal, `verdict` or repeated, whose `title`, `expected`, `steps`, `issue`, `blocking` or `needs` has the wrong type, or whose strings hold control characters. | `tests/test_manual_list.py:9-51` | `test_the_list_has_a_known_shape`, `test_every_entry_is_complete` |
| FR-50 | The `propose`, `test` and `release-gate` workflows shall run `.\recipe.ps1 test` and then fail on `git diff --exit-code`, so a commit that the formatter would change does not pass. | `.gitea/workflows/propose.yaml:144-152`; `test.yaml:54-58`; `release-gate.yaml:49-53` | workflow run |

### 3.9 Pull-request, release and maintenance gates

All workflow files are under `.gitea/workflows/`.

| ID | Requirement | Source | Verification |
| --- | --- | --- | --- |
| FR-51 | On a pull request into `test` (opened, reopened, synchronised), `propose` shall fail unless the head branch matches `<$Code>-<issue>/<type>/<slug>` with a type from `$BranchTypes`; branches named `from-*` are exempt. | `propose.yaml:2-5,33-75`; `recipe.ps1:12,23` | workflow run |
| FR-52 | `propose` shall fail when a non-merge commit between `test` and the head, not already on `release`, has a subject outside `<type>(<scope>): <text>` with the type and scope from `$CommitTypes` and `$CommitScopes`, or lacks a `Signed-off-by: Name <email>` trailer. | `propose.yaml:77-142`; `recipe.ps1:21-22` | workflow run |
| FR-53 | When a pull request is opened or reopened and the earlier steps pass, `propose` shall request a review from the `$Reviewers` other than the poster and comment that the gate passed; without the bot token it shall only warn. | `propose.yaml:154-193`; `recipe.ps1:24` | workflow run |
| FR-54 | On a push to `test`, daily by cron `0 3 * * *` and on manual dispatch, `test` shall check the runner (at least 5 GB free on `C:`; `python`, `ruff`, `pytest` and `mypy` answer), run the offline checks (FR-50), build the wheel, and regenerate the wiki with `tools/wiki.py`. A wiki that cannot be pushed is a warning, not a failure. | `test.yaml:2-7,32-60,73-110` | workflow run |
| FR-55 | On a pull request into `release`, `release-gate` shall fail unless the head branch is `test`, then run the offline checks and the build. The install and uninstall proof runs only for a recipe with `$App`. | `release-gate.yaml:2-5,16-85` | workflow run |
| FR-56 | `release-gate` shall read `tests/manual/tests.json`. With an empty list it shall set the commit status `manual/verdict` to success ("no manual tests defined"). Otherwise it shall post the tester's instructions and wait up to 150 minutes for a verdict or the owner's override. In both cases it stores `manual-report.md` in the generic package `<name>-manual-reports/<sha>/`. The list in this tree is empty. | `release-gate.yaml:87-150,204-221,435-446`; `tests/manual/tests.json:1-4` | workflow run |
| FR-57 | On a push to `release`, `release` shall take the version from `$Version` and publish nothing when tag `v<version>` exists. Otherwise it shall build, write release notes from the commits since the last `v*` tag and the issues and pull requests they reference, open an issue when `release` changed past the gate without a `Why:` line, and create release `v<version>` with an installer `.exe` from `dist\` (if one exists), `manual-report.md` and `sbom.md` (recipe metadata and `uv.lock`) attached. | `release.yaml:2-4,45-60,72-336` | workflow run |
| FR-58 | `release` shall close the open issues whose work branches the release merges, merge `release` into `test` (or open or update the issue "test is behind release" when that fails), and put the release notes on top of `CHANGELOG.md` on `test` as the commit `docs(docs): changelog for v<version>`. | `release.yaml:93-116,338-422` | workflow run; commit `42206b3` on `test` is such a changelog commit |
| FR-59 | Every Monday by cron `0 4 * * 1`, `sweep` shall delete remote work branches already merged into `test`, comment on open pull requests idle for 30 days and close those idle for 60. | `sweep.yaml:2-4,23-80` | workflow run |
| FR-60 | The `commit-msg` hook shall refuse a message that starts with a UTF-8 BOM, a subject outside `<type>(<scope>): <text>` (case-sensitive, with the lists from `recipe.ps1`), or a message without a `Signed-off-by` trailer. A `Merge …` subject during `git merge` passes. | `.claude/hooks/git/commit-msg:1-3`; `.claude/hooks/git/commit-msg.ps1:14-53` | no test |
| FR-61 | The `pre-push` hook shall run `git lfs pre-push` when Git LFS is installed, refuse branch names outside the branch grammar (`test` and `from-*` exempt), refuse deleting `release`, and refuse a push to an existing `release` unless one pushed commit carries `$OverridePhrase` alone on a line and a `Why:` line. | `.claude/hooks/git/pre-push:3-7`; `.claude/hooks/git/pre-push.ps1:8-96` | no test |
| FR-62 | The Claude Code `PreToolUse` hook shall deny a tool call it cannot read, a recursive delete outside `$env:TEMP`, a force push (`--force`, `-f`, `--mirror`, a `+ref`; `--force-with-lease` passes), and on `test` or `release` a `reset --hard` (except to `origin/<branch>` or the upstream), `clean -f`, `commit --amend`, `rebase` or `filter-branch`. | `.claude/settings.json:10-12`; `.claude/hooks/gate.ps1:34-36,131-182` | no test |
| FR-63 | The Claude Code hooks shall append one JSON line per event (time, event, session, tool, and a detail cut to 200 characters or the gate's decision) to `.claude/logs/ai.jsonl`, rotate that file at session start when it exceeds 10 MB, and set `core.hooksPath` at session start. | `.claude/settings.json:7-19`; `.claude/hooks/log.ps1:1-26`; `.claude/hooks/session-start-stop.ps1:25-48`; `gate.ps1:6-23` | no test |

### 3.10 Helper scripts

| ID | Requirement | Source | Verification |
| --- | --- | --- | --- |
| FR-64 | `scripts/build_release_artifact.py` shall build a wheel and an sdist with `python -m build`, copy them into `release/`, install the wheel without dependencies into a throwaway venv and import `xray_classifier` (unless `--skip-verify`), and write a `<name>.sha256` file per artefact. It needs the `build` package, which is the `release` extra. | `scripts/build_release_artifact.py:38-114`; `pyproject.toml:36-38` | no test |
| FR-65 | `scripts/collect_quality_metrics.py` shall measure test pass rate, coverage, radon complexity / maintainability index minus lint findings, and portability (operating systems named in `.gitea/workflows/*.y*ml` and Python versions named with `>=` in `requires-python`), print an overall mean, optionally write JSON (`--output`) and exit 1 below `--fail-under`. | `scripts/collect_quality_metrics.py:55-216` | no test |
| FR-66 | `tools/wiki.py` shall build wiki pages from the docstrings and tag comments of the packages in `$Provides`. With `--out` it writes them to a folder; with `--push`, `--wiki-url` and a token it pushes them, exiting 1 without URL or token, when the wiki clone fails, or after three failed attempts; otherwise it lists the page names. | `tools/wiki.py:2-12,336-342,362-363,368-404` | workflow run (`test.yaml:85,106`) |
| FR-67 | `tests/manual/serve.py` shall serve the manual-test page on `127.0.0.1:8765` and forward calls under `/api/v1/` and `/api/packages/` to the server of the `origin` remote, adding `GITEA_TOKEN` from the environment or else from `.claude/.env`. | `tests/manual/serve.py:1-8,23-25,40-62` | no test |

### 3.11 Declared but not implemented

| Declared behaviour | What the code does | Where declared |
| --- | --- | --- |
| Honour the command-line options | `main.py` parses `--epochs`, `--batch-size`, `--weights` and `--count` (`main.py:23-41`), but nothing reads them: `run_pipeline` loops `range(1)` at the default batch size 32 (`engine.py:209,218`) and `visualize_attention.py` hardcodes `classifier_0.pt` and `count=10` (`visualize_attention.py:12,39`); done when `train --epochs 2 --batch-size 8` trains two epochs at batch size 8, `explain --weights other.pt --count 3` loads `other.pt` and writes at most three maps, and the option table in `docs/manual.md:17-24` is removed. | Jira KAN-60 (task proposed 2026-11-17) |
| Save the trained model and evaluate from a weights file | No command writes weights, because nothing calls `ModelWrapper.save` (`models.py:56-58`), and `eval` re-runs the whole training pipeline (`main.py:50-53`); done when `train` saves its state dict to `classifier_0.pt`, the file `explain` loads by default, and `eval --weights <file>` loads that file and only runs `evaluate_and_sample`. | Jira KAN-60 (task proposed 2026-11-17) |
| End-to-end pipeline test on a fixture dataset | `TrainingEngine` and `AttentionVisualizer` have no tests and the coverage floor is 9 % (`pyproject.toml:107-108`; the gap is recorded in `docs/quality-iso25010.md:34-36`); done when a test builds a tiny train/val/test image folder under pytest's `tmp_path`, runs `train_epoch`, `validate` and `evaluate_and_sample` on the small `ViTConfig` model offline, asserts the report keys and the written sample files, and the floor is raised to what the suite then measures. | Jira KAN-60 (task proposed 2026-11-17) |
| Make the profiling script profile this project | `scripts/profile_application.py` came from a template and imports `quality_template.core` (`scripts/profile_application.py:22`), which this repository does not have, so it stops with `ModuleNotFoundError`; done when it profiles this project's own code offline (for example `ViTTransform` with a stand-in processor as in `benchmarks/test_preprocessing_performance.py:28-36`) and runs to completion, or when it is deleted and `README.md:34` is updated. | Jira KAN-60 (task proposed 2026-11-17) |
| Pre-trained weights `xray_vit_classifier_v0.1.0.pt` as a release asset | No code produces weights; whether the asset exists was not checked | `CHANGELOG.md:71` |
| Half precision (FP16) | No `half()` or `autocast` anywhere | `CHANGELOG.md:65`, `docs/architecture.md:10` |
| GPU memory monitoring in the UI | The panel shows CUDA ON/OFF and the device type (`cuda`/`cpu`) only (`engine.py:194-200`) | `docs/architecture.md:11` |
| Backbone swap without changing core logic | Protocols exist (`models.py:9-17`, `data.py:14-23`); `run_pipeline` and `visualize_attention.py` hard-code `ModelWrapper`/`DatasetManager`; the engine reads `ImageFolder.samples` and `outputs.logits` | `docs/architecture.md:15` |
| Console entry point in the wheel | No `[project.scripts]` in `pyproject.toml`; `docs/install.md:13` says `main.py` is not in the wheel | `scripts/build_release_artifact.py:12-13` |
| Accuracy target > 80 % | No test or check asserts it (`docs/quality-iso25010.md:34-36` states the gap) | `docs/architecture.md:7` |
| Scheduled secret scan of the full git history | No workflow in `.gitea/workflows/` scans for secrets; commit `8db1bde` deleted `.github/workflows/secret-scan.yml` | `docs/quality-iso25010.md:106-108` |
| Pinned model revision hash | Stated there as not implemented; `from_pretrained` gets no `revision` (`models.py:32-35`) | `docs/quality-iso25010.md:114-115` |
| A committed `.env.example` listing the variable names | No such file in the tree (`git ls-files`) | `SECURITY.md:28-29` |

## 4. Interfaces

### 4.1 Command line (`main.py`)

Started as `.\recipe.ps1 run <command> [options]` (`docs/manual.md:3-5`, FR-35).

| Command | Argument | Type, default | Used by the code |
| --- | --- | --- | --- |
| *(none)* | — | — | prints help, exit 0 |
| `train` | `--epochs` | int, 1 | log line only |
| `train` | `--batch-size` | int, 32 | no |
| `eval` | `--weights` | str, `classifier_0.pt` | log line only |
| `explain` | `--weights` | str, `classifier_0.pt` | log line only |
| `explain` | `--count` | int, 10 | log line only |

Source: `main.py:16-65`.

### 4.2 Recipe verbs and helper scripts

| Command | Arguments | Exit codes | Source |
| --- | --- | --- | --- |
| `.\recipe.ps1 <verb>` | one of `build`, `install`, `uninstall`, `clean`, `run`, `publish`, `info`, `test`; `run` passes the rest to `main.py` | `build`, `run`: the exit code of uv; `install`, `uninstall`, `clean`, `info`: 0; `publish`: 1 without `LAB_SHARE` or after a failed build, else 0; `test`: 0 when every step passes, 1 at the first failure | `recipe.ps1:1-118` |
| `scripts/build_release_artifact.py` | `--version` (default `v0.0.0-dev`, log only), `--skip-verify` | 0; `SystemExit` with message if `build` is missing, nothing was produced, or the wheel does not import; an uncaught `CalledProcessError` if `python -m build`, venv creation or the wheel install fails | `scripts/build_release_artifact.py:38-82,85-114` |
| `scripts/collect_quality_metrics.py` | `--output PATH`, `--fail-under FLOAT` | 0; 1 below threshold | `scripts/collect_quality_metrics.py:168-216` |
| `scripts/profile_application.py` | `--paragraphs`, `--sort`, `--save` (docstring) | fails at import with `ModuleNotFoundError: No module named 'quality_template'` | `scripts/profile_application.py:3-5,22` |
| `tools/wiki.py` | `--push`, `--wiki-url`, `--token`, `--sha`, `--out` | 0; 1 if `--push` lacks a URL or a token, the wiki clone fails, or the push fails three times | `tools/wiki.py:336-342,362-363,377-404` |
| `tests/manual/serve.py` | none | exits with a message if git fails, `origin` is not `http(s)://<server>/<owner>/<repository>`, no `GITEA_TOKEN` is found, or port 8765 is taken or cannot be bound | `tests/manual/serve.py:30-49,168-180` |

### 4.3 Files read and written

| File | Direction | Format | Source |
| --- | --- | --- | --- |
| `Bone_Fracture_Binary_Classification/{train,val,test}/<class>/*` | read | Image files in torchvision `ImageFolder` layout, one sub-folder per class | `data.py:72-78` |
| `classifier_0.pt` | read | PyTorch state dict of `ViTForImageClassification` with a 2-output head, loadable with `weights_only=True` | `visualize_attention.py:24-27` |
| `prediction_samples/correct_<0-2>.png`, `incorrect_<0-2>.png` | written | RGB PNG, original test image with text overlay | `engine.py:171-186` |
| `attention_maps/error_<index>_t<true>_p<pred>.png` | written | Matplotlib PNG, two panels | `interpretability.py:92-121` |
| `dist/*.whl` | written | Wheel of `src/xray_classifier` | `recipe.ps1:49,69` |
| `release/*.whl`, `release/*.tar.gz`, `release/*.sha256` | written | Distributions; checksum line `<sha256-hex>  <file name>\n` | `scripts/build_release_artifact.py:92-109` |
| `quality-report.json` (any `--output` path) | written | JSON: `standard`, `measured_characteristics[]` (each with `characteristic`, `score`, evidence fields), `assessed_by_hand[]`, `overall_measured_score`, `note` | `scripts/collect_quality_metrics.py:188-211` |
| `.claude/.env` | read | `NAME=value` lines; git-ignored | `recipe.ps1:34-43`; `tests/manual/serve.py:52-62`; `.gitignore:24` |
| `.claude/logs/ai.jsonl` | written | One JSON object per line, UTF-8 without BOM; git-ignored | `.claude/hooks/log.ps1:4-24`; `.gitignore:17` |
| `tests/manual/tests.json` | read | `{"version": 1, "tests": [...]}`, entries as in FR-49 | `tests/test_manual_list.py:24-51`; `release-gate.yaml:107-137` |
| `CHANGELOG.md` on `test` | written by `release` | Newest release notes first | `release.yaml:402-422` |

### 4.4 External interfaces

| Peer | Call | Source |
| --- | --- | --- |
| Hugging Face Hub (via `transformers`) | `ViTImageProcessor.from_pretrained` and `ViTForImageClassification.from_pretrained(..., output_attentions=True)` for `google/vit-base-patch16-224` | `models.py:32-35`, `engine.py:203`, `visualize_attention.py:11` |
| Gitea API (`<server>/api/v1`, `<server>/api/packages`) | Review requests, comments, commit statuses, releases and assets, generic packages, issues, branch deletion; the wiki through git | `propose.yaml:154-193`, `release-gate.yaml:139-150,435-440`, `release.yaml:279-336`, `sweep.yaml:23-80`, `test.yaml:73-110` |
| Lab share | Copy of the wheel to `$LAB_SHARE\needs\<name>\v<version>` | `recipe.ps1:66-75` |

## 5. Data

| Entity | Representation | Source |
| --- | --- | --- |
| Image sample | `(path, class_index)` in `ImageFolder.samples`; tensor `(3, H, W)` after `ViTTransform` (224 × 224 in the tests) | `data.py:36-45,76-78`; `tests/test_core.py:28,37` |
| Class list | `list[str]` from the train split's folder names | `data.py:80` |
| Split statistics | `dict[str, int]` with keys `Training`, `Validation`, `Test` | `data.py:88-92` |
| Prediction sample | `dict` with `path`, `true`, `pred` (class names) | `engine.py:151-155` |
| Metrics | scikit-learn `classification_report(..., output_dict=True)` dictionary | `engine.py:166-168` |
| Model weights | `state_dict` of the full ViT model; held in memory only during `train`/`eval` | `models.py:56-58`; `visualize_attention.py:25-27` |
| Recipe metadata | One literal per line (`$Name`, `$Code`, `$Version`, `$Summary`, `$Needs`, `$Tools`, `$Provides`, the commit and branch vocabularies, `$Reviewers`, `$OverridePhrase`), read as text by the hooks and workflows | `recipe.ps1:9-25` |
| Manual test entry | `id`, `title`, `expected`, `steps[]`, optional `issue`, `blocking`, `needs` | `tests/test_manual_list.py:36-51` |
| Hook event | JSON line with `ts`, `event`, `session`, `tool`, `detail` or `decision` | `.claude/hooks/log.ps1:4-13`; `gate.ps1:8-11` |

Storage is the local file system only. There is no database and no network
storage. Committed sample outputs: 6 files in `prediction_samples/`, 10 in
`attention_maps/`, 2 in `docs/assets/` (`git ls-files`), stored through Git LFS
(`.gitattributes:9`).

## 6. Non-functional requirements

| ID | Requirement | Source |
| --- | --- | --- |
| NFR-1 | Computation shall use a CUDA device when present and fall back to the CPU. On Windows the locked `torch` is the CPU build from PyPI, so training runs on the CPU there. | `engine.py:38`, `interpretability.py:33`, `visualize_attention.py:19`, `pyproject.toml:23-24`, `docs/install.md:7` |
| NFR-2 | Weight files shall be loaded with `torch.load(..., weights_only=True)`. | `visualize_attention.py:26` |
| NFR-3 | Truncated image files shall not abort loading. | `data.py:11` |
| NFR-4 | `src`, `main.py`, `visualize_attention.py` and `tools` shall pass `mypy` in strict mode (Python 3.13 target). Only `sklearn.*` and `torchvision.*` are exempt from missing-import errors. | `pyproject.toml:118-130`, `recipe.ps1:105` |
| NFR-5 | Source code shall pass ruff 0.16.1 with rule sets E, W, I, N, UP, B, C4, SIM, RUF, S (bandit), TID, ARG, PT, line length 100, target `py313`; `S101` ignored. `scripts/*` additionally ignore `S404`, `S603`, `S607`, `S310` (the list also names `PLR0912`, `C901`, `ANN`, `D`, which are not selected), and `tests/test_manual_list.py` ignores `E501` and `PT018`. `.claude`, `tests/manual/serve.py` and `tools/wiki.py` are excluded. | `pyproject.toml:61-98` |
| NFR-6 | Source code shall be formatted with `ruff format` with LF line endings. The recipe formats in place, and the gates fail on any resulting diff (FR-50). | `pyproject.toml:100-101`, `recipe.ps1:109-111` |
| NFR-7 | Text files shall be checked out with LF line endings, `*.ps1`, `*.bat` and `*.cmd` with CRLF, and PDF, PNG, JPG, ZIP and 7z files shall be stored through Git LFS. | `.gitattributes:1-13` |
| NFR-8 | No key shall be committed: `.claude/.env` and `.env` are git-ignored, and so is `.claude/logs/`, which holds prompt text. | `.gitignore:16-24,40`, `CLAUDE.md:20-26` |
| NFR-9 | The test suite shall never download model weights. | `tests/conftest.py:6`, `recipe.ps1:100-101`, `tests/test_no_weights_are_downloaded.py:6-7`, `tests/test_core.py:43-49` |
| NFR-10 | Every artefact of `build_release_artifact.py` shall be accompanied by a SHA-256 checksum file, and the wheel shall be shown to install and import, unless `--skip-verify` is given. The `release` workflow does not run this script. | `scripts/build_release_artifact.py:56-82,102-109`, `.gitea/workflows/release.yaml:72-74` |
| NFR-11 | The package shall support CPython `>=3.13,<3.14`; `uv.lock` is resolved for 3.13 only. Only Windows is exercised, by every workflow. | `pyproject.toml:21`, `uv.lock:3`, `.gitea/workflows/*.yaml` (`runs-on: windows`) |
| NFR-12 | Every declared dependency and the build backend shall be pinned with `==`, and the recipe shall use the lock file unchanged (`uv run --frozen`). | `pyproject.toml:10,22-52`, `recipe.ps1:63,103-110` |

The application code defines no timeouts, retries, rate limits, memory limits
or performance targets. The workflows set job timeouts: 60 minutes for
`propose`, `test`, the `release-gate` checks and `release`, 170 minutes for the
manual gate, which waits up to 150 minutes for a verdict, and 15 minutes for
`sweep` (`propose.yaml:15`, `test.yaml:17`, `release-gate.yaml:19,91,101`,
`release.yaml:14`, `sweep.yaml:14`). The benchmark figures in
`docs/quality-iso25010.md:46-50` are measurements, not requirements.

## 7. Constraints and dependencies

| Package | Declared (`pyproject.toml`) | Locked (`uv.lock`) | Group |
| --- | --- | --- | --- |
| torch | `==2.11.0` | 2.11.0 | runtime |
| torchvision | `==0.26.0` | 0.26.0 | runtime |
| transformers | `==5.5.4` | 5.5.4 | runtime |
| datasets | `==4.8.4` | 4.8.4 | runtime (not imported by any module) |
| scikit-learn | `==1.8.0` | 1.8.0 | runtime |
| rich | `==15.0.0` | 15.0.0 | runtime |
| pillow | `==12.2.0` | 12.2.0 | runtime |
| matplotlib | `==3.10.8` | 3.10.8 | runtime (imported by `interpretability.py:4`) |
| huggingface-hub | not declared | 1.33.0 | transitive, through `transformers`, `datasets` and `tokenizers`; imported by `tests/test_no_weights_are_downloaded.py:3` |
| build | `==1.3.0` | 1.3.0 | optional extra `release` |
| ruff | `==0.16.1` | 0.16.1 | dev |
| mypy | `==2.3.0` | 2.3.0 | dev |
| pytest | `==8.4.2` | 8.4.2 | dev |
| pytest-cov | `==5.0.0` | 5.0.0 | dev |
| coverage | `==7.15.2` | 7.15.2 | dev |
| pytest-benchmark | `==5.2.3` | 5.2.3 | dev |
| radon | `==6.0.1` | 6.0.1 | dev (used by `scripts/collect_quality_metrics.py:102-103`) |

Build backend: `setuptools==80.9.0` (`pyproject.toml:7-11`). `[tool.uv]
package = false`: uv does not install the project itself
(`pyproject.toml:57-59`). Workflow actions: `actions/checkout@v4`,
`actions/upload-artifact@v3` (`.gitea/workflows/*.yaml`). Licence: Apache 2.0
(`pyproject.toml:20`, `LICENSE`).

Hard constraints in the code: the model id and data root are constants
(`engine.py:203,208`, `visualize_attention.py:11-13`). The attention grid
assumes 196 patches (14 × 14), i.e. a 224-pixel input with 16-pixel patches
(`interpretability.py:88-89`). Commands must be run from the directory that
holds the dataset folder and the weights file, because both are relative paths;
`recipe.ps1` runs every verb from the repository root (`recipe.ps1:45`). The
recipe and the workflows are PowerShell; the recipe needs `uv`
(`recipe.ps1:49,63,103-110`), and the committed images need Git LFS
(`docs/install.md:4`).

## 8. Acceptance

Automated checks (sources in brackets):

```powershell
.\recipe.ps1 test                                               # FR-43 to FR-49   (recipe.ps1:84-113)
git diff --exit-code                                            # what the gates check after it (FR-50)
uv run --frozen python -m pytest benchmarks --benchmark-only    # 4 benchmark cases (benchmarks/test_preprocessing_performance.py:12)
```

`.\recipe.ps1 test` verifies FR-10, FR-12 to FR-15, FR-47 to FR-49, NFR-4 to
NFR-6 and NFR-9, and exercises FR-43 to FR-46 by running. No test needs network
access: `from_pretrained` is patched (`tests/test_core.py:43-49`) and the hub is
offline (FR-47). `pyproject.toml:108` adds `--cov --cov-fail-under=9` to every
pytest run, the benchmark run included. FR-50 to FR-59 are verified by the
workflow runs themselves. None of these commands was run for this document.

Manual checks for the remaining requirements (dataset in
`Bone_Fracture_Binary_Classification/` at the repository root, `uv sync
--frozen` done):

| Check | Expected | FRs |
| --- | --- | --- |
| `.\recipe.ps1 run` | Help text, exit code 0 | FR-1, FR-2, FR-35 |
| `.\recipe.ps1 run train` | Environment panel, Dataset Statistics table, progress bar, one `Epoch 0 \| …` line, Final Metrics table; `prediction_samples/` recreated with up to 3 `correct_*` and 3 `incorrect_*` PNGs | FR-3, FR-5 to FR-9, FR-17 to FR-27, FR-34 |
| `.\recipe.ps1 run eval` | Same output as `train` | FR-3 |
| `.\recipe.ps1 run explain` without `classifier_0.pt` | Log line `Could not load weights from classifier_0.pt: …`, no new files | FR-4, FR-28 |
| `.\recipe.ps1 run explain` with a compatible `classifier_0.pt` | Up to 10 `attention_maps/error_<i>_t<t>_p<p>.png` two-panel images, or "No errors found to analyze." | FR-29 to FR-33 |
| `.\recipe.ps1 info` | Name, code, version, summary and provided packages; exit 0 | FR-40 |
| `.\recipe.ps1 build` | A wheel in `dist\` | FR-38 |
| `.\recipe.ps1 publish` with `LAB_SHARE` unset | "LAB_SHARE is not set, so there is nowhere to publish to", exit 1 | FR-42 |
| `python scripts/build_release_artifact.py --version v1.1.0`, with the `release` extra | `release/` contains a `.whl`, a `.tar.gz` and a `.sha256` for each; "the wheel installs and imports as xray_classifier" | FR-64, NFR-10 |
| `python scripts/collect_quality_metrics.py --output quality-report.json` | Four scores and an overall score printed; JSON written | FR-65 |
| `python tools/wiki.py --out <folder>` | `wiki: wrote <n> pages to <folder>` | FR-66 |
| `python tests/manual/serve.py`, with `GITEA_TOKEN` set | The manual-test page at `http://127.0.0.1:8765/` | FR-67 |

## 9. Open questions

1. Where does `Bone_Fracture_Binary_Classification/` come from? Its licence,
   class names, split sizes and image sizes are not documented in this tree;
   `docs/install.md:11` says only where to put it.
2. How were the metrics in `README.md:20-26` (83 % accuracy) obtained?
   `README.md:18` calls them one run recorded before the repository came to the
   lab, and no script or result file records it. `docs/adr/ADR-001-Model-Selection.md:18`
   cites the same 83 % as a comparison with ResNet, which the repository does
   not contain.
3. Where does `classifier_0.pt` come from? Partly answered by git history:
   earlier versions of the code wrote it. Commit `34ed173` saved
   `classifier_{epoch}.pt` from `main.py:144-145` at that commit, and commit
   `e144bab` saved it from `run_pipeline` when the validation loss improved
   (`src/xray_classifier/engine.py:170-172` at that commit). Commit `f36f1f3`
   removed the save. With one epoch the file name is `classifier_0.pt`. Still
   open: does the release asset `xray_vit_classifier_v0.1.0.pt`
   (`CHANGELOG.md:71`) exist? It was not checked.
4. Who are the intended users, and what is the intended use (teaching,
   research, other)? The code states no clinical use and no regulatory status.
5. Target hardware, GPU memory, and acceptable training or inference time are
   not stated. `docs/adr/ADR-001-Model-Selection.md:22` says a GPU is needed
   for efficient training, while the locked `torch` on Windows, the only CI
   platform, is the CPU build (`docs/install.md:7`).
6. Are `--epochs`, `--batch-size`, `--weights` and `--count` meant to be wired
   through, and is `eval` meant to load weights instead of training? Partly
   answered: Jira KAN-60 carries both as tasks to do ([3.11](#311-declared-but-not-implemented)).
7. Is `datasets` still needed? It is declared (`pyproject.toml:28`) but no
   module imports it. The former question about `matplotlib` and `radon` is
   answered on this branch: both are declared (`pyproject.toml:32-33,51`).
8. Should `__version__` follow `$Version`? `recipe.ps1:13` and
   `pyproject.toml:16` say `1.1.0` and `recipe.ps1 test` keeps them equal, but
   `src/xray_classifier/__init__.py:3` still says `0.1.0`, a value unchanged
   since commit `e144bab`. Tag `v1.1.0` points at `7c56d4d` on `release` and is
   not on `test`.
9. Is `dev` still a working branch? `docs/branching.md:3` names only `release`
   and `test` as long-lived, and `docs/branching.md:14` says the GitHub `dev`
   is kept as the tag `import/dev`; the GitHub remote still has a branch `dev`
   at that tag (`git branch -r`).
10. Is Linux or macOS a supported platform? Every workflow runs on Windows
    only, and `uv.lock:4-8` resolves for `win32` and other platforms.
11. Is the deletion of `prediction_samples/` on every run (FR-24) intended,
    given that the directory is committed and `README.md:26` presents its
    images as the results of the recorded run?
12. Where are users meant to get the wheel? The `release` workflow attaches no
    wheel (out of scope in §1), `recipe.ps1 publish` copies it to the lab share
    only, and `scripts/build_release_artifact.py` is not called by any
    workflow.
