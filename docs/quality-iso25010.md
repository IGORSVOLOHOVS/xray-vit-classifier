# Quality assessment — ISO/IEC 25010

ISO/IEC 25010 defines eight product quality characteristics. Four of them are
**measured** by `scripts/collect_quality_metrics.py`; the other four need
judgement and are argued below, citing those measurements rather than asserting
quality.

Regenerate the numbers with:

```bash
python scripts/collect_quality_metrics.py --output quality-report.json
```

Latest run, 2026-08-05 — overall measured score **78.2**:

| Characteristic | Score | Evidence |
| --- | --- | --- |
| Functional suitability | 100.0 | 7 tests, 0 failures |
| Reliability | 83.5 | 83.5 % branch coverage |
| Maintainability | 79.1 | mean cyclomatic complexity 2.11, maintainability index 79.1, 0 lint findings |
| Portability | 50.0 | one operating system and one interpreter version in CI |

---

## 1. Functional suitability — measured

*Does it do what it claims, correctly and completely?*

The claim is narrow and therefore checkable: binary classification of bone
fractures in X-ray images with a Vision Transformer. `tests/test_core.py` covers
the three pieces that decide whether a prediction is even possible — the image
transform, the model head's label count, and dataset path resolution.

**Known gap:** there is no end-to-end test that trains on a fixture dataset and
asserts an accuracy floor. Pass rate therefore says the parts work, not that the
pipeline learns.

## 2. Performance efficiency — assessed, with measurements

*Time behaviour, resource use, capacity.*

`benchmarks/test_preprocessing_performance.py` measures the transform that runs
on every image in both training and inference, on the CPU, while the GPU waits.
Measured on this machine:

| Input | Median |
| --- | --- |
| 64 × 64 grayscale | 13.5 µs |
| 256 × 256 grayscale | 80.0 µs |
| 1024 × 1024 grayscale | 916 µs |

Sixteen times the pixels costs about eleven times the time, so the cost is
driven by resizing and is roughly linear in input area — as expected, and worth
knowing: feeding full-resolution radiographs straight in makes preprocessing,
not the model, the bottleneck. Downscaling before the loader is the lever.

The model itself is not benchmarked. Loading pretrained weights downloads
several hundred megabytes, so the figure would measure the network rather than
this code.

## 3. Compatibility — assessed

*Co-existence and interoperability.*

The package installs as a normal wheel and holds no global state, so it can sit
alongside other projects in one environment. Interoperability rests on the
HuggingFace `transformers` model interface and on torch tensors, both of which
are the ordinary currency of this ecosystem.

**Known limit:** `torch>=2.0` is a wide range, and torch does break minor
behaviour between versions. The lock file pins what CI actually uses; the range
in `pyproject.toml` is deliberately permissive so the package can be installed
next to an existing torch rather than fighting it.

## 4. Usability — assessed

*Can someone who did not write it get a result?*

`main.py` is a single entry point with `--help`, and `UIHandler` reports
progress through `rich` rather than leaving a silent process. `README.md`
carries a worked example, and `docs/architecture.md` explains where a new data
source would be plugged in.

`attention_maps/` and `prediction_samples/` are committed, so a reader can see
what the model does — including where it is wrong — without running anything.

**Known gap:** there is no GUI. The audience is assumed to be comfortable with a
terminal.

## 5. Reliability — measured

*Does it keep working?*

83.5 % branch coverage. The uncovered branches are concentrated in
`engine.evaluate_and_sample`, the most complex function in the project
(cyclomatic complexity 10), which is exactly where a defect is most likely.
Raising coverage there is the highest-value test work available.

Errors surface rather than being swallowed: `rich.traceback` is installed at the
entry point, so a failure prints a readable stack rather than a bare message.

## 6. Security — assessed

*Confidentiality, integrity, resistance to misuse.*

Ruff runs the `flake8-bandit` (`S`) rules over the whole tree on every push;
current outstanding findings: **0**. A scheduled `secret-scan` workflow searches
the full git history for credentials, not just the working tree.

The project handles medical images, which are sensitive by nature. It never
transmits them: all processing is local, and the only network access is
HuggingFace weight download at model initialisation.

**Known limit:** downloaded weights are trusted implicitly. A pinned revision
hash per model would close that, and is not implemented.

## 7. Maintainability — measured

*Can it be changed safely?*

Mean cyclomatic complexity 2.11 and maintainability index 79.1, with zero
outstanding lint findings. `mypy --strict` passes over the entire package, so
types are checked rather than suggested.

The one outlier is `evaluate_and_sample` at complexity 10 — the natural next
refactor, and the same function that carries the weakest test coverage.

Architecture decisions are recorded in `docs/adr/`, so a future change can read
why rather than guessing.

## 8. Portability — measured

*Where does it run?*

Score 50.0, and the reason is specific: CI exercises **ubuntu-latest with
Python 3.10 only**. `requires-python` claims `>=3.10`, so 3.11, 3.12 and 3.13
are claimed but untested, and Windows and macOS are untested.

This is a deliberate trade rather than an oversight. Every added matrix cell
downloads torch again — hundreds of megabytes and several minutes of runner
time per cell — for a package that is pure Python and whose portability risk
lives entirely in its dependencies. The released wheel is `py3-none-any`, so it
installs anywhere its dependencies do.

**To raise this honestly:** add a Python 3.12 cell on ubuntu first. That is the
cheapest cell that tests a real claim, since 3.12 changed enough for it to
matter.
