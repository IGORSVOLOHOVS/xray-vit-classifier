# Manual

```powershell
.\recipe.ps1 run <command> [options]
```

Run it from the root of the repository, with the dataset in place (see `docs/install.md`).

| command | what it does |
| --- | --- |
| `train` | load the ViT, train the head for one epoch, validate, then evaluate on the test split and print the classification report |
| `eval` | the same run as `train`: pure evaluation is not written yet |
| `explain` | load `classifier_0.pt` from the root, find the test images the model got wrong, and draw an attention map for up to 10 of them into `attention_maps/` |

`train` and `eval` empty `prediction_samples/` first, then write up to three correct and three incorrect samples there, marked with the true and the predicted class.

The options are accepted, and the program does not act on them yet:

| option | on | status |
| --- | --- | --- |
| `--epochs <n>` | `train` | printed in the log; the run is always one epoch |
| `--batch-size <n>` | `train` | not read; the batch size is 32 |
| `--weights <file>` | `eval`, `explain` | printed in the log; `explain` always reads `classifier_0.pt` |
| `--count <n>` | `explain` | printed in the log; `explain` always draws up to 10 |

No command writes a weights file. `explain` needs a `classifier_0.pt` made outside this program, and without one it logs that it could not load the weights and stops.
