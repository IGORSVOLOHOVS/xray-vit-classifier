# Install

1. Install [uv](https://docs.astral.sh/uv/) and Python 3.13. The pins are locked for 3.13 only.
2. Install [Git LFS](https://git-lfs.com/): the PNG images are stored in it.
3. Clone this repository and run `uv sync --frozen` at its root.

On Windows the locked `torch` is the CPU build from PyPI, so training runs on the CPU there.

The first `train`, `eval` or `explain` downloads `google/vit-base-patch16-224`, several hundred megabytes, into the Hugging Face cache, so it needs a network connection. `.\recipe.ps1 test` needs none of that: it sets `HF_HUB_OFFLINE`, and a test that reached for the weights would fail instead of fetching them.

The dataset is not in this repository. Put it in a folder called `Bone_Fracture_Binary_Classification` at the root, with `train`, `val` and `test` folders, each holding one folder of images per class. git ignores that folder.

`.\recipe.ps1 build` packs the package into a wheel in `dist\`. `main.py` is not in the wheel: start the program from a clone, with `.\recipe.ps1 run`.
