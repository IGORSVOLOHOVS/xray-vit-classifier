import os

# huggingface_hub reads this once, on import, and nothing has imported it yet: conftest.py comes
# first. A test that reaches for the ViT weights then fails on the spot instead of downloading
# several hundred megabytes, however pytest was started.
os.environ["HF_HUB_OFFLINE"] = "1"
