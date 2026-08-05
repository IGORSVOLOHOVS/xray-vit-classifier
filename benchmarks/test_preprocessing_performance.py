"""Point 7: benchmarks, so a performance regression shows up as a number.

Preprocessing is the part of this pipeline that runs on every image, in
training and in inference alike, and it runs on the CPU while the GPU waits.
A slow transform therefore caps throughput no matter how fast the model is,
which makes it the piece worth measuring.

The model itself is deliberately not benchmarked here: loading pretrained
weights downloads several hundred megabytes and the result would measure the
network, not the code.

    pytest benchmarks --benchmark-only
    pytest benchmarks --benchmark-only --benchmark-compare
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest
import torch
from PIL import Image

from xray_classifier.data import ViTTransform


def make_processor(batch: int = 1) -> MagicMock:
    """A stand-in for ViTImageProcessor that returns a fixed tensor.

    The benchmark targets this project's own conversion and unwrapping, not
    HuggingFace's resizing, so the processor is held constant.
    """
    processor = MagicMock()
    processor.return_value = {"pixel_values": torch.zeros((batch, 3, 224, 224))}
    return processor


@pytest.mark.parametrize("side", [64, 256, 1024])
def test_transform_scales_with_image_size(benchmark: Any, side: int) -> None:
    """X-rays arrive at wildly different resolutions; the cost should be visible."""
    transform = ViTTransform(make_processor())
    image = Image.new("L", (side, side))

    result = benchmark(transform, image)

    assert result.shape == (3, 224, 224)


def test_transform_on_rgb_input(benchmark: Any) -> None:
    """An image already in RGB skips the conversion - the cheaper path."""
    transform = ViTTransform(make_processor())
    image = Image.new("RGB", (256, 256))

    result = benchmark(transform, image)

    assert result.shape == (3, 224, 224)
