import torch
from PIL import Image

from complexity.generative.vision_language.pretraining import (
    HuggingFaceImageDataset,
    cosine_schedule,
)


def test_hugging_face_image_dataset_converts_images_and_labels():
    source = [{"image": Image.new("L", (16, 16)), "label": 3}]
    dataset = HuggingFaceImageDataset(
        source, lambda image: torch.zeros(3, image.height, image.width)
    )
    pixels, label = dataset[0]
    assert pixels.shape == (3, 16, 16)
    assert label == 3


def test_vision_pretraining_schedule_warms_up_then_decays():
    assert cosine_schedule(0, warmup_steps=10, total_steps=100, min_ratio=0.05) == 0.1
    assert cosine_schedule(9, warmup_steps=10, total_steps=100, min_ratio=0.05) == 1.0
    assert cosine_schedule(100, warmup_steps=10, total_steps=100, min_ratio=0.05) == 0.05
