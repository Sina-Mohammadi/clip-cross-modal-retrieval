from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from datasets import Dataset, load_dataset
from PIL import Image


CAPTION_COLUMNS = [f"caption_{i}" for i in range(5)]


@dataclass
class FlickrSubset:
    dataset: Dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def image(self, index: int) -> Image.Image:
        image = self.dataset[index]["image"]
        return image.convert("RGB")

    def captions(self, index: int) -> list[str]:
        row = self.dataset[index]
        return [str(row[column]).strip() for column in CAPTION_COLUMNS]

    def all_captions(self) -> tuple[list[str], list[int]]:
        """
        Flatten all captions and return the paired image index for each caption.

        With five captions per image, caption_to_image contains each image index
        five times.
        """
        captions: list[str] = []
        caption_to_image: list[int] = []

        for image_index in range(len(self)):
            image_captions = self.captions(image_index)
            captions.extend(image_captions)
            caption_to_image.extend([image_index] * len(image_captions))

        return captions, caption_to_image


def load_flickr8k_subset(
    cache_dir: str | Path,
    split: str = "test",
    max_images: int | None = 500,
    seed: int = 42,
) -> FlickrSubset:
    """Load a deterministic Flickr8k subset with five human captions per image."""
    dataset = load_dataset(
        "intro/flickr8k",
        split=split,
        cache_dir=str(cache_dir),
    )
    dataset = dataset.shuffle(seed=seed)

    if max_images is not None:
        dataset = dataset.select(range(min(max_images, len(dataset))))

    return FlickrSubset(dataset=dataset)
