from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoProcessor, CLIPModel


class ClipEncoder:
    """Small wrapper around a pretrained CLIP image/text encoder."""

    def __init__(self, model_name: str, device: str = "auto") -> None:
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = torch.device(device)
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def encode_images(
        self,
        images: Sequence[Image.Image],
        batch_size: int = 32,
    ) -> np.ndarray:
        chunks: list[torch.Tensor] = []

        for start in range(0, len(images), batch_size):
            batch = list(images[start : start + batch_size])
            inputs = self.processor(images=batch, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(self.device)

            vision_output = self.model.vision_model(pixel_values=pixel_values)
            features = self.model.visual_projection(vision_output.pooler_output)
            features = F.normalize(features, p=2, dim=-1)
            chunks.append(features.cpu())

        return torch.cat(chunks, dim=0).numpy().astype("float32")

    @torch.inference_mode()
    def encode_texts(
        self,
        texts: Sequence[str],
        batch_size: int = 64,
    ) -> np.ndarray:
        chunks: list[torch.Tensor] = []

        for start in range(0, len(texts), batch_size):
            batch = list(texts[start : start + batch_size])
            inputs = self.processor(
                text=batch,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            text_inputs = {
                key: value.to(self.device)
                for key, value in inputs.items()
                if key in {"input_ids", "attention_mask"}
            }

            text_output = self.model.text_model(**text_inputs)
            features = self.model.text_projection(text_output.pooler_output)
            features = F.normalize(features, p=2, dim=-1)
            chunks.append(features.cpu())

        return torch.cat(chunks, dim=0).numpy().astype("float32")
