from __future__ import annotations

from pathlib import Path
import textwrap

import faiss
import numpy as np
from PIL import Image, ImageDraw


def build_image_index(image_embeddings: np.ndarray) -> faiss.IndexFlatIP:
    """Build an exact cosine-similarity index for L2-normalized embeddings."""
    embeddings = np.ascontiguousarray(image_embeddings.astype("float32"))
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index


def search_images(
    index: faiss.IndexFlatIP,
    query_embedding: np.ndarray,
    top_k: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the top-k image indices and cosine similarities for one query."""
    query = np.ascontiguousarray(query_embedding.astype("float32"))
    if query.ndim == 1:
        query = query[None, :]
    scores, indices = index.search(query, min(top_k, index.ntotal))
    return scores[0], indices[0]


def cross_modal_recall_at_k(
    image_embeddings: np.ndarray,
    caption_embeddings: np.ndarray,
    caption_to_image: np.ndarray,
    ks: tuple[int, ...] = (1, 5, 10),
) -> tuple[dict[int, float], dict[int, float]]:
    """
    Compute symmetric image-to-text and text-to-image Recall@K.

    Each caption has one relevant image. Each image can have multiple relevant
    captions; a retrieval hit occurs if any caption paired with that image is
    present in the top-k results.
    """
    similarities = image_embeddings @ caption_embeddings.T
    n_images = image_embeddings.shape[0]

    image_to_text: dict[int, float] = {}
    image_ranking = np.argsort(-similarities, axis=1)
    for k in ks:
        k_eff = min(k, caption_embeddings.shape[0])
        hits = []
        for image_index in range(n_images):
            retrieved_caption_indices = image_ranking[image_index, :k_eff]
            relevant = caption_to_image[retrieved_caption_indices] == image_index
            hits.append(bool(relevant.any()))
        image_to_text[k] = float(np.mean(hits))

    text_to_image: dict[int, float] = {}
    text_ranking = np.argsort(-similarities.T, axis=1)
    for k in ks:
        k_eff = min(k, n_images)
        hit = (
            text_ranking[:, :k_eff]
            == caption_to_image[:, None]
        ).any(axis=1)
        text_to_image[k] = float(hit.mean())

    return image_to_text, text_to_image


def save_montage(
    images: list[Image.Image],
    captions: list[str],
    scores: list[float],
    output_path: str | Path,
    thumb_size: int = 220,
) -> None:
    """Save a compact qualitative text-to-image retrieval montage."""
    margin = 12
    caption_h = 92
    width = len(images) * (thumb_size + margin) + margin
    height = thumb_size + caption_h + 2 * margin
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)

    for i, (image, caption, score) in enumerate(zip(images, captions, scores)):
        x = margin + i * (thumb_size + margin)
        y = margin
        thumb = image.copy()
        thumb.thumbnail((thumb_size, thumb_size))

        tile = Image.new("RGB", (thumb_size, thumb_size), "white")
        tx = (thumb_size - thumb.width) // 2
        ty = (thumb_size - thumb.height) // 2
        tile.paste(thumb, (tx, ty))
        canvas.paste(tile, (x, y))

        wrapped = "\n".join(textwrap.wrap(caption, width=32)[:3])
        label = f"{wrapped}\nscore={score:.3f}"
        draw.multiline_text((x, y + thumb_size + 4), label, fill="black", spacing=2)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
