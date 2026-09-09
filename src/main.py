from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from clip_model import ClipEncoder
from data import load_flickr8k_subset
from retrieval import (
    build_image_index,
    cross_modal_recall_at_k,
    save_montage,
    search_images,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cross-modal image-text retrieval with pretrained CLIP and FAISS."
    )
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--split", choices=["train", "dev", "test"], default="test")
    parser.add_argument("--max-images", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--text-batch-size", type=int, default=128)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, ...")
    parser.add_argument(
        "--model",
        default="openai/clip-vit-base-patch32",
        help="Hugging Face CLIP checkpoint.",
    )
    parser.add_argument(
        "--query",
        default="a dog running through snow",
        help="Free-text query used for qualitative text-to-image retrieval.",
    )
    parser.add_argument(
        "--output",
        default="outputs/retrieval.jpg",
        help="Path for the retrieval montage.",
    )
    return parser.parse_args()


def print_metrics(title: str, metrics: dict[int, float]) -> None:
    print(f"\n{title}")
    for k, value in metrics.items():
        print(f"Recall@{k}: {value:.3f}")


def main() -> None:
    args = parse_args()

    dataset = load_flickr8k_subset(
        cache_dir=args.data_dir,
        split=args.split,
        max_images=args.max_images,
        seed=args.seed,
    )

    images = [dataset.image(i) for i in range(len(dataset))]
    captions, caption_to_image = dataset.all_captions()
    caption_to_image_array = np.asarray(caption_to_image, dtype=np.int64)

    print(
        f"Loaded {len(images)} images and {len(captions)} human-written captions "
        f"from Flickr8k ({args.split} split)."
    )

    encoder = ClipEncoder(args.model, device=args.device)
    image_embeddings = encoder.encode_images(images, batch_size=args.batch_size)
    caption_embeddings = encoder.encode_texts(
        captions,
        batch_size=args.text_batch_size,
    )

    image_to_text, text_to_image = cross_modal_recall_at_k(
        image_embeddings=image_embeddings,
        caption_embeddings=caption_embeddings,
        caption_to_image=caption_to_image_array,
        ks=(1, 5, 10),
    )

    print_metrics("Image -> text retrieval", image_to_text)
    print_metrics("Text -> image retrieval", text_to_image)

    image_index = build_image_index(image_embeddings)
    query_embedding = encoder.encode_texts([args.query])
    scores, retrieved = search_images(image_index, query_embedding, top_k=args.top_k)

    print(f'\nFree-form text -> image retrieval for: "{args.query}"')
    retrieved_images = []
    retrieved_captions = []
    retrieved_scores = []

    for rank, (score, image_index) in enumerate(zip(scores, retrieved), start=1):
        reference_caption = dataset.captions(int(image_index))[0]
        print(f"{rank:>2}. cosine={score:.3f} | {reference_caption}")
        retrieved_images.append(dataset.image(int(image_index)))
        retrieved_captions.append(reference_caption)
        retrieved_scores.append(float(score))

    save_montage(
        retrieved_images,
        retrieved_captions,
        retrieved_scores,
        Path(args.output),
    )
    print(f"\nSaved qualitative retrieval montage to {args.output}")


if __name__ == "__main__":
    main()
