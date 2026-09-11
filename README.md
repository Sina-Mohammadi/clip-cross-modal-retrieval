# CLIP Cross-Modal Retrieval

A compact cross-modal retrieval system using pretrained **CLIP** representations to connect natural language and images in a shared embedding space.

The project uses **Flickr8k**, where each image is paired with five human-written captions, and supports:

- image encoding with a CLIP vision encoder;
- text encoding with a CLIP text encoder;
- image-to-text retrieval over free-form captions;
- text-to-image retrieval over paired images;
- bidirectional Recall@K evaluation;
- free-form text search with FAISS; and
- qualitative retrieval output as an image montage.

The implementation keeps the pipeline compact and reproducible while covering the main components of modern vision-language retrieval.

## Architecture

```text
                         shared CLIP embedding space

image --> CLIP vision encoder --> normalized image embeddings ----\
                                                                   |--> cosine similarity matrix
text  --> CLIP text encoder   --> normalized text embeddings  -----/
                                                                   |
                                                                   +--> image-to-text Recall@K
                                                                   |
                                                                   +--> text-to-image Recall@K

normalized image embeddings --> FAISS image index
                                      ^
                                      |
normalized text query ---------------+
                                      |
                                      +--> top-K retrieved images
```

CLIP is pretrained with a contrastive objective that places matching images and text close together in a shared representation space. This project uses the pretrained representation directly rather than retraining CLIP from scratch.

For evaluation, similarities between all image and caption embeddings form a cross-modal similarity matrix used for both retrieval directions. For free-form search, normalized image embeddings are stored in a FAISS index and the normalized text embedding is used as the query.

## Dataset

The project uses the **Flickr8k** image-caption dataset through the Hugging Face `datasets` library. The selected Hugging Face version contains 8,000 images divided into train, development, and test splits, with **five human-written captions per image**.

Example captions in the dataset describe complete visual scenes rather than class labels, including objects, actions, attributes, and context. This makes the evaluation a genuine image-text retrieval problem instead of a reformulation of image classification.

The default run uses a deterministic subset of the test split to keep inference manageable. The subset size can be changed from the command line.

## Repository structure

```text
clip-cross-modal-retrieval/
├── README.md
├── requirements.txt
├── .gitignore
└── src/
    ├── data.py
    ├── clip_model.py
    ├── retrieval.py
    └── main.py
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run

```bash
python src/main.py
```

The first run downloads the selected Flickr8k split and the pretrained CLIP checkpoint.

A smaller CPU-friendly run:

```bash
python src/main.py --max-images 200 --batch-size 16 --device cpu
```

Try a different natural-language search query:

```bash
python src/main.py --query "two people walking beside the water"
```

or:

```bash
python src/main.py --query "a child playing outside"
```

## Evaluation

For a subset containing `N` images, the system evaluates `N` image embeddings against `5N` human-caption embeddings.

### Image -> text retrieval

Each image has five relevant captions. The system ranks all captions by cosine similarity and counts the query as correct at `K` when **at least one of its five paired captions** occurs in the top-K results.

### Text -> image retrieval

Each caption has one paired image. The system ranks all images and counts the caption as correct at `K` when its paired image occurs in the top-K results.

The program reports:

- `Recall@1`
- `Recall@5`
- `Recall@10`

for both retrieval directions.

No performance values are hard-coded in the repository. Metrics are calculated from the selected subset when the program is run.

## Free-form retrieval

After evaluation, the image embeddings are stored in an exact FAISS inner-product index. Because all CLIP embeddings are L2-normalized, inner product is equivalent to cosine similarity.

A user-provided natural-language query is encoded with the CLIP text encoder and used to retrieve the nearest images. The top matches and similarities are printed, and a montage is saved to:

```text
outputs/retrieval.jpg
```

The montage also shows a human-written reference caption for each retrieved image, which helps inspect whether the retrieved visual content is semantically consistent with the query.

## Key concepts demonstrated

### 1. Shared multimodal representations

Images and natural-language descriptions are represented in the same vector space, enabling direct comparison across modalities.

### 2. Image-to-text retrieval

An image can retrieve semantically corresponding free-form descriptions without converting those descriptions into predefined categories.

### 3. Text-to-image retrieval

Natural-language queries can retrieve relevant images without a task-specific classifier for each query.

### 4. Multiple positives per image

Because every image has five valid captions, image-to-text evaluation supports multiple semantically correct textual matches rather than requiring one template prompt.

### 5. Bidirectional retrieval evaluation

Both image-to-text and text-to-image performance are measured, making it possible to assess the quality of the shared embedding space in both directions.

### 6. FAISS similarity search

The same CLIP representations used for evaluation can be indexed for practical nearest-neighbour search over an image collection.

## Limitations

The system uses a pretrained CLIP checkpoint without dataset-specific fine-tuning, and the default experiment evaluates a subset of Flickr8k rather than a large-scale retrieval collection.

Possible extensions include:

- CLIP or adapter fine-tuning on paired image-caption data;
- mean or median retrieval rank in addition to Recall@K;
- hard-negative mining;
- approximate FAISS indices for substantially larger collections;
- reranking with a cross-modal matching model; and
- systematic error analysis by scene type and linguistic complexity.

## Technical stack

`PyTorch` · `Transformers` · `CLIP` · `Hugging Face Datasets` · `vision-language models` · `multimodal embeddings` · `cross-modal retrieval` · `FAISS` · `cosine similarity` · `Recall@K`

## Dataset reference

Flickr8k was introduced in:

> Hodosh, M., Young, P., & Hockenmaier, J. (2013). *Framing Image Description as a Ranking Task: Data, Models and Evaluation Metrics*. Journal of Artificial Intelligence Research, 47, 853–899.
