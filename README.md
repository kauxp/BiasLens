---
title: BiasLens
emoji: 🔍
colorFrom: purple
colorTo: indigo
sdk: gradio
sdk_version: 4.20.0
python_version: "3.10"
app_file: app.py
pinned: false
short_description: Detect cognitive, linguistic & structural bias in text, URLs, and images
---

# BiasLens

Detect cognitive, linguistic, and structural bias in text, URLs, and images.
Powered by **Gemma** (text) + **PaliGemma** (vision) + **FAISS RAG** — no API key required.

## Architecture

```
app.py                   ← Gradio UI entry point
engine/
  input_layer.py         ← Unified text / URL / image context builder
  retrieval_layer.py     ← FAISS evidence retriever (BAAI/bge-base-en-v1.5)
  reasoning_layer.py     ← Gemma + PaliGemma inference
  utils.py               ← HTML/JSON formatting helpers
data/
  faiss.index            ← Pre-built vector index
  metadata.json          ← Knowledge-base documents
scripts/
  build_kb_index.py      ← Rebuild the FAISS index from scratch
```

## Setup

### 1. Clone and enter the directory

```bash
git clone <repo-url>
cd hf_space
```

### 2. Create and activate a virtual environment

```bash
python -m venv biasenv
source biasenv/bin/activate      # Windows: biasenv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download embedding model (offline-first)

The retrieval layer uses `BAAI/bge-base-en-v1.5` with `local_files_only=True`.
Download it once so it is cached locally:

```bash
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-base-en-v1.5')"
```

### 5. (Optional) Rebuild the FAISS index

The `data/` directory ships with a pre-built index. Run this only if you want
to regenerate it after editing the knowledge base in `scripts/build_kb_index.py`:

```bash
python scripts/build_kb_index.py
```

### 6. Hugging Face model access

Gemma models are gated. Accept the terms on the model card and log in:

```bash
huggingface-cli login
```

## Running locally

```bash
python app.py
```

The app starts at `http://0.0.0.0:7860`.

## Deploying to Hugging Face Spaces

1. Create a new Space (SDK: **Gradio**, hardware: **T4** or better recommended).
2. Push this directory — **exclude** `biasenv/` and `__pycache__/` (see `.gitignore`).
3. Set the `HF_TOKEN` secret in the Space settings if the Gemma models require auth.
4. The Space will install `requirements.txt` automatically on first boot.

## Hardware requirements

| Mode | Minimum |
|------|---------|
| CPU-only | 16 GB RAM (slow inference) |
| GPU | 8 GB VRAM (T4 / A10G recommended) |

Models load in `float16` on GPU and `float32` on CPU automatically.
