## Tigris RAG Lab

![lab](lab.png)

This repo is a small, self‑contained **RAG experimentation lab**.

It shows how to:
- keep a text dataset in **Tigris** with **snapshots and forks** (so you can branch your data like git), and  
- build **multiple Qdrant indexes** over those forks, each with different preprocessing / chunking, and  
- run a **repeatable evaluation** to compare those indexes side by side.

### Who this is for

- People building **RAG systems** who want more discipline around:
  - dataset versioning,
  - trying different chunking strategies,
  - swapping embedding configs,
  - and being able to reproduce “that good run from last week”.
- Anyone who already has Qdrant + S3‑compatible storage and wants a **minimal pattern** for:
  - “dataset branch” → “index build” → “eval” as one clean loop.

### What this lab demonstrates

- **Tigris as a branching filesystem for datasets**
  - Base bucket: `product-dataset` with a handful of `p_i/review.txt` documents.  
  - Fork buckets: `product-dataset-naive`, `product-dataset-heading`, `product-dataset-semantic`, created from snapshots.
  - Each fork can have its own **manifest** describing how to turn that data into an index.

- **Qdrant as a branching index**
  - For each fork + manifest, `ingest.py` builds a separate Qdrant collection:
    - `products-naive` – one vector per review.  
    - `products-heading` – alternative strategy (still one per review here).  
    - `products-semantic` – paragraph‑level chunks per review.
  - All points carry payload linking them back to:
    - dataset bucket + snapshot,  
    - manifest + hash,  
    - chunker + embedding model.

- **A tiny, code‑level eval loop**
  - `eval.py` runs a few hard‑coded queries and prints Hit@5 for each collection.  
  - You can extend this into a proper eval harness (more queries, labels, metrics) without changing the control flow.

The goal is not to be “complete”, but to give you a **clear, working skeleton** you can adapt to your own datasets, chunkers, and embedding models.

### 1. Install

```bash
cd /Users/errant/qdrant-ingestion

python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install boto3 python-dotenv qdrant-client fastembed tigris-boto3-ext
```

Create `.env`:

```bash
ACCESS_KEY=...
SECRET_ACCESS_KEY=...
S3_ENDPOINT=https://t3.storage.dev
QDRANT_URL=...
QDRANT_KEY=...
```

### 2. One-command run

```bash
chmod +x run_all.sh      # first time only
./run_all.sh
```

This will:
- ensure Tigris forks exist (`dataset.py create-forks-all`)
- build three Qdrant collections from manifests (`ingest.py`)
- run a small eval over all three (`eval.py`)

### 3. Quick query example

```python
from qdrant_client import QdrantClient, models
from dotenv import load_dotenv
import os

load_dotenv()

client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_KEY"),
)

res = client.query_points(
    collection_name="products-naive",
    query=models.Document(
        text="Phones with improved design",
        model="sentence-transformers/all-MiniLM-L6-v2",
    ),
    using="text_embedding",
    limit=3,
)

for p in res.points:
    print(p.id, p.score, p.payload.get("folder"))
```

### 4. Files in this repo (what they do)

- `dataset.py`  
  - CLI to manage **Tigris** buckets, snapshots, and forks.  
  - Subcommands:
    - `create-bucket` – create `product-dataset` with snapshots enabled.  
    - `create-snapshot` – create a named snapshot for `product-dataset`.  
    - `create-fork` – create a fork bucket from a specific snapshot version.  
    - `create-forks-all` – create snapshots + forks for `product-dataset-naive` and `product-dataset-semantic`.

- `ingest.py`  
  - Reads a **manifest JSON** from a Tigris bucket (under `manifests/exp-*.json`).  
  - Creates a **Qdrant collection** with named vector `text_embedding`.  
  - Ingests documents from the chosen fork bucket, using either:
    - full-doc mode (`chunker: "none"`) or  
    - paragraph-chunk mode (`chunker: "paragraph"`).  
  - Attaches payload fields so every point is traceable back to dataset + manifest.

- `eval.py`  
  - Runs a small Hit@5 evaluation against:
    - `products-naive`  
    - `products-heading`  
    - `products-semantic`  
  - Uses a few hard-coded queries and checks whether the expected `folder` appears in the top‑K results.

- `run_all.sh`  
  - One-button pipeline:
    1. Ensures forks exist via `dataset.py create-forks-all`.  
    2. Builds all three Qdrant collections via `ingest.py`.  
    3. Runs `eval.py` and prints scores.

