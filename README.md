# S3 (Tigris) → Qdrant Ingestion — Minimal Text

## Install
```bash
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install boto3 python-dotenv qdrant-client fastembed tigris-boto3-ext
```

## .env
```
ACCESS_KEY=...
SECRET_ACCESS_KEY=...
S3_ENDPOINT=...
QDRANT_URL=...
QDRANT_KEY=...
```

## Bucket layout
- `product-dataset/p_1/review.txt`
- …
- `product-dataset/p_10/review.txt`

## Run

- Create / fork buckets (snapshots enabled):
```bash
.venv/bin/python dataset.py create-bucket --bucket product-dataset
# ... create snapshot(s) + forks as needed ...
```

- Build indexes from manifests:
```bash
.venv/bin/python ingest.py --bucket product-dataset-naive    --manifest-key manifests/exp-naive-v1.json
.venv/bin/python ingest.py --bucket product-dataset-heading  --manifest-key manifests/exp-heading-v1.json
.venv/bin/python ingest.py --bucket product-dataset-semantic --manifest-key manifests/exp-semantic-v1.json
```

- Eval all three collections:
```bash
.venv/bin/python eval.py
```

## Query
```python
from qdrant_client import QdrantClient, models

client = QdrantClient(url="...", api_key="...")

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

