"""
Evaluate Qdrant collections for this project.

Runs a tiny hand-crafted Hit@K eval over:
  - products-naive
  - products-heading
  - products-semantic

Usage:
  .venv/bin/python eval.py
"""

from qdrant_client import QdrantClient, models
from dotenv import load_dotenv
import os


# Simple hand-crafted eval over your three experiment collections
EXPERIMENTS = [
    "products-naive",
    "products-heading",
    "products-semantic",
]

# Tiny eval set: adjust these to match your actual expectations
EVAL_QUERIES = [
    {"query": "great battery life", "relevant_folders": ["p_1", "p_3"]},
    {"query": "poor screen quality", "relevant_folders": ["p_5"]},
    {"query": "good camera performance", "relevant_folders": ["p_2", "p_4"]},
]

K = 5  # Hit@K


def eval_collection(client: QdrantClient, collection_name: str) -> None:
    print(f"\n=== Evaluating {collection_name} ===")
    total = 0
    hits = 0

    for q in EVAL_QUERIES:
        total += 1
        query_text = q["query"]
        relevant = set(q["relevant_folders"])

        res = client.query_points(
            collection_name=collection_name,
            query=models.Document(
                text=query_text,
                model="sentence-transformers/all-MiniLM-L6-v2",
            ),
            using="text_embedding",
            limit=K,
        )

        folders = [p.payload.get("folder") for p in res.points]
        hit = any(f in relevant for f in folders if f is not None)
        hits += int(hit)

        print(f"\nQuery: {query_text}")
        print(f"Relevant folders: {sorted(relevant)}")
        print(f"Top-{K} folders: {folders}")
        print(f"Hit@{K}: {bool(hit)}")

    hit_at_k = hits / total if total else 0.0
    print(f"\nSummary for {collection_name}: Hit@{K} = {hit_at_k:.3f}")


def main() -> None:
    load_dotenv()
    client = QdrantClient(
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_KEY"),
    )

    for col in EXPERIMENTS:
        if client.collection_exists(col):
            eval_collection(client, col)
        else:
            print(f"\n[skip] Collection {col} does not exist")


if __name__ == "__main__":
    main()

