"""
Evaluate Qdrant collections for the Tigris RAG Lab (Wikipedia only).

Collections:
  - wiki-naive     (one vector per article)
  - wiki-semantic  (paragraph chunks)
  - wiki-window    (sliding window chunks)

Usage:
  .venv/bin/python eval.py
"""

from qdrant_client import QdrantClient, models
from dotenv import load_dotenv
import os


WIKI_EXPERIMENTS = [
  "wiki-naive",
  "wiki-semantic",
  "wiki-window",
  "wiki-sentence",
  "wiki-markdown",
  "wiki-recursive",
]

WIKI_QUERIES = [
  {"query": "history of the Roman Empire"},
  {"query": "introduction to quantum mechanics"},
  {"query": "causes of World War II"},
]

K = 5


def eval_wiki(client: QdrantClient, collection_name: str) -> None:
  print(f"\n=== Evaluating {collection_name} (wiki) ===")

  for q in WIKI_QUERIES:
    query_text = q["query"]
    res = client.query_points(
      collection_name=collection_name,
      query=models.Document(
        text=query_text,
        model="sentence-transformers/all-MiniLM-L6-v2",
      ),
      using="text_embedding",
      limit=K,
    )

    print(f"\nQuery: {query_text}")
    print(f"Top-{K} source keys:")
    for p in res.points:
      print(f"  score={p.score:.4f}  source_key={p.payload.get('source_key')}")


def main() -> None:
  load_dotenv()
  client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_KEY"),
  )

  for col in WIKI_EXPERIMENTS:
    if client.collection_exists(col):
      eval_wiki(client, col)
    else:
      print(f"\n[skip] Collection {col} does not exist")


if __name__ == "__main__":
  main()


