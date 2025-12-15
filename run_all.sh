#!/usr/bin/env bash
set -euo pipefail

# Tigris RAG Lab — end-to-end run:
# - ensure forks exist
# - build all Qdrant indexes from manifests
# - run eval
#
# Usage:
#   ./run_all.sh
#
# Assumes:
#   - .venv is created and dependencies are installed
#   - .env is configured with Tigris + Qdrant credentials

cd "$(dirname "$0")"

echo "==> Ensuring forks exist (may print BucketAlreadyExists errors)..."
.venv/bin/python dataset.py create-forks-all --source-bucket product-dataset || true

echo
echo "==> Ingesting: products-naive"
.venv/bin/python ingest.py \
  --bucket product-dataset-naive \
  --manifest-key manifests/exp-naive-v1.json

echo
echo "==> Ingesting: products-heading"
.venv/bin/python ingest.py \
  --bucket product-dataset-heading \
  --manifest-key manifests/exp-heading-v1.json

echo
echo "==> Ingesting: products-semantic"
.venv/bin/python ingest.py \
  --bucket product-dataset-semantic \
  --manifest-key manifests/exp-semantic-v1.json

echo
echo "==> Running eval"
.venv/bin/python eval.py

echo
echo "✅ Tigris RAG Lab run complete."


