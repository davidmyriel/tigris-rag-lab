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
.venv/bin/python dataset.py create-forks-all --source-bucket wiki-dataset || true

echo
echo "==> Ingesting: wiki-naive"
.venv/bin/python ingest.py \
  --bucket wiki-dataset-naive \
  --manifest-key manifests/exp-wiki-naive-v1.json

echo
echo "==> Ingesting: wiki-semantic"
.venv/bin/python ingest.py \
  --bucket wiki-dataset-semantic \
  --manifest-key manifests/exp-wiki-semantic-v1.json

echo
echo "==> Ingesting: wiki-window"
.venv/bin/python ingest.py \
  --bucket wiki-dataset-window \
  --manifest-key manifests/exp-wiki-window-v1.json

echo
echo "==> Ingesting: wiki-sentence"
.venv/bin/python ingest.py \
  --bucket wiki-dataset-sentence \
  --manifest-key manifests/exp-wiki-sentence-v1.json

echo
echo "==> Ingesting: wiki-markdown"
.venv/bin/python ingest.py \
  --bucket wiki-dataset-markdown \
  --manifest-key manifests/exp-wiki-markdown-v1.json

echo
echo "==> Ingesting: wiki-recursive"
.venv/bin/python ingest.py \
  --bucket wiki-dataset-recursive \
  --manifest-key manifests/exp-wiki-recursive-v1.json

echo
echo "==> Running eval"
.venv/bin/python eval.py

echo
echo "✅ Tigris RAG Lab run complete."


