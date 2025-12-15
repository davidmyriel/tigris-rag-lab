"""
Build a Qdrant collection from a Tigris bucket + manifest.

Examples:
  .venv/bin/python ingest.py \\
      --bucket product-dataset-naive \\
      --manifest-key manifests/exp-naive-v1.json
"""

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List

import boto3
from dotenv import load_dotenv
from qdrant_client import QdrantClient, models


def load_env() -> None:
    """Load environment variables from .env if present."""
    load_dotenv()


def make_s3_client():
    endpoint = os.getenv("S3_ENDPOINT")
    access_key = os.getenv("ACCESS_KEY")
    secret_key = os.getenv("SECRET_ACCESS_KEY")

    if not endpoint or not access_key or not secret_key:
        raise RuntimeError(
            "Missing S3 configuration. Make sure S3_ENDPOINT, ACCESS_KEY, and "
            "SECRET_ACCESS_KEY are set in your environment."
        )

    return boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
    )


def make_qdrant_client() -> QdrantClient:
    url = os.getenv("QDRANT_URL")
    key = os.getenv("QDRANT_KEY")

    if not url or not key:
        raise RuntimeError(
            "Missing Qdrant configuration. Make sure QDRANT_URL and QDRANT_KEY "
            "are set in your environment."
        )

    return QdrantClient(url=url, api_key=key)


def fetch_manifest(
    s3_client, bucket: str, manifest_key: str
) -> Dict[str, Any]:
    resp = s3_client.get_object(Bucket=bucket, Key=manifest_key)
    data = resp["Body"].read()
    manifest = json.loads(data.decode("utf-8"))
    return manifest


def compute_manifest_hash(manifest: Dict[str, Any]) -> str:
    # Stable JSON representation for hashing
    encoded = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def ensure_collection(client: QdrantClient, manifest: Dict[str, Any]) -> str:
    index_cfg = manifest.get("index", {}) or {}
    collection_name = index_cfg.get("collection_name")
    if not collection_name:
        # Minimal fallback naming if not provided
        dataset_bucket = manifest.get("dataset_bucket", "dataset")
        exp_name = index_cfg.get("experiment_name", "exp")
        ts = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
        collection_name = f"rag_{dataset_bucket}_{exp_name}_{ts}"

    embedding_cfg = manifest.get("embedding", {}) or {}
    dim = embedding_cfg.get("dim") or embedding_cfg.get("size") or 384
    distance_name = (embedding_cfg.get("distance") or "cosine").upper()

    try:
        distance = getattr(models.Distance, distance_name)
    except AttributeError:
        distance = models.Distance.COSINE

    if client.collection_exists(collection_name):
        client.delete_collection(collection_name)

    client.create_collection(
        collection_name=collection_name,
        vectors_config={
            "text_embedding": models.VectorParams(
                size=int(dim),
                distance=distance,
            )
        },
    )

    print(
        f"✓ Created collection '{collection_name}' "
        f"(dim={dim}, distance={distance_name})"
    )
    return collection_name


def list_objects(s3_client, bucket: str, prefix: str) -> List[str]:
    """List object keys under a prefix (single page, minimal)."""
    resp = s3_client.list_objects_v2(Bucket=bucket, Prefix=prefix)
    return [obj["Key"] for obj in resp.get("Contents", [])]


def fetch_text(s3_client, bucket: str, key: str) -> str:
    body = s3_client.get_object(Bucket=bucket, Key=key)["Body"].read()
    return body.decode("utf-8")


def iter_documents_from_layout(
    s3_client, bucket: str, manifest: Dict[str, Any]
):
    """
    Minimal loader for the current repo layout.

    Supports:
    - doc_layout.type == "folders_text" with:
        - "folders": ["p_1", "p_2", ...]
        - "file_name": "review.txt" (default)
    Falls back to the existing p_1..p_10/review.txt convention.
    """
    doc_layout = manifest.get("doc_layout") or {}
    layout_type = doc_layout.get("type", "folders_text")

    if layout_type != "folders_text":
        raise ValueError(
            f"Unsupported doc_layout.type '{layout_type}'. "
            "For now only 'folders_text' is supported."
        )

    folders = doc_layout.get("folders")
    file_name = doc_layout.get("file_name", "review.txt")

    if not folders:
        # Fallback to current convention p_1..p_10
        folders = [f"p_{i}" for i in range(1, 11)]

    for folder in folders:
        prefix = f"{folder}/"
        for key in list_objects(s3_client, bucket, prefix):
            if not key.endswith(file_name):
                continue
            text = fetch_text(s3_client, bucket, key)
            yield folder, key, text


def ingest_from_manifest(
    client: QdrantClient,
    s3_client,
    bucket: str,
    manifest_key: str,
    manifest: Dict[str, Any],
    collection_name: str,
) -> None:
    manifest_hash = compute_manifest_hash(manifest)

    dataset_bucket = manifest.get("dataset_bucket", bucket)
    dataset_snapshot = manifest.get("dataset_snapshot")

    preprocess = manifest.get("preprocess", {}) or {}
    chunker_version = preprocess.get("chunker_version")

    embedding_cfg = manifest.get("embedding", {}) or {}
    embed_model = embedding_cfg.get(
        "model", "sentence-transformers/all-MiniLM-L6-v2"
    )

    point_id = 1
    all_points: List[models.PointStruct] = []

    chunker = (preprocess.get("chunker") or "none").lower()

    for folder, key, text in iter_documents_from_layout(
        s3_client, bucket, manifest
    ):
        if chunker == "paragraph":
            # Paragraph-based chunking: split on blank lines and index each chunk.
            raw_parts = text.split("\n\n")
            chunks = [p.strip() for p in raw_parts if p.strip()]

            if not chunks:
                continue

            for local_idx, chunk in enumerate(chunks, start=1):
                payload = {
                    "chunk_index": local_idx,
                    "text": chunk,
                    "folder": folder,
                    "source_key": key,
                    "dataset_bucket": dataset_bucket,
                    "dataset_snapshot": dataset_snapshot,
                    "manifest_key": manifest_key,
                    "manifest_hash": manifest_hash,
                    "chunker_version": chunker_version,
                    "embed_model": embed_model,
                }

                all_points.append(
                    models.PointStruct(
                        id=point_id,
                        vector={
                            "text_embedding": models.Document(
                                text=chunk,
                                model=embed_model,
                            )
                        },
                        payload=payload,
                    )
                )
                point_id += 1
        elif chunker == "window":
            # Sliding-window chunking over raw text (character-based).
            window_size = int(preprocess.get("window_size", 500))
            stride = int(preprocess.get("window_stride", window_size // 2 or 1))

            if window_size <= 0:
                window_size = 500
            if stride <= 0:
                stride = window_size

            text_len = len(text)
            local_idx = 0
            pos = 0
            while pos < text_len:
                chunk = text[pos : pos + window_size].strip()
                if chunk:
                    local_idx += 1
                    payload = {
                        "chunk_index": local_idx,
                        "text": chunk,
                        "folder": folder,
                        "source_key": key,
                        "dataset_bucket": dataset_bucket,
                        "dataset_snapshot": dataset_snapshot,
                        "manifest_key": manifest_key,
                        "manifest_hash": manifest_hash,
                        "chunker_version": chunker_version,
                        "embed_model": embed_model,
                        "window_size": window_size,
                        "window_stride": stride,
                    }

                    all_points.append(
                        models.PointStruct(
                            id=point_id,
                            vector={
                                "text_embedding": models.Document(
                                    text=chunk,
                                    model=embed_model,
                                )
                            },
                            payload=payload,
                        )
                    )
                    point_id += 1

                pos += stride
        else:
            # Naive / heading-style ingestion: one point per document.
            payload = {
                "review": text,
                "folder": folder,
                "source_key": key,
                "dataset_bucket": dataset_bucket,
                "dataset_snapshot": dataset_snapshot,
                "manifest_key": manifest_key,
                "manifest_hash": manifest_hash,
                "chunker_version": chunker_version,
                "embed_model": embed_model,
            }

            all_points.append(
                models.PointStruct(
                    id=point_id,
                    vector={
                        "text_embedding": models.Document(
                            text=text,
                            model=embed_model,
                        )
                    },
                    payload=payload,
                )
            )
            point_id += 1

    if not all_points:
        print("No documents found to ingest.")
        return

    client.upsert(collection_name=collection_name, points=all_points)
    print(f"✓ Ingested {len(all_points)} point(s) into '{collection_name}'")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Build a Qdrant collection from a Tigris bucket + manifest.\n\n"
            "Example:\n"
            "  python ingest.py "
            "--bucket product-dataset-naive "
            "--manifest-key manifests/exp-naive-v1.json"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument(
        "--bucket",
        required=True,
        help="Tigris bucket name (e.g. product-dataset-naive)",
    )
    p.add_argument(
        "--manifest-key",
        required=True,
        help="S3 key to the manifest JSON (e.g. manifests/exp-naive-v1.json)",
    )
    return p.parse_args()


def main() -> None:
    load_env()
    args = parse_args()

    s3_client = make_s3_client()
    qdrant = make_qdrant_client()

    print(
        f"Using bucket='{args.bucket}', manifest='{args.manifest_key}'"
    )

    manifest = fetch_manifest(s3_client, args.bucket, args.manifest_key)
    manifest_hash = compute_manifest_hash(manifest)
    print(f"✓ Loaded manifest (sha256={manifest_hash[:8]}...)")

    collection_name = ensure_collection(qdrant, manifest)
    ingest_from_manifest(
        qdrant,
        s3_client,
        args.bucket,
        args.manifest_key,
        manifest,
        collection_name,
    )

    print("\nDone.")


if __name__ == "__main__":
    main()



