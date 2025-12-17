"""
Build a Qdrant collection from pre-chunked JSONL files in a Tigris fork bucket.

This version reads pre-chunked JSONL files instead of chunking on-the-fly.
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


def fetch_manifest(s3_client, bucket: str, manifest_key: str) -> Dict[str, Any]:
    resp = s3_client.get_object(Bucket=bucket, Key=manifest_key)
    data = resp["Body"].read()
    manifest = json.loads(data.decode("utf-8"))
    return manifest


def compute_manifest_hash(manifest: Dict[str, Any]) -> str:
    encoded = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def ensure_collection(client: QdrantClient, manifest: Dict[str, Any]) -> str:
    index_cfg = manifest.get("index", {}) or {}
    collection_name = index_cfg.get("collection_name")
    if not collection_name:
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

    print(f"✓ Created collection '{collection_name}' (dim={dim}, distance={distance_name})")
    return collection_name


def list_objects(s3_client, bucket: str, prefix: str) -> List[str]:
    """List object keys under a prefix."""
    resp = s3_client.list_objects_v2(Bucket=bucket, Prefix=prefix)
    return [obj["Key"] for obj in resp.get("Contents", [])]


def iter_chunks_from_prechunked(s3_client, bucket: str, manifest: Dict[str, Any]):
    """Iterate over pre-chunked JSONL files in the fork bucket."""
    doc_layout = manifest.get("doc_layout") or {}
    folders = doc_layout.get("folders", ["source"])
    file_name = doc_layout.get("file_name", "review.txt")
    
    for folder in folders:
        # Look for chunks in chunks/{folder}/{file_name}.jsonl
        chunks_key = f"chunks/{folder}/{file_name}.jsonl"
        
        try:
            # Fetch the JSONL file
            resp = s3_client.get_object(Bucket=bucket, Key=chunks_key)
            jsonl_content = resp["Body"].read().decode("utf-8")
            
            # Parse each line as JSON
            for line in jsonl_content.strip().split("\n"):
                if not line.strip():
                    continue
                chunk_data = json.loads(line)
                yield chunk_data
        except s3_client.exceptions.NoSuchKey:
            print(f"⚠ No pre-chunked file at {chunks_key}")
            continue


def ingest_from_manifest(
    client: QdrantClient,
    s3_client,
    bucket: str,
    manifest_key: str,
    manifest: Dict[str, Any],
    collection_name: str,
) -> None:
    """Ingest pre-chunked documents into Qdrant."""
    manifest_hash = compute_manifest_hash(manifest)
    
    dataset_bucket = manifest.get("dataset_bucket", bucket)
    dataset_snapshot = manifest.get("dataset_snapshot")
    
    preprocess = manifest.get("preprocess", {}) or {}
    chunker_version = preprocess.get("chunker_version")
    
    embedding_cfg = manifest.get("embedding", {}) or {}
    embed_model = embedding_cfg.get("model", "sentence-transformers/all-MiniLM-L6-v2")
    
    point_id = 1
    all_points: List[models.PointStruct] = []
    
    # Read pre-chunked chunks from fork bucket
    for chunk_data in iter_chunks_from_prechunked(s3_client, bucket, manifest):
        text = chunk_data.get("text", "")
        if not text:
            continue
        
        payload = {
            "chunk_index": chunk_data.get("chunk_index", 1),
            "text": text,
            "source_folder": chunk_data.get("source_folder", "source"),
            "source_key": chunk_data.get("source_key", ""),
            "dataset_bucket": dataset_bucket,
            "dataset_snapshot": dataset_snapshot,
            "manifest_key": manifest_key,
            "manifest_hash": manifest_hash,
            "chunker_version": chunker_version,
            "embed_model": embed_model,
        }
        
        # Add any additional metadata from chunk_data
        if "start_char" in chunk_data:
            payload["start_char"] = chunk_data["start_char"]
        if "end_char" in chunk_data:
            payload["end_char"] = chunk_data["end_char"]
        
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
        print("No chunks found to ingest.")
        return
    
    client.upsert(collection_name=collection_name, points=all_points)
    print(f"✓ Ingested {len(all_points)} point(s) into '{collection_name}'")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build a Qdrant collection from pre-chunked JSONL files in a Tigris fork bucket"
    )
    p.add_argument(
        "--bucket",
        required=True,
        help="Fork bucket name (where pre-chunked files are stored)",
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
    qdrant_client = make_qdrant_client()
    
    print(f"Fetching manifest: {args.bucket}/{args.manifest_key}")
    manifest = fetch_manifest(s3_client, args.bucket, args.manifest_key)
    
    print("Creating Qdrant collection...")
    collection_name = ensure_collection(qdrant_client, manifest)
    
    print("Ingesting pre-chunked documents...")
    ingest_from_manifest(
        qdrant_client, s3_client, args.bucket, args.manifest_key, manifest, collection_name
    )
    
    print("✓ Done")


if __name__ == "__main__":
    main()

