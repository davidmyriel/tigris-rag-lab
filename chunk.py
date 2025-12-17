"""
Pre-chunk documents and store chunks in Tigris fork bucket.

This script reads raw documents from a source bucket, chunks them according to
the strategy specified in the manifest, and stores the chunks as JSONL files
in the fork bucket.

Usage:
    .venv/bin/python chunk.py --bucket <fork-bucket> --manifest-key <manifest-key>
"""

import argparse
import json
import os
import re
from typing import Dict, Any, List

import boto3
from dotenv import load_dotenv


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


def fetch_manifest(s3_client, bucket: str, manifest_key: str) -> Dict[str, Any]:
    """Fetch manifest from bucket."""
    resp = s3_client.get_object(Bucket=bucket, Key=manifest_key)
    data = resp["Body"].read()
    manifest = json.loads(data.decode("utf-8"))
    return manifest


def fetch_text(s3_client, bucket: str, key: str) -> str:
    """Fetch text content from S3."""
    resp = s3_client.get_object(Bucket=bucket, Key=key)
    return resp["Body"].read().decode("utf-8")


def list_objects(s3_client, bucket: str, prefix: str = ""):
    """List object keys with given prefix."""
    paginator = s3_client.get_paginator("list_objects_v2")
    pages = paginator.paginate(Bucket=bucket, Prefix=prefix)
    for page in pages:
        if "Contents" in page:
            for obj in page["Contents"]:
                yield obj["Key"]


def iter_documents_from_layout(s3_client, bucket: str, manifest: Dict[str, Any]):
    """Iterate over documents from the layout specified in manifest."""
    doc_layout = manifest.get("doc_layout") or {}
    layout_type = doc_layout.get("type", "folders_text")

    if layout_type != "folders_text":
        raise ValueError(
            f"Unsupported doc_layout.type '{layout_type}'. "
            "For now only 'folders_text' is supported."
        )

    folders = doc_layout.get("folders", ["source"])
    file_name = doc_layout.get("file_name", "review.txt")
    
    # Get source bucket from manifest
    source_bucket = manifest.get("dataset_bucket", bucket)

    for folder in folders:
        prefix = f"{folder}/"
        for key in list_objects(s3_client, source_bucket, prefix):
            if not key.endswith(file_name):
                continue
            text = fetch_text(s3_client, source_bucket, key)
            yield folder, key, text


def chunk_text(text: str, chunker: str, **chunker_params) -> List[Dict[str, Any]]:
    """Chunk text according to strategy and return list of chunk dicts."""
    chunks = []
    
    if chunker == "none":
        # No chunking - one chunk per document
        chunks.append({
            "text": text,
            "chunk_index": 1,
            "start_char": 0,
            "end_char": len(text),
        })
    
    elif chunker == "paragraph":
        # Split on blank lines
        paragraphs = re.split(r'\n\s*\n', text)
        for idx, para in enumerate(paragraphs, 1):
            para = para.strip()
            if para:
                chunks.append({
                    "text": para,
                    "chunk_index": idx,
                    "start_char": text.find(para),
                    "end_char": text.find(para) + len(para),
                })
    
    elif chunker == "sentence":
        # Split on sentence boundaries
        sentences = re.split(r'([.!?]+(?:\s+|$))', text)
        current = ""
        idx = 1
        for part in sentences:
            current += part
            if part.strip() and re.search(r'[.!?]', part):
                chunk = current.strip()
                if chunk:
                    chunks.append({
                        "text": chunk,
                        "chunk_index": idx,
                        "start_char": text.find(chunk),
                        "end_char": text.find(chunk) + len(chunk),
                    })
                    idx += 1
                current = ""
        if current.strip():
            chunks.append({
                "text": current.strip(),
                "chunk_index": idx,
                "start_char": text.find(current.strip()),
                "end_char": text.find(current.strip()) + len(current.strip()),
            })
    
    elif chunker == "markdown":
        # Split on markdown headers
        lines = text.split('\n')
        current_chunk = []
        idx = 1
        
        for line in lines:
            if re.match(r'^#{1,6}\s+', line):
                # New header - save previous chunk
                if current_chunk:
                    chunk_text = '\n'.join(current_chunk).strip()
                    if chunk_text:
                        chunks.append({
                            "text": chunk_text,
                            "chunk_index": idx,
                            "start_char": text.find(chunk_text),
                            "end_char": text.find(chunk_text) + len(chunk_text),
                        })
                        idx += 1
                current_chunk = [line]
            else:
                current_chunk.append(line)
        
        # Add final chunk
        if current_chunk:
            chunk_text = '\n'.join(current_chunk).strip()
            if chunk_text:
                chunks.append({
                    "text": chunk_text,
                    "chunk_index": idx,
                    "start_char": text.find(chunk_text),
                    "end_char": text.find(chunk_text) + len(chunk_text),
                })
    
    elif chunker == "window":
        # Sliding window chunking
        window_size = chunker_params.get("window_size", 500)
        window_stride = chunker_params.get("window_stride", 250)
        
        pos = 0
        idx = 1
        while pos < len(text):
            chunk_text = text[pos:pos + window_size]
            if chunk_text.strip():
                chunks.append({
                    "text": chunk_text.strip(),
                    "chunk_index": idx,
                    "start_char": pos,
                    "end_char": min(pos + window_size, len(text)),
                })
                idx += 1
            pos += window_stride
    
    elif chunker == "recursive":
        # Recursive hierarchical splitting
        max_chunk_size = chunker_params.get("max_chunk_size", 1000)
        chunk_overlap = chunker_params.get("chunk_overlap", 200)
        
        def recursive_split(text: str, max_size: int) -> List[str]:
            if len(text) <= max_size:
                return [text]
            
            # Try splitting on paragraphs first
            paragraphs = re.split(r'\n\s*\n', text)
            if len(paragraphs) > 1:
                result = []
                for para in paragraphs:
                    result.extend(recursive_split(para.strip(), max_size))
                return result
            
            # Then try sentences
            sentences = re.split(r'([.!?]+(?:\s+|$))', text)
            if len(sentences) > 1:
                result = []
                current = ""
                for part in sentences:
                    if len(current + part) > max_size and current:
                        result.append(current.strip())
                        current = part
                    else:
                        current += part
                if current.strip():
                    result.append(current.strip())
                return result
            
            # Finally, split by character
            result = []
            pos = 0
            while pos < len(text):
                result.append(text[pos:pos + max_size])
                pos += max_size - chunk_overlap
            return result
        
        split_texts = recursive_split(text, max_chunk_size)
        for idx, chunk_text in enumerate(split_texts, 1):
            if chunk_text.strip():
                chunks.append({
                    "text": chunk_text.strip(),
                    "chunk_index": idx,
                    "start_char": text.find(chunk_text),
                    "end_char": text.find(chunk_text) + len(chunk_text),
                })
    
    else:
        raise ValueError(f"Unknown chunker: {chunker}")
    
    return chunks


def prechunk_from_manifest(s3_client, bucket: str, manifest_key: str, manifest: Dict[str, Any]) -> None:
    """Pre-chunk documents and store chunks in fork bucket."""
    preprocess = manifest.get("preprocess", {}) or {}
    chunker = (preprocess.get("chunker") or "none").lower()
    chunker_params = {k: v for k, v in preprocess.items() if k != "chunker" and k != "chunker_version"}
    
    total_chunks = 0
    total_docs = 0
    
    for folder, key, text in iter_documents_from_layout(s3_client, bucket, manifest):
        # Chunk the document
        chunks = chunk_text(text, chunker, **chunker_params)
        
        if not chunks:
            continue
        
        total_docs += 1
        
        # Store chunks as JSONL in fork bucket
        # Format: chunks/{folder}/{file_name}.jsonl
        chunks_key = f"chunks/{folder}/{os.path.basename(key)}.jsonl"
        
        # Create JSONL content
        jsonl_lines = []
        for chunk in chunks:
            chunk_data = {
                "text": chunk["text"],
                "chunk_index": chunk["chunk_index"],
                "source_folder": folder,
                "source_key": key,
                "start_char": chunk.get("start_char", 0),
                "end_char": chunk.get("end_char", len(chunk["text"])),
            }
            jsonl_lines.append(json.dumps(chunk_data))
        
        jsonl_content = "\n".join(jsonl_lines) + "\n"
        
        # Upload chunks to fork bucket
        s3_client.put_object(
            Bucket=bucket,
            Key=chunks_key,
            Body=jsonl_content.encode("utf-8"),
        )
        
        total_chunks += len(chunks)
        print(f"✓ Chunked {key}: {len(chunks)} chunks -> {chunks_key}")
    
    print(f"\n✓ Pre-chunking complete: {total_docs} document(s), {total_chunks} chunk(s) total")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Pre-chunk documents and store chunks in Tigris fork bucket"
    )
    p.add_argument(
        "--bucket",
        required=True,
        help="Fork bucket name (where chunks will be stored)",
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
    
    # Fetch manifest
    print(f"Fetching manifest: {args.bucket}/{args.manifest_key}")
    manifest = fetch_manifest(s3_client, args.bucket, args.manifest_key)
    
    # Pre-chunk and store
    print(f"Pre-chunking documents...")
    prechunk_from_manifest(s3_client, args.bucket, args.manifest_key, manifest)
    
    print("✓ Done")


if __name__ == "__main__":
    main()

