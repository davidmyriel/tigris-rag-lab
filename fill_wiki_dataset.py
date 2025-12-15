"""
Populate a Tigris bucket called `wiki-dataset` with a small sample of Wikipedia articles.

This script is intended to be run locally in your venv, not in a sandboxed environment.

Usage:
  .venv/bin/pip install datasets
  .venv/bin/python fill_wiki_dataset.py
"""

from datasets import load_dataset
import os
import boto3
from dotenv import load_dotenv


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


def main() -> None:
    load_env()
    s3 = make_s3_client()

    bucket = "wiki-dataset"

    print("Loading a small English Wikipedia sample...")
    # Use the modern config name; adjust date if needed
    ds = (
        load_dataset("wikimedia/wikipedia", "20231101.en", split="train")
        .shuffle(seed=42)
        .select(range(10))
    )

    for i, row in enumerate(ds, start=1):
        title = (row.get("title") or "").strip()
        text = (row.get("text") or "").strip()
        if not text:
            continue

        body = f"# {title}\n\n{text}" if title else text
        key = f"p_{i}/review.txt"
        s3.put_object(Bucket=bucket, Key=key, Body=body.encode("utf-8"))
        print(f"Uploaded {key} ({len(body)} chars)")

    print("\n✓ Uploaded 10 Wikipedia articles into wiki-dataset/p_*/review.txt")


if __name__ == "__main__":
    main()


