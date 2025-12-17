"""
Unified CLI for managing Tigris buckets, snapshots and forks for this project.

Examples:
  # 1) Create base bucket with snapshots enabled
  .venv/bin/python dataset.py create-bucket --bucket wiki-dataset

  # 2) Create a named snapshot for the base bucket
  .venv/bin/python dataset.py create-snapshot \\
      --bucket wiki-dataset \\
      --name wiki-base

  # 3) Create a single fork from a known snapshot version
  .venv/bin/python dataset.py create-fork \\
      --source-bucket wiki-dataset \\
      --snapshot-version <version> \\
      --fork-bucket wiki-dataset-exp

  # 4) (Optional) Create snapshots + forks for multiple wiki strategies
  .venv/bin/python dataset.py create-forks-all \\
      --source-bucket wiki-dataset
"""

import argparse
import os
from typing import Dict, List

import boto3
from dotenv import load_dotenv

try:
    from tigris_boto3_ext import (
        create_snapshot_bucket,
        create_snapshot,
        create_fork,
    )
except ImportError:
    print("Error: tigris-boto3-ext not installed")
    print("Install with: pip install tigris-boto3-ext")
    raise SystemExit(1)


def make_s3_client():
    load_dotenv()

    access_key = os.getenv("ACCESS_KEY")
    secret_access_key = os.getenv("SECRET_ACCESS_KEY")
    endpoint = os.getenv("S3_ENDPOINT", "https://t3.storage.dev")

    if not access_key or not secret_access_key:
        raise RuntimeError(
            "Missing ACCESS_KEY / SECRET_ACCESS_KEY. "
            "Set them in your environment or .env file."
        )

    return boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_access_key,
    )


def cmd_create_bucket(args: argparse.Namespace) -> None:
    s3_client = make_s3_client()
    bucket = args.bucket

    print(f"Creating bucket '{bucket}' with snapshots enabled")
    try:
        create_snapshot_bucket(s3_client, bucket)
        print(f"✓ Bucket '{bucket}' created with snapshots enabled")
    except s3_client.exceptions.BucketAlreadyExists:
        print(f"✓ Bucket '{bucket}' already exists")
    except Exception as e:  # noqa: BLE001
        print(f"✗ Error creating bucket: {e}")


def cmd_create_snapshot(args: argparse.Namespace) -> None:
    s3_client = make_s3_client()
    bucket = args.bucket
    snapshot_name = args.name

    print(f"Creating snapshot '{snapshot_name}' for bucket: {bucket}")
    print("-" * 50)
    try:
        result = create_snapshot(
            s3_client,
            bucket,
            snapshot_name=snapshot_name,
        )
        headers = result.get("ResponseMetadata", {}).get("HTTPHeaders", {})
        snapshot_version = headers.get("x-tigris-snapshot-version")
        print("✓ Snapshot created successfully")
        print(f"  Name:    {snapshot_name}")
        print(f"  Version: {snapshot_version}")
        print(f"  Bucket:  {bucket}")
        if snapshot_version:
            print(
                "\nUse this snapshot version with 'create-fork' or "
                "'create-forks-all'."
            )
    except Exception as e:  # noqa: BLE001
        print(f"✗ Error creating snapshot: {e}")


def cmd_create_fork(args: argparse.Namespace) -> None:
    s3_client = make_s3_client()
    source = args.source_bucket
    fork_bucket = args.fork_bucket
    snapshot_version = args.snapshot_version

    print(f"Creating fork bucket: {fork_bucket}")
    print(f"  Source bucket:   {source}")
    print(f"  Snapshot version: {snapshot_version}")
    print("-" * 50)

    try:
        create_fork(
            s3_client,
            fork_bucket,
            source,
            snapshot_version=snapshot_version,
        )
        print("✓ Fork bucket created successfully")
        print(f"  Fork bucket: {fork_bucket}")
        print(f"  Forked from: {source}@{snapshot_version}")
    except Exception as e:  # noqa: BLE001
        print(f"✗ Error creating fork: {e}")


def cmd_create_forks_all(args: argparse.Namespace) -> None:
    s3_client = make_s3_client()
    source = args.source_bucket

    # Strategies for wiki-dataset
    base_name = source.replace("-dataset", "")
    strategies: List[Dict[str, str]] = [
        {
            "name": "naive",
            "snapshot_name": "naive-chunking",
            "fork_bucket": f"{base_name}-dataset-naive",
        },
        {
            "name": "semantic",
            "snapshot_name": "semantic-chunking",
            "fork_bucket": f"{base_name}-dataset-semantic",
        },
        {
            "name": "window",
            "snapshot_name": "window-chunking",
            "fork_bucket": f"{base_name}-dataset-window",
        },
        {
            "name": "sentence",
            "snapshot_name": "sentence-chunking",
            "fork_bucket": f"{base_name}-dataset-sentence",
        },
        {
            "name": "markdown",
            "snapshot_name": "markdown-chunking",
            "fork_bucket": f"{base_name}-dataset-markdown",
        },
        {
            "name": "recursive",
            "snapshot_name": "recursive-chunking",
            "fork_bucket": f"{base_name}-dataset-recursive",
        },
    ]

    print("Creating snapshots and forks for chunking strategies")
    print(f"Source bucket: {source}")
    print("=" * 60)

    for strat in strategies:
        name = strat["name"]
        snapshot_name = strat["snapshot_name"]
        fork_bucket = strat["fork_bucket"]

        print(f"\n[{name.upper()} CHUNKING]")
        print("-" * 60)

        # 1) Snapshot
        print(f"1. Creating snapshot: {snapshot_name}")
        try:
            snap_res = create_snapshot(
                s3_client,
                source,
                snapshot_name=snapshot_name,
            )
            headers = snap_res.get("ResponseMetadata", {}).get("HTTPHeaders", {})
            snapshot_version = headers.get("x-tigris-snapshot-version")
            print(f"   ✓ Snapshot created: {snapshot_version}")
        except Exception as e:  # noqa: BLE001
            print(f"   ✗ Error creating snapshot: {e}")
            continue

        # 2) Fork
        print(f"2. Creating fork bucket: {fork_bucket}")
        try:
            create_fork(
                s3_client,
                fork_bucket,
                source,
                snapshot_version=snapshot_version,
            )
            print(f"   ✓ Fork created: {fork_bucket}")
            print(f"   Forked from: {source}@{snapshot_version}")
        except Exception as e:  # noqa: BLE001
            print(f"   ✗ Error creating fork: {e}")

    print("\n" + "=" * 60)
    print("Summary (forks created):")
    for strat in strategies:
        print(f"  - {strat['fork_bucket']} ({strat['name']} chunking)")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Manage Tigris buckets, snapshots and forks for this project.",
    )
    sub = p.add_subparsers(dest="command", required=True)

    # create-bucket
    p_bucket = sub.add_parser(
        "create-bucket",
        help="Create a bucket with snapshots enabled.",
    )
    p_bucket.add_argument(
        "--bucket",
        required=True,
        help="Bucket name to create (e.g. product-dataset).",
    )
    p_bucket.set_defaults(func=cmd_create_bucket)

    # create-snapshot
    p_snap = sub.add_parser(
        "create-snapshot",
        help="Create a snapshot for a bucket.",
    )
    p_snap.add_argument(
        "--bucket",
        required=True,
        help="Source bucket name (e.g. product-dataset).",
    )
    p_snap.add_argument(
        "--name",
        required=True,
        help="Human-readable snapshot name (e.g. heading-chunking).",
    )
    p_snap.set_defaults(func=cmd_create_snapshot)

    # create-fork
    p_fork = sub.add_parser(
        "create-fork",
        help="Create a fork bucket from a snapshot version.",
    )
    p_fork.add_argument(
        "--source-bucket",
        required=True,
        help="Source bucket name (e.g. product-dataset).",
    )
    p_fork.add_argument(
        "--snapshot-version",
        required=True,
        help="Snapshot version ID (from create-snapshot).",
    )
    p_fork.add_argument(
        "--fork-bucket",
        required=True,
        help="Name of fork bucket to create (e.g. product-dataset-heading).",
    )
    p_fork.set_defaults(func=cmd_create_fork)

    # create-forks-all
    p_all = sub.add_parser(
        "create-forks-all",
        help="Create snapshots + forks for naive and semantic strategies.",
    )
    p_all.add_argument(
        "--source-bucket",
        required=True,
        help="Source bucket name (e.g. product-dataset).",
    )
    p_all.set_defaults(func=cmd_create_forks_all)

    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()



