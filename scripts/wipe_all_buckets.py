"""
Wipe all buckets in your Tigris account.

⚠️  WARNING: This will delete ALL buckets and ALL data in your account!
This action cannot be undone.

Usage:
    .venv/bin/python wipe_all_buckets.py [--confirm]
"""

import argparse
import os
import sys
from typing import List

import boto3
from dotenv import load_dotenv

load_dotenv()


def make_s3_client():
    """Create S3 client for Tigris."""
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


def list_all_buckets(s3_client) -> List[str]:
    """List all bucket names."""
    try:
        response = s3_client.list_buckets()
        bucket_names = [bucket["Name"] for bucket in response.get("Buckets", [])]
        return bucket_names
    except Exception as e:
        print(f"✗ Error listing buckets: {e}")
        return []


def delete_bucket_contents(s3_client, bucket_name: str) -> bool:
    """Delete all objects in a bucket."""
    try:
        # List all objects
        paginator = s3_client.get_paginator("list_objects_v2")
        pages = paginator.paginate(Bucket=bucket_name)
        
        object_count = 0
        for page in pages:
            if "Contents" in page:
                objects = [{"Key": obj["Key"]} for obj in page["Contents"]]
                if objects:
                    s3_client.delete_objects(
                        Bucket=bucket_name,
                        Delete={"Objects": objects}
                    )
                    object_count += len(objects)
        
        if object_count > 0:
            print(f"   Deleted {object_count} object(s)")
        return True
    except Exception as e:
        print(f"   ✗ Error deleting contents: {e}")
        return False


def delete_bucket(s3_client, bucket_name: str) -> bool:
    """Delete a bucket (after emptying it)."""
    try:
        # First, delete all objects
        if not delete_bucket_contents(s3_client, bucket_name):
            return False
        
        # Then delete the bucket
        s3_client.delete_bucket(Bucket=bucket_name)
        return True
    except s3_client.exceptions.NoSuchBucket:
        print(f"   ⚠ Bucket '{bucket_name}' does not exist (already deleted?)")
        return True
    except Exception as e:
        print(f"   ✗ Error deleting bucket: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Wipe all buckets in your Tigris account",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
⚠️  WARNING: This will delete ALL buckets and ALL data!
This action cannot be undone.

Examples:
  # List buckets first (dry run)
  .venv/bin/python wipe_all_buckets.py

  # Actually delete everything (requires confirmation)
  .venv/bin/python wipe_all_buckets.py --confirm
        """
    )
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="Actually delete buckets (without this flag, only lists buckets)",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("TIGRIS BUCKET WIPER")
    print("=" * 70)
    print()

    s3_client = make_s3_client()

    # List all buckets
    print("📋 Listing all buckets...")
    bucket_names = list_all_buckets(s3_client)

    if not bucket_names:
        print("✓ No buckets found. Nothing to delete.")
        return

    print(f"Found {len(bucket_names)} bucket(s):")
    for i, name in enumerate(bucket_names, 1):
        print(f"  {i}. {name}")
    print()

    if not args.confirm:
        print("⚠️  DRY RUN MODE - No buckets will be deleted")
        print("   Run with --confirm to actually delete buckets")
        print()
        print("   Example: .venv/bin/python wipe_all_buckets.py --confirm")
        return

    # Confirmation prompt
    print("⚠️  WARNING: You are about to delete ALL buckets!")
    print("   This will permanently delete:")
    print(f"   - {len(bucket_names)} bucket(s)")
    print("   - ALL data in those buckets")
    print("   - This action CANNOT be undone!")
    print()
    
    response = input("Type 'DELETE ALL' to confirm: ")
    if response != "DELETE ALL":
        print("❌ Confirmation failed. Aborting.")
        sys.exit(1)

    print()
    print("🗑️  Deleting buckets...")
    print("-" * 70)

    deleted_count = 0
    failed_count = 0

    for bucket_name in bucket_names:
        print(f"\n[{bucket_name}]")
        if delete_bucket(s3_client, bucket_name):
            print(f"   ✓ Deleted bucket '{bucket_name}'")
            deleted_count += 1
        else:
            print(f"   ✗ Failed to delete bucket '{bucket_name}'")
            failed_count += 1

    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total buckets: {len(bucket_names)}")
    print(f"Deleted: {deleted_count}")
    if failed_count > 0:
        print(f"Failed: {failed_count}")
    print("=" * 70)


if __name__ == "__main__":
    main()

