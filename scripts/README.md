# Utility Scripts

This directory contains utility scripts that are not part of the main Tigris RAG Lab demo.

## wipe_all_buckets.py

⚠️ **DANGEROUS**: This script deletes ALL buckets in your Tigris account.

Use with extreme caution. This is a maintenance/cleanup script and should not be part of the demo workflow.

**Usage:**
```bash
# List all buckets (safe, dry run)
.venv/bin/python scripts/wipe_all_buckets.py

# Actually delete everything (requires confirmation)
.venv/bin/python scripts/wipe_all_buckets.py --confirm
```

