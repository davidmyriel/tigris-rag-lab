"""
Create example manifest files for all chunking strategies.
These should be uploaded to the respective fork buckets in Tigris.

Usage:
  python create_manifests.py
"""

import json
import os

# Base manifest structure
BASE_MANIFEST = {
    "dataset_bucket": "wiki-dataset",
    "doc_layout": {
        "type": "folders_text",
        "folders": [f"p_{i}" for i in range(1, 11)],
        "file_name": "review.txt",
    },
    "embedding": {
        "model": "sentence-transformers/all-MiniLM-L6-v2",
        "dim": 384,
        "distance": "cosine",
    },
}

STRATEGIES = {
    "naive": {
        "preprocess": {
            "chunker": "none",
            "chunker_version": "v1",
        },
        "index": {
            "collection_name": "wiki-naive",
            "experiment_name": "naive",
        },
    },
    "semantic": {
        "preprocess": {
            "chunker": "paragraph",
            "chunker_version": "v1",
        },
        "index": {
            "collection_name": "wiki-semantic",
            "experiment_name": "semantic",
        },
    },
    "window": {
        "preprocess": {
            "chunker": "window",
            "chunker_version": "v1",
            "window_size": 500,
            "window_stride": 250,
        },
        "index": {
            "collection_name": "wiki-window",
            "experiment_name": "window",
        },
    },
    "sentence": {
        "preprocess": {
            "chunker": "sentence",
            "chunker_version": "v1",
        },
        "index": {
            "collection_name": "wiki-sentence",
            "experiment_name": "sentence",
        },
    },
    "markdown": {
        "preprocess": {
            "chunker": "markdown",
            "chunker_version": "v1",
        },
        "index": {
            "collection_name": "wiki-markdown",
            "experiment_name": "markdown",
        },
    },
    "recursive": {
        "preprocess": {
            "chunker": "recursive",
            "chunker_version": "v1",
            "max_chunk_size": 1000,
            "chunk_overlap": 200,
        },
        "index": {
            "collection_name": "wiki-recursive",
            "experiment_name": "recursive",
        },
    },
}


def main():
    os.makedirs("manifests", exist_ok=True)

    for strategy_name, strategy_config in STRATEGIES.items():
        manifest = BASE_MANIFEST.copy()
        manifest.update(strategy_config)

        filename = f"manifests/exp-wiki-{strategy_name}-v1.json"
        with open(filename, "w") as f:
            json.dump(manifest, f, indent=2)

        print(f"✓ Created {filename}")

    print("\nManifests created. Upload them to the respective fork buckets:")
    print("  - wiki-dataset-naive/manifests/exp-wiki-naive-v1.json")
    print("  - wiki-dataset-semantic/manifests/exp-wiki-semantic-v1.json")
    print("  - wiki-dataset-window/manifests/exp-wiki-window-v1.json")
    print("  - wiki-dataset-sentence/manifests/exp-wiki-sentence-v1.json")
    print("  - wiki-dataset-markdown/manifests/exp-wiki-markdown-v1.json")
    print("  - wiki-dataset-recursive/manifests/exp-wiki-recursive-v1.json")


if __name__ == "__main__":
    main()

