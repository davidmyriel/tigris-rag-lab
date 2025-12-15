# Tigris RAG Lab
![lab](img/lab.jpeg)

This repo is a small lab for running RAG experiments without losing track of what changed.

The problem it’s solving is boring but real: you tweak chunking or swap an embedding model, results change, and a week later nobody can say *why*—or recreate the run that looked best.

The approach here is:

- Use **Tigris** to version your dataset like code.
- Use **Qdrant** to build a separate index for each experiment.
- Use a small eval script to compare runs.

## How it works

Start with a “base” dataset bucket in Tigris. When you want to try a new idea—different chunking, slightly different cleaning, an extra subset of docs—you don’t copy the whole dataset. You create a **fork** of the bucket. That fork is your dataset “branch.”

Then you build a Qdrant collection from that fork. The code reads a manifest file (a small JSON “recipe”) that says where the data is and how to process it: which text field to embed, what chunking strategy to use, and which embedding model to call.

At the end you run `eval.py`, which runs a few test queries and prints a simple score so you can compare approaches.

## Why there’s a manifest

The manifest exists so the experiment is reproducible. It’s not meant to be fancy—just a way to record what you did.

If you don’t write this down somewhere, you’ll eventually end up with collections named `test2-final-final` and no memory of what differs between them. The manifest keeps the “what did we index, and how?” part explicit.

## Running the lab (Wikipedia)

In this repo, the dataset is a small slice of **English Wikipedia** stored in a Tigris bucket called `wiki-dataset`:

- `wiki-dataset/p_1/review.txt` … `wiki-dataset/p_10/review.txt`

![lab](img/tigris.png)

Each file is a single article (title + body). From that, the lab builds **three** Qdrant collections:

- `wiki-naive` – one vector per article (no chunking)  
- `wiki-semantic` – paragraph-level chunks  
- `wiki-window` – sliding-window chunks

![lab](img/qdrant.png)

Create a virtualenv, install dependencies, and set environment variables for Tigris and Qdrant. Then run:

```bash
chmod +x run_all.sh
./run_all.sh
```

That script:

1. Builds `wiki-naive`, `wiki-semantic`, and `wiki-window` from `wiki-dataset` using their manifests.  
2. Runs `eval.py`, which queries all three and prints the top hits so you can compare how each chunking strategy behaves.

## Files you’ll care about

- `dataset.py` – optional helper to create buckets, snapshots, and forks in Tigris (e.g. `wiki-dataset` and its variants).  
- `fill_wiki_dataset.py` – downloads a few Wikipedia articles and uploads them into `wiki-dataset/p_*/review.txt`.  
- `ingest.py` – reads a manifest (`manifests/exp-*.json`) and builds a Qdrant collection from the corresponding bucket.  
- `eval.py` – runs a few Wikipedia-style queries against `wiki-naive`, `wiki-semantic`, and `wiki-window`.  
- `run_all.sh` – one-button script to rebuild all indexes and run eval.

## Sample eval output

When you run:

```bash
./run_all.sh
```

You’ll see output like:

```text
=== Evaluating wiki-naive (wiki) ===

Query: history of the Roman Empire
Top-5 source keys:
  score=0.1088  source_key=p_4/review.txt
  score=0.0814  source_key=p_10/review.txt
  score=0.0444  source_key=p_2/review.txt
  score=0.0221  source_key=p_7/review.txt
  score=-0.0182  source_key=p_1/review.txt

=== Evaluating wiki-semantic (wiki) ===

Query: history of the Roman Empire
Top-5 source keys:
  score=0.2068  source_key=p_7/review.txt
  score=0.1994  source_key=p_5/review.txt
  score=0.1994  source_key=p_8/review.txt
  score=0.1994  source_key=p_1/review.txt
  score=0.1994  source_key=p_3/review.txt

=== Evaluating wiki-window (wiki) ===

Query: history of the Roman Empire
Top-5 source keys:
  score=0.2317  source_key=p_5/review.txt
  score=0.1815  source_key=p_7/review.txt
  score=0.1768  source_key=p_7/review.txt
  score=0.1502  source_key=p_7/review.txt
  score=0.1134  source_key=p_3/review.txt
```

This lets you see, at a glance, how each chunking strategy (naive, paragraph, sliding window) behaves on the same query.
