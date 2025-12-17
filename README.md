# Tigris RAG Lab for Chunking and Retrieval
![lab](img/lab.jpeg)

## Which embedding or chunking strategy is the best?

This repo is a small lab for running RAG experiments. As you tweak chunking or swap an embedding model, results change, and a week later nobody can say *why*, or recreate the run that looked best.

The approach here is:

- Use **Tigris** to version your dataset like code (bucket forking).
- Use **Qdrant** to build a separate index for each experiment.
- Use **CrewAI** agents to orchestrate experiments end-to-end.
- Use a small eval script to compare runs.

## How it works

Start with a “base” dataset bucket in Tigris. When you want to try a new idea, such as different chunking, slightly different cleaning, an extra subset of docs, then you don’t copy the whole dataset. Instead, you create a **fork** of the bucket. That fork is your dataset “branch.”

Then you build a Qdrant collection from that fork. The code reads a manifest file (a small JSON “recipe”) that says where the data is and how to process it: which text field to embed, what chunking strategy to use, and which embedding model to call.

At the end you run `eval.py`, which runs a few test queries and prints a simple score so you can compare approaches.

## Why there’s a manifest

The manifest exists so the experiment is reproducible. It's simply a way to record what you did. The manifest keeps the “what did we index, and how?” part explicit.

## Running the lab (Wikipedia)

In this repo, the dataset is a small slice of **English Wikipedia** stored in a Tigris bucket called `wiki-dataset`:

- `wiki-dataset/p_1/review.txt` … `wiki-dataset/p_10/review.txt`

![lab](img/bucket.png)

Each file is a single article (title + body). From that, the lab builds **six** Qdrant collections:

- `wiki-naive` – one vector per article (no chunking)  
- `wiki-semantic` – paragraph-level chunks  
- `wiki-window` – sliding-window chunks  
- `wiki-sentence` – sentence-based chunks  
- `wiki-markdown` – markdown header-aware chunks  
- `wiki-recursive` – hierarchical recursive splitting

![lab](img/collection.png)

## Setup

1. Create a virtualenv and install dependencies:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. Set environment variables in `.env`:
   ```bash
   S3_ENDPOINT=https://t3.storage.dev
   ACCESS_KEY=your_tigris_access_key
   SECRET_ACCESS_KEY=your_tigris_secret_key
   QDRANT_URL=https://your-qdrant-cluster.qdrant.io
   QDRANT_KEY=your_qdrant_api_key
   ```

3. Run the lab:

```bash
chmod +x run_all.sh
./run_all.sh
```

That script:

1. Builds all six collections (`wiki-naive`, `wiki-semantic`, `wiki-window`, `wiki-sentence`, `wiki-markdown`, `wiki-recursive`) from `wiki-dataset` using their manifests.  
2. Runs `eval.py`, which queries all collections and prints the top hits so you can compare how each chunking strategy behaves.

## Files you’ll care about

- `dataset.py` – optional helper to create buckets, snapshots, and forks in Tigris (e.g. `wiki-dataset` and its variants).  
- `wiki.py` – downloads a few Wikipedia articles and uploads them into `wiki-dataset/p_*/review.txt`.  
- `ingest.py` – reads a manifest (`manifests/exp-*.json`) and builds a Qdrant collection from the corresponding bucket. Supports chunking strategies: `none`, `paragraph`, `window`, `sentence`, `markdown`, `recursive`.  
- `eval.py` – runs a few Wikipedia-style queries against all wiki collections.  
- `run_all.sh` – one-button script to rebuild all indexes and run eval.  
- `create_manifests.py` – generates example manifest files for all chunking strategies.  
- `crew.py` – **CrewAI agents** that orchestrate experiments: create forks, build collections, evaluate, and report results.  
- `streamlit_app.py` – **Interactive web UI** with CrewAI agent integration for creating experiments, evaluating collections, managing buckets, and querying.

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

This lets you see, at a glance, how each chunking strategy behaves on the same query.

## Chunking Strategies

The lab supports six chunking strategies:

1. **`naive`** – One vector per document (no chunking). Best for short documents.
2. **`paragraph`** – Split on blank lines. Good for well-structured text.
3. **`window`** – Sliding window with overlap. Configurable size and stride.
4. **`sentence`** – Split by sentence boundaries. Good for focused, shorter chunks.
5. **`markdown`** – Split on markdown headers (`#`, `##`, `###`). Preserves document structure.
6. **`recursive`** – Hierarchical splitting: tries paragraphs first, then sentences, then fixed size. Most flexible.

Each strategy has its own fork bucket and manifest, allowing you to experiment and compare results.

## CrewAI Agent Orchestration

The `crew.py` file demonstrates how **CrewAI agents** can automate the entire experiment workflow:

1. **Experiment Runner Agent** – Creates Tigris forks, writes manifests, and builds Qdrant collections
2. **Evaluator Agent** – Runs test queries and analyzes retrieval quality
3. **Reporter Agent** – Summarizes results and provides recommendations

Example usage:

```bash
pip install crewai crewai-tools
# Set OPENAI_API_KEY in .env (or ANTHROPIC_API_KEY for Claude)
python crew.py
```

**Note:** CrewAI agents require an LLM API key (OpenAI, Anthropic, etc.) to run. The tools in `crew.py` can also be used directly without CrewAI if you prefer manual orchestration.

This shows how agents can orchestrate the full pipeline: **Tigris forking → Qdrant indexing → Evaluation → Reporting**.

## Streamlit Web Interface

For an interactive experience, use the Streamlit web UI:

```bash
pip install streamlit
streamlit run streamlit_app.py
```

The interface provides:

- **Create Experiments**: Manual or agent-assisted chunking strategy selection
- **Evaluate Collections**: Run evaluations with AI-powered analysis
- **Manage Buckets**: List and delete experiment buckets (with agent safety checks)
- **Query Collections**: Interactive RAG queries with agent explanations

**Two modes:**
- **Manual**: Full control - you choose everything
- **Agent-Assisted**: AI agents help with strategy selection, evaluation analysis, and safety checks
