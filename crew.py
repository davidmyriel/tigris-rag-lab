"""
CrewAI agents for orchestrating Tigris RAG Lab experiments.

This demonstrates:
1. Tigris bucket forking (dataset versioning)
2. Qdrant vector indexing (per-experiment collections)
3. CrewAI agents (automated experiment orchestration)

Usage:
  python crew.py
"""

import json
import os
import subprocess
from typing import Dict, Any, List

from crewai import Agent, Task, Crew, Process
from crewai.tools import tool
from dotenv import load_dotenv
import boto3
from qdrant_client import QdrantClient, models

load_dotenv()

# ============================================================================
# Tools for CrewAI agents
# ============================================================================


@tool("Create a fork bucket from the base dataset")
def create_fork_bucket(
    source_bucket: str, fork_name: str, strategy_name: str
) -> str:
    """
    Create a Tigris fork bucket for a new experiment.

    Args:
        source_bucket: Base bucket name (e.g., 'wiki-dataset')
        fork_name: Name for the fork bucket (e.g., 'wiki-dataset-experiment-1')
        strategy_name: Human-readable strategy name (e.g., 'aggressive-chunking')

    Returns:
        Success message with fork bucket name
    """
    try:
        # Step 1: Create snapshot
        snap_result = subprocess.run(
            [
                ".venv/bin/python",
                "dataset.py",
                "create-snapshot",
                "--bucket",
                source_bucket,
                "--name",
                f"{strategy_name}-snapshot",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        
        # Parse snapshot version from output (format: "Version: <version>" or "Snapshot created: <version>")
        output_lines = snap_result.stdout.split("\n")
        snapshot_version = None
        for line in output_lines:
            if "Version:" in line:
                # Extract version after "Version:"
                parts = line.split("Version:")
                if len(parts) > 1:
                    snapshot_version = parts[1].strip()
                    break
            elif "snapshot-version" in line.lower() or "x-tigris-snapshot-version" in line.lower():
                # Try to extract from headers
                import re
                match = re.search(r'\d{19,}', line)  # Snapshot versions are long numbers
                if match:
                    snapshot_version = match.group()
                    break
        
        if not snapshot_version:
            # Try to find any long number in the output
            import re
            matches = re.findall(r'\d{15,}', snap_result.stdout)
            if matches:
                snapshot_version = matches[-1]  # Take the last one (likely the version)
        
        if not snapshot_version:
            return f"Snapshot created for {source_bucket}, but could not parse version from output. Output: {snap_result.stdout[:200]}"
        
        # Step 2: Create fork from snapshot
        fork_result = subprocess.run(
            [
                ".venv/bin/python",
                "dataset.py",
                "create-fork",
                "--source-bucket",
                source_bucket,
                "--snapshot-version",
                snapshot_version,
                "--fork-bucket",
                fork_name,
            ],
            capture_output=True,
            text=True,
            check=False,  # Don't fail if bucket already exists
        )
        
        if fork_result.returncode == 0:
            return f"Fork bucket '{fork_name}' created successfully from {source_bucket}@{snapshot_version}"
        elif "BucketAlreadyExists" in fork_result.stderr:
            return f"Fork bucket '{fork_name}' already exists (this is OK)"
        else:
            return f"Fork creation had issues: {fork_result.stderr}"
            
    except subprocess.CalledProcessError as e:
        return f"Error creating fork: {e.stderr}"


@tool("Create a manifest file for an experiment")
def create_manifest(
    strategy_name: str,
    chunker: str,
    collection_name: str,
    bucket_name: str = "wiki-dataset",
    **chunker_params: Dict[str, Any],
) -> str:
    """
    Create a manifest JSON file for an experiment.

    Args:
        strategy_name: Name of the strategy (e.g., 'experiment-1')
        chunker: Chunking strategy ('none', 'paragraph', 'window', 'sentence', 'markdown', 'recursive')
        collection_name: Qdrant collection name (e.g., 'wiki-experiment-1')
        bucket_name: Base dataset bucket name
        **chunker_params: Additional parameters for chunker (e.g., window_size, max_chunk_size)

    Returns:
        Path to the created manifest file
    """
    manifest = {
        "dataset_bucket": bucket_name,
        "doc_layout": {
            "type": "folders_text",
            "folders": [f"p_{i}" for i in range(1, 11)],
            "file_name": "review.txt",
        },
        "preprocess": {
            "chunker": chunker,
            "chunker_version": "v1",
            **chunker_params,
        },
        "embedding": {
            "model": "sentence-transformers/all-MiniLM-L6-v2",
            "dim": 384,
            "distance": "cosine",
        },
        "index": {
            "collection_name": collection_name,
            "experiment_name": strategy_name,
        },
    }

    # Upload manifest directly to Tigris fork bucket
    s3_client = boto3.client(
        "s3",
        endpoint_url=os.getenv("S3_ENDPOINT"),
        aws_access_key_id=os.getenv("ACCESS_KEY"),
        aws_secret_access_key=os.getenv("SECRET_ACCESS_KEY"),
    )

    fork_bucket = f"{bucket_name.replace('-dataset', '')}-dataset-{strategy_name}"
    manifest_key = f"manifests/exp-{strategy_name}-v1.json"

    try:
        s3_client.put_object(
            Bucket=fork_bucket,
            Key=manifest_key,
            Body=json.dumps(manifest, indent=2).encode("utf-8"),
        )
        return f"Manifest created at {manifest_path} and uploaded to {fork_bucket}/{manifest_key}"
    except Exception as e:
        return f"Manifest created at {manifest_path} but upload failed: {e}"


@tool("Build a Qdrant collection from a fork bucket and manifest")
def build_collection(fork_bucket: str, manifest_key: str) -> str:
    """
    Build a Qdrant collection from a Tigris fork bucket using a manifest.

    Args:
        fork_bucket: Fork bucket name (e.g., 'wiki-dataset-experiment-1')
        manifest_key: Path to manifest in bucket (e.g., 'manifests/exp-experiment-1-v1.json')

    Returns:
        Success message with collection name
    """
    try:
        result = subprocess.run(
            [
                ".venv/bin/python",
                "ingest.py",
                "--bucket",
                fork_bucket,
                "--manifest-key",
                manifest_key,
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        return f"Collection built successfully: {result.stdout}"
    except subprocess.CalledProcessError as e:
        return f"Error building collection: {e.stderr}"


@tool("Evaluate a Qdrant collection with test queries")
def evaluate_collection(collection_name: str, queries: List[str] = None) -> str:
    """
    Evaluate a Qdrant collection by running test queries.

    Args:
        collection_name: Name of the Qdrant collection to evaluate
        queries: Optional list of custom queries (defaults to standard wiki queries)

    Returns:
        Evaluation results as a formatted string
    """
    if queries is None:
        queries = [
            "history of the Roman Empire",
            "introduction to quantum mechanics",
            "causes of World War II",
        ]

    client = QdrantClient(
        url=os.getenv("QDRANT_URL"), api_key=os.getenv("QDRANT_KEY")
    )

    if not client.collection_exists(collection_name):
        return f"Collection '{collection_name}' does not exist"

    results = []
    for query in queries:
        res = client.query_points(
            collection_name=collection_name,
            query=models.Document(
                text=query, model="sentence-transformers/all-MiniLM-L6-v2"
            ),
            using="text_embedding",
            limit=3,
        )
        top_scores = [f"{p.score:.4f}" for p in res.points[:3]]
        results.append(f"Query: {query}\n  Top-3 scores: {', '.join(top_scores)}")

    return "\n\n".join(results)


@tool("Query Qdrant collection for RAG retrieval")
def query_collection(collection_name: str, query: str, top_k: int = 5) -> str:
    """
    Query a Qdrant collection for RAG retrieval.

    Args:
        collection_name: Name of the Qdrant collection
        query: Search query text
        top_k: Number of results to return

    Returns:
        Formatted results with scores and source keys
    """
    client = QdrantClient(
        url=os.getenv("QDRANT_URL"), api_key=os.getenv("QDRANT_KEY")
    )

    if not client.collection_exists(collection_name):
        return f"Collection '{collection_name}' does not exist"

    res = client.query_points(
        collection_name=collection_name,
        query=models.Document(
            text=query, model="sentence-transformers/all-MiniLM-L6-v2"
        ),
        using="text_embedding",
        limit=top_k,
    )

    results = [f"Query: {query}\nTop-{top_k} results:"]
    for i, point in enumerate(res.points, 1):
        source = point.payload.get("source_key", "unknown")
        score = point.score
        results.append(f"  {i}. score={score:.4f} source={source}")

    return "\n".join(results)


# ============================================================================
# CrewAI Agents (created on demand to avoid requiring API key at import time)
# ============================================================================

def create_agents():
    """Create CrewAI agents. Requires OPENAI_API_KEY or other LLM API key."""
    experiment_runner = Agent(
        role="Experiment Runner",
        goal="Create new RAG experiments by forking datasets, creating manifests, and building Qdrant collections",
        backstory="""You are an expert at setting up RAG experiments. You understand that 
        each experiment needs its own Tigris fork bucket and manifest file. You carefully 
        create forks from the base dataset, write manifest files with the correct chunking 
        strategy, and build Qdrant collections.""",
        tools=[create_fork_bucket, create_manifest, build_collection],
        verbose=True,
        allow_delegation=False,
    )

    evaluator = Agent(
        role="RAG Evaluator",
        goal="Evaluate Qdrant collections by running test queries and analyzing retrieval quality",
        backstory="""You are a quality assurance specialist for RAG systems. You run 
        standardized test queries against Qdrant collections and analyze the results. 
        You compare different chunking strategies and provide insights on which works best.""",
        tools=[evaluate_collection, query_collection],
        verbose=True,
        allow_delegation=False,
    )

    reporter = Agent(
        role="Experiment Reporter",
        goal="Summarize experiment results and provide recommendations",
        backstory="""You are a technical writer who specializes in summarizing RAG 
        experiments. You take evaluation results and create clear, actionable summaries 
        that help teams understand which chunking strategies work best for their use case.""",
        verbose=True,
        allow_delegation=True,
    )
    
    return experiment_runner, evaluator, reporter


# ============================================================================
# Example Tasks and Crew
# ============================================================================

def run_experiment(
    strategy_name: str,
    chunker: str,
    description: str,
    **chunker_params: Dict[str, Any],
) -> str:
    """
    Run a complete RAG experiment using CrewAI agents.

    Args:
        strategy_name: Unique name for this experiment
        chunker: Chunking strategy to use
        description: Human-readable description of the experiment
        **chunker_params: Additional parameters for the chunker

    Returns:
        Summary of the experiment results
    """
    base_bucket = "wiki-dataset"
    fork_bucket = f"wiki-dataset-{strategy_name}"
    collection_name = f"wiki-{strategy_name}"
    manifest_key = f"manifests/exp-{strategy_name}-v1.json"

    # Create agents
    experiment_runner, evaluator, reporter = create_agents()
    
    # Task 1: Set up the experiment
    setup_task = Task(
        description=f"""
        Set up a new RAG experiment called '{strategy_name}' with the following:
        - Strategy description: {description}
        - Chunking method: {chunker}
        - Chunker parameters: {chunker_params if chunker_params else 'default'}
        
        Steps:
        1. Create a fork bucket '{fork_bucket}' from '{base_bucket}'
        2. Create a manifest file for this experiment with chunker '{chunker}'
        3. Build a Qdrant collection '{collection_name}' from the fork
        
        Make sure the manifest is uploaded to the fork bucket at '{manifest_key}'.
        """,
        agent=experiment_runner,
        expected_output="Confirmation that the fork bucket, manifest, and Qdrant collection were created successfully",
    )

    # Task 2: Evaluate the collection
    eval_task = Task(
        description=f"""
        Evaluate the Qdrant collection '{collection_name}' by running test queries.
        Run the standard evaluation queries and report the top-3 scores for each query.
        """,
        agent=evaluator,
        expected_output="Evaluation results showing query scores and top results",
        context=[setup_task],
    )

    # Task 3: Generate report
    report_task = Task(
        description=f"""
        Based on the experiment setup and evaluation results, create a summary report:
        - What chunking strategy was used
        - How the collection performed on test queries
        - Any insights or recommendations
        
        Keep it concise and actionable.
        """,
        agent=reporter,
        expected_output="A clear summary report of the experiment results",
        context=[setup_task, eval_task],
    )

    # Create and run the crew
    crew = Crew(
        agents=[experiment_runner, evaluator, reporter],
        tasks=[setup_task, eval_task, report_task],
        process=Process.sequential,
        verbose=True,
    )

    result = crew.kickoff()
    return result


def main():
    """Example: Run a few experiments with different chunking strategies."""
    
    # Check for required API key
    if not os.getenv("OPENAI_API_KEY"):
        print("=" * 70)
        print("⚠️  CrewAI requires an LLM API key")
        print("=" * 70)
        print("\nTo run CrewAI agents, set OPENAI_API_KEY in your .env file:")
        print("  OPENAI_API_KEY=your_openai_api_key")
        print("\nAlternatively, you can use other providers:")
        print("  - ANTHROPIC_API_KEY (for Claude)")
        print("  - Or configure via CrewAI's LLM settings")
        print("\n" + "=" * 70)
        print("\nThe tools can still be used directly without CrewAI:")
        print("  - create_fork_bucket()")
        print("  - create_manifest()")
        print("  - build_collection()")
        print("  - evaluate_collection()")
        print("  - query_collection()")
        return

    print("=" * 70)
    print("Tigris RAG Lab - CrewAI Orchestration Demo")
    print("=" * 70)
    print("\nThis demo shows:")
    print("1. Tigris bucket forking (dataset versioning)")
    print("2. Qdrant vector indexing (per-experiment collections)")
    print("3. CrewAI agents (automated experiment orchestration)")
    print("\n" + "=" * 70 + "\n")

    # Example: Run a sentence-based chunking experiment
    print("\n[Experiment 1: Sentence-based chunking]")
    print("-" * 70)
    result1 = run_experiment(
        strategy_name="sentence-test",
        chunker="sentence",
        description="Test sentence-based chunking for focused retrieval",
    )
    print("\nResult:", result1)

    # Uncomment to run more experiments:
    # print("\n[Experiment 2: Recursive chunking]")
    # print("-" * 70)
    # result2 = run_experiment(
    #     strategy_name="recursive-test",
    #     chunker="recursive",
    #     description="Test recursive hierarchical chunking",
    #     max_chunk_size=800,
    #     chunk_overlap=150,
    # )
    # print("\nResult:", result2)


if __name__ == "__main__":
    main()

