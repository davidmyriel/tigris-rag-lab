"""
Streamlit interface for Tigris RAG Lab with CrewAI agent integration.

Simplified two-column interface:
- Left: Upload dataset, select optimization goal, start
- Right: Real-time progress and results
"""

import os
import json
import streamlit as st
from dotenv import load_dotenv
import subprocess
from typing import Dict, Any, List
import uuid
from datetime import datetime

# Load environment
load_dotenv()

# Import CrewAI components
try:
    from crewai import Agent, Task, Crew, Process
    from crewai.tools import tool
    from qdrant_client import QdrantClient, models
    import boto3
    CREWAI_AVAILABLE = True
except ImportError:
    CREWAI_AVAILABLE = False
    st.error("CrewAI not installed. Run: pip install crewai crewai-tools")

# Page config
st.set_page_config(
    page_title="Tigris RAG Lab",
    page_icon="🔬",
    layout="wide",
)

# Available chunking strategies
CHUNKING_STRATEGIES = {
    "none": {
        "name": "None (Full Document)",
        "description": "One vector per document - no chunking",
        "params": {},
    },
    "paragraph": {
        "name": "Paragraph",
        "description": "Split on blank lines (paragraph boundaries)",
        "params": {},
    },
    "window": {
        "name": "Sliding Window",
        "description": "Overlapping character-based chunks",
        "params": {
            "window_size": 500,
            "window_stride": 250,
        },
    },
    "sentence": {
        "name": "Sentence",
        "description": "Split by sentence boundaries",
        "params": {},
    },
    "markdown": {
        "name": "Markdown",
        "description": "Split on markdown headers (# ## ###)",
        "params": {},
    },
    "recursive": {
        "name": "Recursive",
        "description": "Hierarchical: paragraphs → sentences → fixed size",
        "params": {
            "max_chunk_size": 1000,
            "chunk_overlap": 200,
        },
    },
}

BASE_BUCKET = "wiki-dataset"

# ============================================================================
# Tool Functions (decorated for CrewAI)
# ============================================================================

@tool("Create a fork bucket from the base dataset")
def create_fork_bucket_tool(source_bucket: str, fork_name: str, strategy_name: str) -> str:
    """Create a Tigris fork bucket."""
    import re
    try:
        snap_result = subprocess.run(
            [".venv/bin/python", "dataset.py", "create-snapshot",
             "--bucket", source_bucket, "--name", f"{strategy_name}-snapshot"],
            capture_output=True, text=True, check=True,
        )
        matches = re.findall(r'\d{15,}', snap_result.stdout)
        snapshot_version = matches[-1] if matches else None
        
        if not snapshot_version:
            return f"Snapshot created but version not found. Output: {snap_result.stdout[:200]}"
        
        fork_result = subprocess.run(
            [".venv/bin/python", "dataset.py", "create-fork",
             "--source-bucket", source_bucket,
             "--snapshot-version", snapshot_version,
             "--fork-bucket", fork_name],
            capture_output=True, text=True, check=False,
        )
        
        if fork_result.returncode == 0:
            return f"✓ Fork bucket '{fork_name}' created successfully"
        elif "BucketAlreadyExists" in fork_result.stderr:
            return f"✓ Fork bucket '{fork_name}' already exists"
        else:
            return f"Fork creation had issues: {fork_result.stderr}"
    except Exception as e:
        return f"Error: {e}"


@tool("Create a manifest file for an experiment")
def create_manifest_tool(strategy_name: str, chunker: str, collection_name: str,
                         bucket_name: str = "wiki-dataset", file_name: str = "review.txt", **chunker_params) -> str:
    """Create and upload manifest to the fork bucket."""
    manifest = {
        "dataset_bucket": bucket_name,
        "doc_layout": {
            "type": "folders_text",
            "folders": ["source"],
            "file_name": file_name,
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
    
    s3_client = boto3.client(
        "s3",
        endpoint_url=os.getenv("S3_ENDPOINT"),
        aws_access_key_id=os.getenv("ACCESS_KEY"),
        aws_secret_access_key=os.getenv("SECRET_ACCESS_KEY"),
    )
    
    # Fork bucket name: source-bucket-experiment-strategy
    fork_bucket = f"{bucket_name}-{strategy_name}"
    manifest_key = f"manifests/exp-{strategy_name}-v1.json"
    
    try:
        s3_client.put_object(
            Bucket=fork_bucket,
            Key=manifest_key,
            Body=json.dumps(manifest, indent=2).encode("utf-8"),
        )
        return f"✓ Manifest created and uploaded successfully.\nFORK_BUCKET={fork_bucket}\nMANIFEST_KEY={manifest_key}\nUse these exact values for build_collection_tool."
    except Exception as e:
        return f"✗ Manifest upload failed: {e}\nFORK_BUCKET={fork_bucket}\nMANIFEST_KEY={manifest_key}"


@tool("Pre-chunk documents and store chunks in fork bucket")
def prechunk_tool(fork_bucket: str, manifest_key: str) -> str:
    """Pre-chunk documents according to manifest and store in fork bucket."""
    try:
        if not manifest_key.startswith("manifests/"):
            manifest_key = f"manifests/{manifest_key}" if not manifest_key.startswith("/") else manifest_key[1:]
        
        result = subprocess.run(
            [".venv/bin/python", "chunk.py", "--bucket", fork_bucket, "--manifest-key", manifest_key],
            capture_output=True, text=True, check=True,
        )
        
        output_lines = result.stdout.split("\n")
        success_lines = [line for line in output_lines if "✓" in line]
        output_summary = "\n".join(success_lines[-3:]) if success_lines else result.stdout[-200:]
        
        return f"✓ Pre-chunking complete\n{output_summary}"
    except subprocess.CalledProcessError as e:
        error_msg = e.stderr if e.stderr else e.stdout
        return f"✗ Error pre-chunking: {error_msg}"
    except Exception as e:
        return f"✗ Unexpected error: {e}"


@tool("Build a Qdrant collection from a fork bucket and manifest")
def build_collection_tool(fork_bucket: str, manifest_key: str) -> str:
    """Build Qdrant collection."""
    try:
        if not manifest_key.startswith("manifests/"):
            manifest_key = f"manifests/{manifest_key}" if not manifest_key.startswith("/") else manifest_key[1:]
        
        s3_client = boto3.client(
            "s3",
            endpoint_url=os.getenv("S3_ENDPOINT"),
            aws_access_key_id=os.getenv("ACCESS_KEY"),
            aws_secret_access_key=os.getenv("SECRET_ACCESS_KEY"),
        )
        
        try:
            s3_client.head_object(Bucket=fork_bucket, Key=manifest_key)
        except Exception as e:
            return f"✗ Manifest not found: {fork_bucket}/{manifest_key}. Error: {e}"
        
        result = subprocess.run(
            [".venv/bin/python", "ingest.py", "--bucket", fork_bucket, "--manifest-key", manifest_key],
            capture_output=True, text=True, check=True,
        )
        
        output_lines = result.stdout.split("\n")
        success_lines = [line for line in output_lines if "✓" in line or "Ingested" in line]
        output_summary = "\n".join(success_lines[-3:]) if success_lines else result.stdout[-200:]
        
        return f"✓ Collection built successfully\n{output_summary}"
    except subprocess.CalledProcessError as e:
        error_msg = e.stderr if e.stderr else e.stdout
        return f"✗ Error building collection: {error_msg}"
    except Exception as e:
        return f"✗ Unexpected error: {e}"


@tool("Evaluate a Qdrant collection and return performance scores")
def evaluate_collection_tool(collection_name: str, queries: List[str] = None) -> str:
    """Evaluate collection and return scores as JSON string."""
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
        return json.dumps({"error": f"Collection '{collection_name}' does not exist", "avg_score": 0.0})
    
    scores = []
    for query in queries:
        res = client.query_points(
            collection_name=collection_name,
            query=models.Document(text=query, model="sentence-transformers/all-MiniLM-L6-v2"),
            using="text_embedding",
            limit=3,
        )
        if res.points:
            avg_score = sum(p.score for p in res.points[:3]) / len(res.points[:3])
            scores.append(avg_score)
    
    if scores:
        avg_overall = sum(scores) / len(scores)
        result = {
            "avg_score": avg_overall,
            "query_scores": scores,
            "collection": collection_name,
        }
        return json.dumps(result)
    return json.dumps({"error": "No scores calculated", "avg_score": 0.0})


@tool("Get a sample chunk from a Qdrant collection")
def get_sample_chunk_tool(collection_name: str) -> str:
    """Get a sample chunk from the collection to show as example."""
    try:
        client = QdrantClient(
            url=os.getenv("QDRANT_URL"), api_key=os.getenv("QDRANT_KEY")
        )
        
        if not client.collection_exists(collection_name):
            return f"Collection '{collection_name}' does not exist"
        
        # Get a random point
        res = client.scroll(
            collection_name=collection_name,
            limit=1,
        )
        
        if res[0]:
            point = res[0][0]
            payload = point.payload or {}
            chunk_text = payload.get("text", "No text found in payload")
            return f"Sample chunk from '{collection_name}':\n\n{chunk_text[:500]}..."
        return "No chunks found in collection"
    except Exception as e:
        return f"Error retrieving sample: {e}"


@tool("Delete a Tigris bucket")
def delete_bucket_tool(bucket_name: str) -> str:
    """Delete a Tigris bucket."""
    if bucket_name == BASE_BUCKET:
        return "Cannot delete base bucket"
    
    try:
        s3_client = boto3.client(
            "s3",
            endpoint_url=os.getenv("S3_ENDPOINT"),
            aws_access_key_id=os.getenv("ACCESS_KEY"),
            aws_secret_access_key=os.getenv("SECRET_ACCESS_KEY"),
        )
        
        objects = s3_client.list_objects_v2(Bucket=bucket_name)
        if "Contents" in objects:
            for obj in objects["Contents"]:
                s3_client.delete_object(Bucket=bucket_name, Key=obj["Key"])
        
        s3_client.delete_bucket(Bucket=bucket_name)
        return f"✓ Successfully deleted bucket '{bucket_name}'"
    except Exception as e:
        return f"✗ Error deleting bucket: {e}"


@tool("Delete a Qdrant collection")
def delete_collection_tool(collection_name: str) -> str:
    """Delete a Qdrant collection."""
    try:
        client = QdrantClient(
            url=os.getenv("QDRANT_URL"), api_key=os.getenv("QDRANT_KEY")
        )
        if client.collection_exists(collection_name):
            client.delete_collection(collection_name)
            return f"✓ Successfully deleted collection '{collection_name}'"
        return f"Collection '{collection_name}' does not exist"
    except Exception as e:
        return f"✗ Error deleting collection: {e}"


# ============================================================================
# Helper Functions
# ============================================================================

def upload_file_to_tigris(file_content: bytes, file_name: str) -> tuple[str, str] | tuple[None, None]:
    """Upload a file to a Tigris snapshot-enabled bucket named after the file.

    Returns:
        Tuple of (bucket_name, uploaded_file_name) or (None, None) on error
    """
    try:
        # Derive bucket name from file name (remove extension, sanitize)
        # Tigris bucket names: lowercase letters, numbers, hyphens, dots only
        # No underscores allowed - convert them to hyphens
        bucket_name = os.path.splitext(file_name)[0].lower()
        bucket_name = "".join(
            c if c.isalnum() or c in ["-", "."] else "-" for c in bucket_name.replace("_", "-")
        ).strip("-")

        if not bucket_name:
            raise ValueError("Could not derive a valid bucket name from file name")

        # Create a proper Tigris snapshot bucket via dataset.py
        # This ensures snapshots/forks work for this uploaded dataset.
        create_cmd = [
            ".venv/bin/python",
            "dataset.py",
            "create-bucket",
            "--bucket",
            bucket_name,
        ]
        result = subprocess.run(
            create_cmd,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0 and "already exists" not in result.stdout:
            st.error(f"Bucket creation failed: {result.stdout or result.stderr}")
            return None

        # Now use boto3 to upload the file into that bucket
        s3_client = boto3.client(
            "s3",
            endpoint_url=os.getenv("S3_ENDPOINT"),
            aws_access_key_id=os.getenv("ACCESS_KEY"),
            aws_secret_access_key=os.getenv("SECRET_ACCESS_KEY"),
        )

        # Store file with its original name in source/ folder
        key = f"source/{file_name}"
        s3_client.put_object(Bucket=bucket_name, Key=key, Body=file_content)
        return bucket_name, file_name
    except Exception as e:
        st.error(f"Upload error: {e}")
        return None, None


# ============================================================================
# Main UI
# ============================================================================

def main():
    st.title("🔬 Tigris RAG Lab")
    st.markdown("**Autonomous chunking strategy optimization**")
    
    if not CREWAI_AVAILABLE:
        st.error("⚠️ CrewAI not available. Run: pip install crewai crewai-tools")
        st.stop()
    
    if not os.getenv("OPENAI_API_KEY"):
        st.error("⚠️ OPENAI_API_KEY not set in .env file")
        st.stop()
    
    # Two-column layout
    col_left, col_right = st.columns([1, 2])
    
    with col_left:
        st.header("📤 Setup")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Upload your dataset (text file)",
            type=["txt", "md"],
            help="Upload a text file to optimize chunking for"
        )
        
        # Optimization goal
        optimization_goal = st.radio(
            "Optimize for:",
            ["Recall (retrieval quality)", "Token Cost (efficiency)"],
            help="Select your primary optimization goal"
        )
        
        # Start button
        start_experiment = st.button(
            "🚀 Start Experiment",
            type="primary",
            use_container_width=True,
            disabled=uploaded_file is None
        )
    
    with col_right:
        st.header("📊 Progress")
        
        if not start_experiment:
            st.info("👈 Upload a file and select optimization goal, then click Start")
        else:
            # Initialize session state for progress tracking
            if "experiment_status" not in st.session_state:
                st.session_state.experiment_status = {
                    "step": "initializing",
                    "strategies": [],
                    "results": {},
                    "winner": None,
                    "sample_chunk": None,
                }
            
            status = st.session_state.experiment_status
            
            # Upload file if provided and get bucket name
            source_bucket = BASE_BUCKET  # Default to wiki-dataset if no file uploaded
            uploaded_file_name = "review.txt"  # Default filename
            if uploaded_file:
                with st.spinner("📤 Uploading file to Tigris..."):
                    file_content = uploaded_file.read()
                    result = upload_file_to_tigris(file_content, uploaded_file.name)
                    if result and result[0]:
                        bucket_name, uploaded_file_name = result
                        source_bucket = bucket_name
                        st.success(f"✓ File uploaded to bucket: {bucket_name}")
                    else:
                        st.error("✗ Upload failed")
                        st.stop()
            
            # Store bucket name and filename in session state for later use
            st.session_state.source_bucket = source_bucket
            st.session_state.uploaded_file_name = uploaded_file_name
            
            # Create progress container
            progress_container = st.container()
            
            with progress_container:
                # Progress bar
                progress_bar = st.progress(0, text="Initializing...")
                
                # Step display
                steps_display = st.empty()
                
                # Results display
                results_display = st.empty()
            
            # Determine optimization focus
            if "recall" in optimization_goal.lower():
                use_case = "Optimize for maximum retrieval quality and recall. Prioritize strategies that preserve semantic context and document structure. Quality is more important than token efficiency."
            else:
                use_case = "Optimize for token cost efficiency. Prioritize strategies that minimize the number of chunks while maintaining reasonable retrieval quality. Efficiency is more important than perfect recall."
            
            # Generate unique experiment name
            experiment_base_name = f"exp-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            
            # Create the autonomous agent
            executor = Agent(
                role="Autonomous RAG Experiment Orchestrator",
                goal="Select top 3 chunking strategies, create experiments, evaluate them, and keep only the best",
                backstory=f"""You are an expert RAG systems engineer optimizing for: {optimization_goal}.
                
                You understand text breakdown priorities:
                - For structured documents with headers: prefer markdown or recursive
                - For long-form content: prefer paragraph or recursive
                - For short documents: prefer sentence or none
                - For overlapping context needs: prefer window or recursive
                - For speed/cost: prefer none or paragraph
                - For precision/recall: prefer sentence or markdown
                
                You autonomously execute complete experiments: create forks, manifests, collections, evaluate, and clean up losers.""",
                tools=[
                    create_fork_bucket_tool,
                    create_manifest_tool,
                    prechunk_tool,
                    build_collection_tool,
                    evaluate_collection_tool,
                    get_sample_chunk_tool,
                    delete_bucket_tool,
                    delete_collection_tool,
                ],
                verbose=True,
                allow_delegation=False,
            )
            
            # Build task description
            task_description = f"""
                The user wants to optimize chunking for: {optimization_goal}
                
                Use case: {use_case}
                
                Available chunking strategies: {list(CHUNKING_STRATEGIES.keys())}
                
                IMPORTANT: The source dataset bucket is '{source_bucket}'. Use this bucket name for all operations.
                
                Your complete workflow:
                
                1. ANALYZE the optimization goal and select the top 3 chunking strategies.
                
                2. CREATE 3 experiments (for each selected strategy):
                   For each strategy (e.g., 'markdown', 'recursive', 'sentence'):
                   
                   a) Create fork bucket:
                      - Call: create_fork_bucket_tool(
                        source_bucket='{source_bucket}',
                        fork_name='{source_bucket}-{experiment_base_name}-<strategy_name>',
                        strategy_name='<strategy_name>'
                      )
                   
                   b) Create manifest:
                      - Call: create_manifest_tool(
                        strategy_name='{experiment_base_name}-<strategy_name>',
                        chunker='<strategy_name>',
                        collection_name='{source_bucket}-{experiment_base_name}-<strategy_name>',
                        bucket_name='{source_bucket}',
                        file_name='{uploaded_file_name}',
                        **optimal_params_for_strategy
                      )
                      Use the EXACT fork bucket name and manifest key path returned!
                   
                   c) Pre-chunk documents:
                      - Call: prechunk_tool(
                        fork_bucket='{source_bucket}-{experiment_base_name}-<strategy_name>',
                        manifest_key='manifests/exp-{experiment_base_name}-<strategy_name>-v1.json'
                      )
                      This stores chunks in the fork bucket as JSONL files.
                   
                   d) Build collection:
                      - Call: build_collection_tool(
                        fork_bucket='{source_bucket}-{experiment_base_name}-<strategy_name>',
                        manifest_key='manifests/exp-{experiment_base_name}-<strategy_name>-v1.json'
                      )
                      This reads the pre-chunked files and builds the Qdrant collection.
                
                3. EVALUATE all 3 collections:
                   - Run evaluation queries using evaluate_collection_tool
                   - Compare average scores
                   - Identify the best performer
                
                4. GET SAMPLE CHUNK from the winning collection:
                   - Call: get_sample_chunk_tool(collection_name='{source_bucket}-{experiment_base_name}-<best_strategy>')
                
                5. CLEANUP:
                   - Delete the 2 worse-performing collections using delete_collection_tool
                   - Delete the 2 worse-performing fork buckets using delete_bucket_tool
                   - Keep only the best performing experiment
                
                IMPORTANT: You must actually execute all steps using the tools. Don't just plan - execute!
                
                After completion, provide a summary in this EXACT format:
                
                "Selected chunking strategies were [strategy1], [strategy2], and [strategy3] because [reasoning about why these strategies fit the optimization goal]. After creating fork buckets, manifests, pre-chunking, and building collections for each, I evaluated all with relevant queries. The [winning strategy] strategy collection scored highest (avg_score [score]), followed by [second strategy] ([score]), and [third strategy] last ([score]). I retrieved a sample chunk from the [winning strategy] collection to verify chunking quality. Finally, I deleted the two worse performing collections and associated fork buckets ([deleted strategies]), preserving only the best performing [winning strategy] experiment."
                
                The summary must:
                - List all 3 selected strategies with reasoning
                - Mention all workflow steps (fork buckets, manifests, pre-chunking, building collections)
                - Include evaluation scores for all 3 strategies
                - Identify the winner with its score
                - Mention sample chunk retrieval
                - Specify which collections and buckets were deleted
                - Be written as a single coherent paragraph
                """
            
            task = Task(
                description=task_description,
                agent=executor,
                expected_output="A single paragraph summary in the specified format: strategies selected with reasoning, workflow steps completed, evaluation scores for all 3 strategies, winner identified, sample chunk retrieved, and cleanup actions (which collections/buckets were deleted).",
            )
            
            crew = Crew(
                agents=[executor],
                tasks=[task],
                process=Process.sequential,
                verbose=True,
            )
            
            # Run the experiment
            try:
                with st.spinner("🤖 Agent is working..."):
                    progress_bar.progress(10, text="Agent analyzing optimization goal...")
                    
                    result = crew.kickoff()
                    
                    progress_bar.progress(100, text="✅ Experiment complete!")
                    
                    # Parse result - CrewAI returns different object types
                    # The result is typically a CrewOutput with tasks_output
                    result_text = None
                    
                    # Debug: Print result structure
                    print(f"DEBUG: Result type: {type(result)}")
                    print(f"DEBUG: Result dir: {[x for x in dir(result) if not x.startswith('_')]}")
                    
                    # Method 1: Check for tasks_output (most common)
                    if hasattr(result, 'tasks_output') and result.tasks_output:
                        task_result = result.tasks_output[0]
                        print(f"DEBUG: Task result type: {type(task_result)}")
                        print(f"DEBUG: Task result dir: {[x for x in dir(task_result) if not x.startswith('_')]}")
                        
                        if hasattr(task_result, 'raw') and task_result.raw:
                            result_text = str(task_result.raw)
                            print(f"DEBUG: Got text from task_result.raw (length: {len(result_text)})")
                        elif hasattr(task_result, 'output') and task_result.output:
                            result_text = str(task_result.output)
                            print(f"DEBUG: Got text from task_result.output (length: {len(result_text)})")
                        else:
                            result_text = str(task_result)
                            print(f"DEBUG: Got text from str(task_result) (length: {len(result_text)})")
                    
                    # Method 2: Check for raw attribute directly
                    if not result_text and hasattr(result, 'raw') and result.raw:
                        result_text = str(result.raw)
                        print(f"DEBUG: Got text from result.raw (length: {len(result_text)})")
                    
                    # Method 3: Check for output attribute directly
                    if not result_text and hasattr(result, 'output') and result.output:
                        result_text = str(result.output)
                        print(f"DEBUG: Got text from result.output (length: {len(result_text)})")
                    
                    # Method 4: Just convert to string
                    if not result_text:
                        result_text = str(result)
                        print(f"DEBUG: Got text from str(result) (length: {len(result_text)})")
                    
                    # Clean up the text - remove any extra whitespace/newlines
                    if result_text:
                        result_text = result_text.strip()
                    
                    # Store in session state for debugging
                    st.session_state.last_result = result_text
                    st.session_state.raw_result = result
                    result_lower = result_text.lower() if result_text else ""
                    
                    print(f"DEBUG: Final result_text length: {len(result_text) if result_text else 0}")
                    print(f"DEBUG: First 200 chars: {result_text[:200] if result_text else 'None'}")
                    
                    # Extract strategies
                    strategies_found = []
                    for strategy in CHUNKING_STRATEGIES.keys():
                        if strategy in result_lower:
                            strategies_found.append(strategy)
                    
                    # Extract scores from result text - look for patterns like "scored highest (avg_score 0.2759)"
                    import re
                    
                    # Pattern 1: Look for "strategy scored highest (avg_score X), followed by Y (Z), and Z last (W)"
                    # This matches the exact format: "recursive strategy collection scored highest (avg_score 0.2759), followed by sentence (0.2670), and markdown last (0.1993)"
                    pattern1 = r'\(avg_score\s+([\d.]+)\)'
                    scores_from_parens = re.findall(pattern1, result_text, re.IGNORECASE)
                    
                    # Pattern 2: Look for "strategy (score)" format
                    pattern2 = r'(\w+)\s+\(([\d.]+)\)'
                    strategy_score_pairs = re.findall(pattern2, result_text, re.IGNORECASE)
                    
                    # Pattern 3: Look for any 4-decimal numbers (0.XXXX format)
                    pattern3 = r'\b0\.\d{4}\b'
                    all_decimals = re.findall(pattern3, result_text)
                    
                    scores = []
                    strategies_with_scores = []
                    
                    # Try pattern 2 first (strategy-score pairs)
                    if strategy_score_pairs:
                        for strategy, score in strategy_score_pairs:
                            strategy_lower = strategy.lower()
                            if strategy_lower in CHUNKING_STRATEGIES:
                                strategies_with_scores.append(strategy_lower)
                                scores.append(float(score))
                    
                    # Fallback to pattern 1 (scores in parentheses)
                    if not scores and scores_from_parens:
                        scores = [float(s) for s in scores_from_parens[:3]]
                        strategies_with_scores = strategies_found[:3] if strategies_found else []
                    
                    # Fallback to pattern 3 (any decimals)
                    if not scores and all_decimals:
                        scores = [float(s) for s in all_decimals[:3] if 0 <= float(s) <= 1]
                        strategies_with_scores = strategies_found[:3] if strategies_found else []
                    
                    # Store in session state
                    if scores and strategies_with_scores:
                        st.session_state.eval_scores = scores
                        st.session_state.strategies_tested = strategies_with_scores
                    elif scores:
                        # We have scores but no strategies matched - use found strategies
                        st.session_state.eval_scores = scores[:3]
                        st.session_state.strategies_tested = strategies_found[:3] if strategies_found else []
                    
                    # Display results
                    with steps_display:
                        st.markdown("### 📋 Steps Completed")
                        st.success("✅ 1. Analysis complete - Top 3 strategies selected")
                        st.success("✅ 2. Fork buckets created")
                        st.success("✅ 3. Manifests created")
                        st.success("✅ 4. Qdrant collections built")
                        st.success("✅ 5. Evaluation complete")
                        st.success("✅ 6. Cleanup complete")
                    
                    with results_display:
                        st.markdown("---")
                        st.markdown("### 🏆 Results")
                        
                        # Box 1: Agent Summary
                        with st.container():
                            st.markdown("#### 📝 Agent Summary")
                            
                            # Use result_text directly - it should be extracted correctly now
                            display_text = result_text
                            
                            # Fallback: try to get from session state or raw result
                            if not display_text or len(display_text.strip()) <= 10:
                                display_text = st.session_state.get("last_result", None)
                            
                            if not display_text or len(display_text.strip()) <= 10:
                                display_text = str(result) if result else "No result available"
                            
                            # Always display the result text prominently
                            if display_text and len(display_text.strip()) > 10:
                                st.markdown(display_text)
                            else:
                                st.warning("⚠️ Agent summary is empty or too short.")
                                # Show raw result text for debugging
                                st.text("Raw result text:")
                                st.code(result_text[:1000] if result_text else 'None', language="text")
                        
                        # Box 2: Sample Chunk
                        if strategies_found:
                            winner = strategies_found[0]  # Assume first mentioned is winner
                            winner_collection = f"{source_bucket}-{experiment_base_name}-{winner}"
                            
                            try:
                                client = QdrantClient(
                                    url=os.getenv("QDRANT_URL"),
                                    api_key=os.getenv("QDRANT_KEY")
                                )
                                if client.collection_exists(winner_collection):
                                    res = client.scroll(collection_name=winner_collection, limit=1)
                                    if res[0]:
                                        point = res[0][0]
                                        payload = point.payload or {}
                                        chunk_text = payload.get("text", "No text found")
                                        chunk_index = payload.get("chunk_index", "N/A")
                                        chunk_size = len(chunk_text)
                                        
                                        with st.container():
                                            st.markdown("#### 🎯 Sample Chunk from Winning Strategy")
                                            st.markdown(f"**Strategy:** {winner} | **Chunk #{chunk_index}** | **{chunk_size} characters**")
                                            # Show first 300 chars for readability
                                            preview = chunk_text[:300] + ("..." if len(chunk_text) > 300 else "")
                                            st.code(preview, language="text")
                                else:
                                    with st.container():
                                        st.markdown("#### 🎯 Sample Chunk")
                                        st.warning(f"Collection '{winner_collection}' not found")
                            except Exception as e:
                                with st.container():
                                    st.markdown("#### 🎯 Sample Chunk")
                                    st.warning(f"Could not retrieve sample chunk: {e}")
                        
                        # Box 3: Eval Metrics
                        with st.container():
                            st.markdown("#### 📊 Evaluation Metrics")
                            
                            # Get scores from session state or extract from result text directly
                            scores = st.session_state.get("eval_scores", [])
                            strategies_tested = st.session_state.get("strategies_tested", [])
                            
                            # If not in session state, try to extract directly from result text
                            if not scores or not strategies_tested:
                                import re
                                # Look for pattern: "strategy scored highest (avg_score X), followed by Y (Z), and Z last (W)"
                                # Or: "X (avg_score Y), Y (Z), Z (W)"
                                pattern = r'(\w+)\s+(?:strategy|collection)\s+(?:scored|avg_score)[:\s]+([\d.]+)'
                                matches = re.findall(pattern, result_text, re.IGNORECASE)
                                
                                if matches:
                                    metrics_dict = {}
                                    for strategy, score in matches:
                                        strategy_lower = strategy.lower()
                                        if strategy_lower in CHUNKING_STRATEGIES:
                                            metrics_dict[strategy_lower] = float(score)
                                    
                                    if metrics_dict:
                                        sorted_items = sorted(metrics_dict.items(), key=lambda x: x[1], reverse=True)
                                        scores = [score for _, score in sorted_items]
                                        strategies_tested = [strat for strat, _ in sorted_items]
                                
                                # Fallback: extract all decimal numbers and match to strategies mentioned
                                if not scores:
                                    # Find all 4-decimal numbers (likely scores)
                                    score_nums = re.findall(r'\b0\.\d{4}\b', result_text)
                                    if score_nums and strategies_found:
                                        scores = [float(s) for s in score_nums[:3]]
                                        strategies_tested = strategies_found[:3]
                            
                            if strategies_tested and scores and len(strategies_tested) == len(scores):
                                # Match strategies with scores
                                metrics_data = {}
                                for i, strategy in enumerate(strategies_tested):
                                    if i < len(scores):
                                        metrics_data[strategy] = scores[i]
                                
                                if metrics_data:
                                    # Sort by score descending
                                    sorted_metrics = sorted(metrics_data.items(), key=lambda x: x[1], reverse=True)
                                    
                                    # Display as columns for better layout
                                    cols = st.columns(len(sorted_metrics))
                                    for idx, (strategy, score) in enumerate(sorted_metrics):
                                        with cols[idx]:
                                            is_winner = idx == 0
                                            st.metric(
                                                label=f"{strategy.capitalize()}",
                                                value=f"{score:.4f}",
                                                delta="🏆 Winner" if is_winner else None
                                            )
                                else:
                                    st.warning("Could not match scores to strategies. Raw scores found: " + str(scores))
                            else:
                                # Show what we found for debugging
                                debug_info = f"Strategies: {strategies_tested}, Scores: {scores}"
                                st.warning(f"Could not extract metrics. {debug_info}")
                                # Show raw result text for debugging
                                with st.expander("Debug: View raw result text"):
                                    st.text(result_text)
                    
                    st.balloons()
                    
            except Exception as e:
                progress_bar.progress(0, text=f"❌ Error: {str(e)}")
                st.error(f"Experiment failed: {e}")
                import traceback
                st.code(traceback.format_exc())


if __name__ == "__main__":
    main()
