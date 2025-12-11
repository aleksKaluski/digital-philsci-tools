"""
Query sentence- and paragraph-level Milvus databases created by build_subcorpus_milvus.py

This script provides a flexible interface for querying subcorpus collections with:
- Configurable database and collection names
- Query input from text file (one query per line)
- Support for both sentence and paragraph level search
- Reciprocal Rank Fusion (RRF) for aggregating results across multiple queries
- Support for both standard sentence-transformers and SPECTER2 models
- Extensible result processing and output formats

Model Support:
    The script supports two types of embedding models:
    
    1. Standard sentence-transformers models (default):
       - Multi-qa-MiniLM-L6-cos-v1 (default, 384-dim)
       - All-MiniLM-L6-v2, all-mpnet-base-v2, etc.
       - Fast, general-purpose embeddings
    
    2. SPECTER2 models (requires 'adapters' library):
       - allenai/specter2_base (768-dim)
       - Trained specifically for scientific papers
       - Uses task-specific adapters (proximity, adhoc_query, etc.)
       - Install with: pip install adapters

Usage:
    # Query sentence collection with default model
    python query_subcorpus.py \\
        --db-name my_subcorpus \\
        --collection sentences \\
        --queries queries.txt \\
        --output results.json
    
    # Query paragraph collection with custom parameters
    python query_subcorpus.py \\
        --db-name my_subcorpus \\
        --collection paragraphs \\
        --queries queries.txt \\
        --limit 100 \\
        --output results.json
    
    # Use Reciprocal Rank Fusion to aggregate results across queries
    python query_subcorpus.py \\
        --db-name my_subcorpus \\
        --collection sentences \\
        --queries queries.txt \\
        --use-rrf \\
        --rrf-output-size 1000 \\
        --rrf-k 60 \\
        --output results.json
    
    # Use SPECTER2 model for scientific paper embeddings
    python query_subcorpus.py \\
        --db-name my_subcorpus \\
        --collection sentences \\
        --queries queries.txt \\
        --model allenai/specter2_base \\
        --output results.json
    
    # Use SPECTER2 with specific adapter for short text queries
    python query_subcorpus.py \\
        --db-name my_subcorpus \\
        --collection sentences \\
        --queries queries.txt \\
        --model allenai/specter2_base \\
        --adapter allenai/specter2_adhoc_query \\
        --output results.json
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from collections import defaultdict

import numpy as np
import pandas as pd
from pymilvus import connections, MilvusClient, db
from tqdm import tqdm

from model_adapter import UnifiedEmbedder


class Tee:
    """Redirect stdout to both terminal and log file."""
    def __init__(self, log_file):
        self.terminal = sys.stdout
        self.log = open(log_file, 'w', encoding='utf-8')
    
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()
    
    def flush(self):
        self.terminal.flush()
        self.log.flush()
    
    def close(self):
        self.log.close()

# Default configuration
DEFAULT_MILVUS_HOST = "localhost"
DEFAULT_MILVUS_PORT = 19530
DEFAULT_MODEL = "multi-qa-MiniLM-L6-cos-v1"
DEFAULT_LIMIT = 1000
DEFAULT_METRIC_TYPE = "COSINE"
DEFAULT_RRF_K = 60
DEFAULT_RRF_OUTPUT_SIZE = 1000


class SubcorpusQueryClient:
    """Client for querying subcorpus Milvus collections."""
    
    def __init__(self,
                 db_name: str,
                 model_name: str = DEFAULT_MODEL,
                 adapter: Optional[str] = None,
                 milvus_host: str = DEFAULT_MILVUS_HOST,
                 milvus_port: int = DEFAULT_MILVUS_PORT,
                 use_gpu: bool = True):
        """
        Initialize query client.
        
        Args:
            db_name: Name of the Milvus database
            model_name: Sentence transformer or SPECTER2 model name
            adapter: Adapter for SPECTER2 models (optional, auto-detected if not specified)
            milvus_host: Milvus server host
            milvus_port: Milvus server port
            use_gpu: Whether to use GPU for encoding
        """
        self.db_name = db_name
        self.model_name = model_name
        self.adapter = adapter
        self.milvus_host = milvus_host
        self.milvus_port = milvus_port
        self.use_gpu = use_gpu
        
        self.client = None
        self.encoder = None
    
    def connect(self):
        """Establish connection to Milvus."""
        print(f"Connecting to Milvus at {self.milvus_host}:{self.milvus_port}...")
        
        # Establish connection
        connections.connect(host=self.milvus_host, port=self.milvus_port)
        
        # Enable the database
        db.using_database(self.db_name)
        
        # Setup client
        self.client = MilvusClient(
            uri=f'http://{self.milvus_host}:{self.milvus_port}',
            token='root:Milvus',
            db_name=self.db_name
        )
        
        print(f"Connected to database: {self.db_name}")
    
    def load_encoder(self):
        """Load sentence transformer model."""
        print(f"Loading encoder model: {self.model_name}...")
        
        self.encoder = UnifiedEmbedder(
            model_name=self.model_name,
            adapter=self.adapter,
            device='cuda' if self.use_gpu else 'cpu'
        )
        
        print(f"Encoder loaded on {'cuda' if self.use_gpu else 'cpu'}")
    
    def encode_queries(self, queries: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Encode queries into embeddings.
        
        Args:
            queries: List of query strings
            batch_size: Batch size for encoding
            
        Returns:
            Array of query embeddings
        """
        print(f"Encoding {len(queries)} queries...")
        
        embeddings = self.encoder.encode(
            queries,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        return embeddings
    
    def query_collection(self,
                        collection_name: str,
                        query_embeddings: np.ndarray,
                        limit: int = DEFAULT_LIMIT,
                        metric_type: str = DEFAULT_METRIC_TYPE,
                        output_fields: Optional[List[str]] = None,
                        release_after: bool = True) -> List[List[Dict]]:
        """
        Query a Milvus collection.
        
        Args:
            collection_name: Name of the collection to query
            query_embeddings: Array of query embeddings
            limit: Maximum number of results per query
            metric_type: Distance metric type (COSINE, L2, IP)
            output_fields: Fields to return in results
            release_after: Whether to release collection after querying (default: True)
            
        Returns:
            List of results for each query
        """
        print(f"Querying collection: {collection_name}")
        
        # Load collection if not already loaded
        load_state = self.client.get_load_state(collection_name=collection_name)
        was_loaded = load_state.get("state") == "Loaded"
        
        if not was_loaded:
            print(f"Loading collection {collection_name}...")
            self.client.load_collection(collection_name=collection_name)
        
        # Default output fields if not specified
        if output_fields is None:
            # Use appropriate fields based on collection type
            if 'paragraph' in collection_name.lower():
                output_fields = ["corpusid", "paragraph_number", "paragraph_indices"]
            else:
                output_fields = ["corpusid", "sentence_number", "sentence_indices"]
        
        try:
            # Perform search
            results = self.client.search(
                collection_name=collection_name,
                data=query_embeddings.tolist(),
                limit=limit,
                search_params={"metric_type": metric_type, "params": {}},
                output_fields=output_fields
            )
            
            print(f"Retrieved results for {len(results)} queries")
            
            return results
            
        finally:
            # Release collection to free up resources (unless it was already loaded)
            if release_after and not was_loaded:
                print(f"Releasing collection {collection_name} to free up resources...")
                self.client.release_collection(collection_name=collection_name)
                print("Collection released")
    
    def format_results(self,
                      queries: List[str],
                      results: List[List[Dict]],
                      collection_name: str) -> pd.DataFrame:
        """
        Format search results into a DataFrame.
        
        Args:
            queries: Original query strings
            results: Search results from Milvus
            collection_name: Name of the queried collection
            
        Returns:
            DataFrame with formatted results
        """
        print("Formatting results...")
        
        formatted_results = []
        
        for query_idx, (query, query_results) in enumerate(zip(queries, results)):
            for rank, result in enumerate(query_results):
                entity = result.get('entity', {})
                
                # Debug: Print first result to see structure
                if query_idx == 0 and rank == 0:
                    print(f"\nDEBUG: First result structure:")
                    print(f"  Full result keys: {result.keys()}")
                    print(f"  Entity keys: {entity.keys()}")
                    print(f"  Entity content: {entity}")
                    print()
                
                formatted_result = {
                    'query_idx': query_idx,
                    'query': query,
                    'rank': rank + 1,
                    'distance': result.get('distance', 0.0),
                    'corpusid': entity.get('corpusid'),
                    'collection': collection_name,
                }
                
                # Add all entity fields
                # Note: Milvus returns ARRAY fields as protobuf RepeatedScalarContainer objects
                # which don't serialize properly to JSON. Convert to plain Python types.
                for key, value in entity.items():
                    if key not in formatted_result:
                        # Check if it's a protobuf container (has __class__.__module__ starting with 'google')
                        if hasattr(value, '__class__') and 'google' in value.__class__.__module__:
                            # It's a protobuf object - convert by iterating and casting
                            formatted_result[key] = [int(x) for x in value]
                        elif hasattr(value, '__iter__') and not isinstance(value, (str, dict, list)):
                            # Other iterable - convert to list
                            formatted_result[key] = list(value)
                        else:
                            formatted_result[key] = value
                
                # Debug: Check conversion result
                if query_idx == 0 and rank == 0 and 'sentence_indices' in formatted_result:
                    print(f"DEBUG: After list comprehension conversion:")
                    print(f"  sentence_indices value: {formatted_result['sentence_indices']}")
                    print(f"  Type: {type(formatted_result['sentence_indices'])}")
                    print()
                
                formatted_results.append(formatted_result)
        
        df = pd.DataFrame(formatted_results)
        
        print(f"Formatted {len(df)} results")
        
        return df
    
    def reciprocal_rank_fusion(self,
                              results: List[List[Dict]],
                              output_size: int = DEFAULT_RRF_OUTPUT_SIZE,
                              k: int = DEFAULT_RRF_K) -> pd.DataFrame:
        """
        Aggregate multiple query results using Reciprocal Rank Fusion.
        
        RRF formula: RRF_score(d) = Σ(1 / (k + rank_i(d)))
        where k is a constant (typically 60) and rank_i(d) is the rank of document d in query i.
        
        Note: RRF is computed per individual sentence/paragraph, not per document.
        Multiple sentences/paragraphs from the same article are kept separate.
        
        Args:
            results: List of search results from multiple queries
            output_size: Number of top results to return
            k: RRF constant (default 60)
            
        Returns:
            DataFrame with aggregated results sorted by RRF score
        """
        print(f"Applying Reciprocal Rank Fusion (k={k}, output_size={output_size})...")
        
        rrf_scores = defaultdict(float)
        entity_data = {}  # Store entity data for each unique sentence/paragraph
        
        for query_idx, query_results in enumerate(results):
            for rank, result in enumerate(query_results, start=1):
                entity = result.get('entity', {})
                corpus_id = entity.get('corpusid')
                
                # Create unique key for each sentence/paragraph
                # Use sentence_number or paragraph_number to distinguish multiple hits from same document
                if 'sentence_number' in entity:
                    unique_key = (corpus_id, entity.get('sentence_number'))
                elif 'paragraph_number' in entity:
                    unique_key = (corpus_id, entity.get('paragraph_number'))
                else:
                    # Fallback to corpus_id only if no sentence/paragraph number
                    unique_key = (corpus_id, 0)
                
                if corpus_id is not None:
                    # RRF formula: 1 / (k + rank)
                    rrf_scores[unique_key] += 1.0 / (k + rank)
                    
                    # Store entity data (keep first occurrence)
                    if unique_key not in entity_data:
                        entity_data[unique_key] = entity
        
        # Sort by RRF score (descending) and take top N
        sorted_results = sorted(
            [(key, score) for key, score in rrf_scores.items()],
            key=lambda x: x[1],
            reverse=True
        )[:output_size]
        
        unique_docs = len(set(key[0] for key in rrf_scores.keys()))
        print(f"Selected top {len(sorted_results)} sentences/paragraphs from {unique_docs} unique documents")
        
        # Format results as DataFrame
        formatted_results = []
        for rank, (unique_key, rrf_score) in enumerate(sorted_results, start=1):
            entity = entity_data[unique_key]
            
            formatted_result = {
                'rank': rank,
                'rrf_score': rrf_score,
                'corpusid': unique_key[0],  # Extract corpus_id from tuple
            }
            
            # Add all entity fields, converting protobuf objects
            for key, value in entity.items():
                if key not in formatted_result:
                    if hasattr(value, '__class__') and 'google' in value.__class__.__module__:
                        formatted_result[key] = [int(x) for x in value]
                    elif hasattr(value, '__iter__') and not isinstance(value, (str, dict, list)):
                        formatted_result[key] = list(value)
                    else:
                        formatted_result[key] = value
            
            formatted_results.append(formatted_result)
        
        return pd.DataFrame(formatted_results)
    
    def close(self):
        """Close connections."""
        if self.client:
            connections.disconnect(alias="default")
            print("Disconnected from Milvus")


def load_queries_from_file(filepath: str) -> List[str]:
    """
    Load queries from text file (one query per line).
    
    Args:
        filepath: Path to queries file
        
    Returns:
        List of query strings
    """
    print(f"Loading queries from: {filepath}")
    
    queries = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):  # Skip empty lines and comments
                queries.append(line)
    
    print(f"Loaded {len(queries)} queries")
    
    return queries


def save_results(df: pd.DataFrame, output_path: str, format: str = 'json'):
    """
    Save results to file.
    
    Args:
        df: Results DataFrame
        output_path: Output file path
        format: Output format (json, csv, or parquet)
    """
    print(f"Saving results to: {output_path}")
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if format == 'json' or output_path.suffix == '.json':
        # Convert DataFrame to dict and use Python's json module to handle protobuf objects
        records = df.to_dict(orient='records')
        
        # Debug: Check first record before final conversion
        if len(records) > 0 and 'sentence_indices' in records[0]:
            print(f"\nDEBUG: Before final JSON conversion:")
            print(f"  First record sentence_indices: {records[0]['sentence_indices']}")
            print(f"  Type: {type(records[0]['sentence_indices'])}")
            print()
        
        # Ensure any remaining protobuf objects are converted to lists
        for record in records:
            for key, value in list(record.items()):
                # Check if it's a protobuf object
                if hasattr(value, '__class__') and 'google' in value.__class__.__module__:
                    # Convert protobuf container to plain Python list
                    record[key] = [int(x) for x in value]
                elif hasattr(value, '__iter__') and not isinstance(value, (str, dict, list)):
                    record[key] = list(value)
        
        # Debug: Check first record after final conversion
        if len(records) > 0 and 'sentence_indices' in records[0]:
            print(f"DEBUG: After final conversion:")
            print(f"  First record sentence_indices: {records[0]['sentence_indices']}")
            print(f"  Type: {type(records[0]['sentence_indices'])}")
            print()
        
        with open(output_path, 'w') as f:
            json.dump(records, f, indent=2)
    elif format == 'csv' or output_path.suffix == '.csv':
        df.to_csv(output_path, index=False)
    elif format == 'parquet' or output_path.suffix == '.parquet':
        df.to_parquet(output_path, index=False)
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    print(f"Results saved ({len(df)} rows)")


def print_summary(df: pd.DataFrame):
    """Print summary statistics of results."""
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    
    print(f"Total results: {len(df)}")
    
    # Handle both regular and RRF results
    if 'query_idx' in df.columns:
        print(f"Unique queries: {df['query_idx'].nunique()}")
    
    print(f"Unique documents: {df['corpusid'].nunique()}")
    
    # Show distance statistics for regular queries
    if 'distance' in df.columns:
        print(f"\nDistance statistics:")
        print(f"  Mean: {df['distance'].mean():.4f}")
        print(f"  Std:  {df['distance'].std():.4f}")
        print(f"  Min:  {df['distance'].min():.4f}")
        print(f"  Max:  {df['distance'].max():.4f}")
        
        print("\nTop 5 results by distance:")
        top_cols = ['distance', 'corpusid', 'rank']
        if 'query' in df.columns:
            top_cols.insert(0, 'query')
        print(df.nlargest(5, 'distance')[top_cols].to_string(index=False))
    
    # Show RRF score statistics for RRF results
    elif 'rrf_score' in df.columns:
        print(f"\nRRF score statistics:")
        print(f"  Mean: {df['rrf_score'].mean():.4f}")
        print(f"  Std:  {df['rrf_score'].std():.4f}")
        print(f"  Min:  {df['rrf_score'].min():.4f}")
        print(f"  Max:  {df['rrf_score'].max():.4f}")
        
        print("\nTop 5 results by RRF score:")
        print(df.head(5)[['rank', 'rrf_score', 'corpusid']].to_string(index=False))
    
    print("=" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Query subcorpus Milvus collections",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Query sentence collection
  python query_subcorpus.py \\
    --db-name my_subcorpus \\
    --collection sentences \\
    --queries queries.txt \\
    --output results.json
  
  # Query paragraph collection with more results
  python query_subcorpus.py \\
    --db-name my_subcorpus \\
    --collection paragraphs \\
    --queries queries.txt \\
    --limit 500 \\
    --output results.csv
  
  # Query with custom model
  python query_subcorpus.py \\
    --db-name my_subcorpus \\
    --collection sentences \\
    --queries queries.txt \\
    --model sentence-transformers/all-MiniLM-L6-v2 \\
    --output results.json
        """
    )
    
    # Required arguments
    parser.add_argument(
        '--db-name',
        type=str,
        required=True,
        help='Name of the Milvus database'
    )
    
    parser.add_argument(
        '--collection',
        type=str,
        required=True,
        help='Name of the collection to query (e.g., sentences, paragraphs)'
    )
    
    parser.add_argument(
        '--queries',
        type=str,
        required=True,
        help='Path to queries file (one query per line)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Path to output file (json, csv, or parquet)'
    )
    
    # Optional arguments
    parser.add_argument(
        '--model',
        type=str,
        default=DEFAULT_MODEL,
        help=f'Sentence transformer or SPECTER2 model (default: {DEFAULT_MODEL})'
    )
    
    parser.add_argument(
        '--adapter',
        type=str,
        default=None,
        help='Adapter for SPECTER2 models (e.g., allenai/specter2 for proximity). Auto-detected if not specified.'
    )
    
    parser.add_argument(
        '--limit',
        type=int,
        default=DEFAULT_LIMIT,
        help=f'Maximum results per query (default: {DEFAULT_LIMIT})'
    )
    
    parser.add_argument(
        '--milvus-host',
        type=str,
        default=DEFAULT_MILVUS_HOST,
        help=f'Milvus server host (default: {DEFAULT_MILVUS_HOST})'
    )
    
    parser.add_argument(
        '--milvus-port',
        type=int,
        default=DEFAULT_MILVUS_PORT,
        help=f'Milvus server port (default: {DEFAULT_MILVUS_PORT})'
    )
    
    parser.add_argument(
        '--metric-type',
        type=str,
        default=DEFAULT_METRIC_TYPE,
        choices=['COSINE', 'L2', 'IP'],
        help=f'Distance metric type (default: {DEFAULT_METRIC_TYPE})'
    )
    
    parser.add_argument(
        '--no-gpu',
        action='store_true',
        help='Disable GPU for encoding'
    )
    
    parser.add_argument(
        '--output-fields',
        type=str,
        nargs='+',
        default=None,
        help='Custom output fields to retrieve (default: corpusid, sentence_number, sentence_indices)'
    )
    
    parser.add_argument(
        '--no-summary',
        action='store_true',
        help='Skip printing results summary'
    )
    
    parser.add_argument(
        '--use-rrf',
        action='store_true',
        help='Use Reciprocal Rank Fusion to aggregate results across queries'
    )
    
    parser.add_argument(
        '--rrf-k',
        type=int,
        default=DEFAULT_RRF_K,
        help=f'RRF constant k (default: {DEFAULT_RRF_K})'
    )
    
    parser.add_argument(
        '--rrf-output-size',
        type=int,
        default=DEFAULT_RRF_OUTPUT_SIZE,
        help=f'Number of top results to return when using RRF (default: {DEFAULT_RRF_OUTPUT_SIZE})'
    )
    
    parser.add_argument(
        '--log-dir',
        type=str,
        default='.',
        help='Directory to save log files (default: current directory)'
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("QUERY SUBCORPUS MILVUS COLLECTIONS")
    print("=" * 70)
    print(f"Database: {args.db_name}")
    print(f"Collection: {args.collection}")
    print(f"Queries file: {args.queries}")
    print(f"Output file: {args.output}")
    print(f"Model: {args.model}")
    print(f"Results per query: {args.limit}")
    print("=" * 70 + "\n")
    
    try:
        # Load queries
        queries = load_queries_from_file(args.queries)
        
        if not queries:
            print("Error: No queries found in file")
            return 1
        
        # Initialize client
        client = SubcorpusQueryClient(
            db_name=args.db_name,
            model_name=args.model,
            adapter=args.adapter,
            milvus_host=args.milvus_host,
            milvus_port=args.milvus_port,
            use_gpu=not args.no_gpu
        )
        
        # Connect to Milvus
        client.connect()
        
        # Load encoder
        client.load_encoder()
        
        # Encode queries
        query_embeddings = client.encode_queries(queries)
        
        # Query collection
        results = client.query_collection(
            collection_name=args.collection,
            query_embeddings=query_embeddings,
            limit=args.limit,
            metric_type=args.metric_type,
            output_fields=args.output_fields
        )
        
        # Format results (with or without RRF)
        if args.use_rrf:
            df = client.reciprocal_rank_fusion(
                results=results,
                output_size=args.rrf_output_size,
                k=args.rrf_k
            )
        else:
            df = client.format_results(
                queries=queries,
                results=results,
                collection_name=args.collection
            )
        
        # Save results
        save_results(df, args.output)
        
        # Print summary
        if not args.no_summary:
            print_summary(df)
        
        # Clean up
        client.close()
        
        print("\n" + "=" * 70)
        print("Query completed successfully!")
        print("=" * 70)
        
        return 0
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    # Parse args first to get log directory
    import sys
    temp_args = sys.argv[1:]
    log_dir = '.'
    for i, arg in enumerate(temp_args):
        if arg == '--log-dir' and i + 1 < len(temp_args):
            log_dir = temp_args[i + 1]
            break
    
    # Setup logging to file
    from pathlib import Path
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = str(Path(log_dir) / f'query_subcorpus_{timestamp}.log')
    tee = Tee(log_file)
    sys.stdout = tee
    
    print(f"Logging output to: {log_file}\n")
    
    try:
        exit_code = main()
        print(f"\n{'='*60}")
        print(f"Script finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*60}")
    finally:
        sys.stdout = tee.terminal
        tee.close()
    
    exit(exit_code)
