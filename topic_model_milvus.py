#!/usr/bin/env python3
"""
Topic Modeling from Milvus Collections

CLI tool for building topic models from Milvus database collections.
Uses pre-computed embeddings stored in Milvus to avoid recomputation.

This tool requires a subcorpus file to efficiently handle large collections (>1M items) 
by iterating through corpus IDs rather than using offset-based pagination, which is 
limited to 16,384 items in Milvus. Accepts both JSON (.json from query_subcorpus.py) 
and pickle (.pkl from query_milvus_rrf.py) formats.

Two main use cases:

1. **Topic modeling on full subcorpus** (from query_milvus_rrf.py):
   Build topic models on all papers in a subcorpus, analyzing the entire collection
   to discover broad themes and patterns across the full dataset.
   Use the .pkl output file from query_milvus_rrf.py.
   Loads ALL sentences/paragraphs from each corpus_id in the file.
   
2. **Topic modeling on query results** (from query_subcorpus.py):
   Build topic models on specific query results (e.g., papers matching "quantum entanglement"),
   allowing focused analysis of particular research areas or concepts.
   Use the .json output file from query_subcorpus.py (default output format).
   Loads ONLY the specific sentences/paragraphs identified in the JSON file (by corpusid 
   + sentence_number/paragraph_number). This allows modeling on exactly the search results.
   The --filter parameter can further refine results (e.g., by RRF score thresholds).

Usage:
    # Model full subcorpus (all papers from query_milvus_rrf.py)
    python topic_model_milvus.py \\
        --db-name my_subcorpus \\
        --collection paragraphs \\
        --subcorpus-file subcorpus_results.pkl \\
        --s2orc-path /path/to/s2orc \\
        --output-dir ./models
    
    # Model query results (from query_subcorpus.py JSON output)
    python topic_model_milvus.py \\
        --db-name my_subcorpus \\
        --collection paragraphs \\
        --subcorpus-file query_results_quantum.json \\
        --s2orc-path /path/to/s2orc \\
        --output-dir ./models/quantum_topics
    
    # Model query results with RRF score filter (high-relevance only)
    python topic_model_milvus.py \\
        --db-name my_subcorpus \\
        --collection paragraphs \\
        --subcorpus-file query_results_quantum.json \\
        --s2orc-path /path/to/s2orc \\
        --filter "rrf_score > 0.5" \\
        --output-dir ./models/quantum_high_relevance
    
    # Model with custom parameters
    python topic_model_milvus.py \\
        --db-name my_subcorpus \\
        --collection sentences \\
        --subcorpus-file query_results.json \\
        --s2orc-path /path/to/s2orc \\
        --nr-topics 20 \\
        --min-cluster-size 30 \\
        --limit 5000 \\
        --output-dir ./models
    
    # Optimize topic number for query results
    python topic_model_milvus.py \\
        --db-name my_subcorpus \\
        --collection paragraphs \\
        --subcorpus-file query_results.json \\
        --s2orc-path /path/to/s2orc \\
        --optimize \\
        --min-topics 10 \\
        --max-topics 50 \\
        --n-trials 20
    
    # Optimize and build best model
    python topic_model_milvus.py \\
        --db-name my_subcorpus \\
        --collection paragraphs \\
        --subcorpus-file query_results.json \\
        --s2orc-path /path/to/s2orc \\
        --optimize \\
        --build-best \\
        --min-topics 10 \\
        --max-topics 50

Requirements:
    - pymilvus: pip install pymilvus
    - pymongo: pip install pymongo
    - Subcorpus file from:
      * query_milvus_rrf.py (outputs .pkl by default), OR
      * query_subcorpus.py (outputs .json by default)
    - Other dependencies from environment_bertopic.yml

Note:
    The subcorpus file is required to efficiently handle large Milvus collections
    (typically >1M embeddings). The tool iterates through corpus IDs from the
    subcorpus file, querying all sentences/paragraphs for each document separately,
    avoiding Milvus's 16,384 offset+limit window constraint.
    
    Both JSON (.json from query_subcorpus.py) and pickle (.pkl from query_milvus_rrf.py) 
    formats are supported. JSON files contain 'corpusid' field in each record, while 
    pickle files contain 'corpus_ids' list.
"""

import argparse
import sys
import json
from pathlib import Path
from typing import Dict, Any, Optional

from topic_modeling_analysis import (
    build_topic_model_from_milvus,
    ModelConfig,
    EnvironmentSetup,
    MilvusEmbeddingLoader
)


def optimize_topic_number(
    db_name: str,
    collection_name: str,
    subcorpus_file: str,
    s2orc_path: str,
    milvus_host: str = 'localhost',
    milvus_port: int = 19530,
    mongo_host: str = 'localhost',
    mongo_port: int = 27017,
    limit: Optional[int] = None,
    filter_expr: Optional[str] = None,
    output_dir: str = './topic_models',
    min_topics: int = 5,
    max_topics: int = 100,
    n_trials: int = 20,
    base_config: Optional[ModelConfig] = None,
    use_gpu: bool = True,
    quiet: bool = False
) -> Dict[str, Any]:
    """
    Find optimal number of topics using Bayesian optimization.
    
    This uses a more efficient approach than grid search by:
    1. Using Bayesian optimization to intelligently sample the parameter space
    2. Early stopping for clearly suboptimal configurations
    3. Focusing search around promising regions
    
    Args:
        db_name: Name of the Milvus database
        collection_name: Name of the collection
        s2orc_path: Path to S2ORC files
        milvus_host: Milvus server host
        milvus_port: Milvus server port
        mongo_host: MongoDB server host
        mongo_port: MongoDB server port
        limit: Maximum number of documents to use
        filter_expr: Optional Milvus filter expression
        output_dir: Output directory for results
        min_topics: Minimum number of topics to try
        max_topics: Maximum number of topics to try
        n_trials: Number of optimization trials
        base_config: Base ModelConfig to use (other params taken from this)
        use_gpu: Whether to use GPU
        quiet: Suppress progress messages
        
    Returns:
        Dictionary with optimization results including best config and all trials
    """
    try:
        from skopt import gp_minimize
        from skopt.space import Integer
        from skopt.utils import use_named_args
    except ImportError:
        if not quiet:
            print("Bayesian optimization requires scikit-optimize.")
            print("Install with: pip install scikit-optimize")
            print("Falling back to grid search...")
        return optimize_topic_number_grid(
            db_name=db_name,
            collection_name=collection_name,
            subcorpus_file=subcorpus_file,
            s2orc_path=s2orc_path,
            milvus_host=milvus_host,
            milvus_port=milvus_port,
            mongo_host=mongo_host,
            mongo_port=mongo_port,
            limit=limit,
            filter_expr=filter_expr,
            output_dir=output_dir,
            min_topics=min_topics,
            max_topics=max_topics,
            base_config=base_config,
            use_gpu=use_gpu,
            quiet=quiet
        )
    
    if not quiet:
        print("="*80)
        print("TOPIC NUMBER OPTIMIZATION (Bayesian)")
        print("="*80)
        print(f"Search range: {min_topics} to {max_topics} topics")
        print(f"Trials: {n_trials}")
        print("="*80)
        print()
    
    # Load data once
    if not quiet:
        print("Loading embeddings and documents...")
    
    loader = MilvusEmbeddingLoader(
        db_name=db_name,
        collection_name=collection_name,
        subcorpus_file=subcorpus_file,
        milvus_host=milvus_host,
        milvus_port=milvus_port,
        mongo_host=mongo_host,
        mongo_port=mongo_port
    )
    
    try:
        loader.connect()
        embeddings, metadata_df = loader.load_embeddings(limit=limit, filter_expr=filter_expr)
        
        # Check for cached documents
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        cache_file = output_path / f"docs_cache_{db_name}_{collection_name}.pkl"
        
        if cache_file.exists():
            if not quiet:
                print(f"Loading cached documents from {cache_file}...")
            import pickle
            with open(cache_file, 'rb') as f:
                cached_data = pickle.load(f)
            
            # Verify cache matches current metadata
            cache_valid = False
            if 'metadata_hash' in cached_data:
                import hashlib
                metadata_str = str(sorted(metadata_df[['corpusid']].values.tolist()))
                current_hash = hashlib.md5(metadata_str.encode()).hexdigest()
                cache_valid = (cached_data['metadata_hash'] == current_hash)
            
            if cache_valid:
                if not quiet:
                    print("✓ Cache is valid, using cached documents")
                docs = cached_data['docs']
            else:
                if not quiet:
                    print("✗ Cache is outdated, reloading documents...")
                docs = None
        else:
            docs = None
        
        # Load documents if not cached
        if docs is None:
            docs = loader.get_documents_from_mongo(metadata_df, s2orc_path=s2orc_path)
            
            # Save to cache
            if not quiet:
                print(f"Saving documents to cache: {cache_file}")
            import pickle
            import hashlib
            metadata_str = str(sorted(metadata_df[['corpusid']].values.tolist()))
            metadata_hash = hashlib.md5(metadata_str.encode()).hexdigest()
            
            with open(cache_file, 'wb') as f:
                pickle.dump({
                    'docs': docs,
                    'metadata_hash': metadata_hash
                }, f)
            if not quiet:
                print("✓ Cache saved")
    finally:
        loader.close()
    
    # Filter empty documents
    valid_indices = [i for i, doc in enumerate(docs) if doc.strip()]
    if len(valid_indices) < len(docs):
        if not quiet:
            print(f"Filtered {len(docs) - len(valid_indices)} empty documents")
        docs = [docs[i] for i in valid_indices]
        embeddings = embeddings[valid_indices]
        metadata_df = metadata_df.iloc[valid_indices].reset_index(drop=True)
    
    if not quiet:
        print(f"Using {len(docs)} documents for optimization\n")
    
    # Import required modules
    from topic_modeling_analysis import TopicModeler, ModelEvaluator
    
    # First, test automatic topic discovery to understand the natural number of topics
    if not quiet:
        print("="*80)
        print("PRELIMINARY TEST: Automatic topic discovery")
        print("="*80)
        print("Testing with nr_topics=None to see how many topics are naturally discovered...")
        print()
    
    try:
        config_auto = ModelConfig(
            nr_topics=None,
            use_gpu=use_gpu,
            output_dir=Path(output_dir)
        )
        modeler_auto = TopicModeler(config_auto)
        model_auto, topics_auto, _ = modeler_auto.fit(docs, embeddings)
        
        num_auto_topics = len(model_auto.get_topic_info()) - 1  # Exclude outlier topic (-1)
        if not quiet:
            print(f"\n✓ Automatic discovery found {num_auto_topics} topics")
            print(f"  This suggests the natural granularity of your corpus.")
            print(f"  Optimization will test reducing this to {min_topics}-{max_topics} topics.\n")
            
            if num_auto_topics < max_topics:
                print(f"⚠️  WARNING: Only {num_auto_topics} topics found automatically,")
                print(f"   which is less than max_topics={max_topics}.")
                print(f"   BERTopic cannot increase topics beyond what's naturally found.")
                print(f"   Consider adjusting clustering parameters (min_cluster_size, umap settings)")
                print(f"   or reducing max_topics to {num_auto_topics}.\n")
        
        # Clean up
        del model_auto, topics_auto, modeler_auto
        EnvironmentSetup.cleanup_memory()
        
    except Exception as e:
        if not quiet:
            print(f"✗ Automatic discovery failed: {e}")
            print("Proceeding with optimization anyway...\n")
    
    if not quiet:
        print("="*80)
        print("OPTIMIZATION: Testing topic number range")
        print("="*80)
        print()
    
    # Track all trials
    trial_results = []
    best_score = float('-inf')
    best_nr_topics = None
    
    # Define search space
    search_space = [Integer(min_topics, max_topics, name='nr_topics')]
    
    # Objective function
    @use_named_args(search_space)
    def objective(nr_topics):
        nonlocal best_score, best_nr_topics
        
        # Convert numpy int to Python int (BERTopic requires Python int or 'auto')
        nr_topics = int(nr_topics)
        
        if not quiet:
            print(f"\nTrial {len(trial_results) + 1}/{n_trials}: Testing {nr_topics} topics...")
        
        try:
            # Create config
            if base_config:
                config = ModelConfig(
                    embedding_model_name=base_config.embedding_model_name,
                    nr_topics=nr_topics,
                    hdbscan_min_cluster_size=base_config.hdbscan_min_cluster_size,
                    umap_n_neighbors=base_config.umap_n_neighbors,
                    umap_n_components=base_config.umap_n_components,
                    use_gpu=use_gpu,
                    output_dir=Path(output_dir)
                )
            else:
                config = ModelConfig(
                    nr_topics=nr_topics,
                    use_gpu=use_gpu,
                    output_dir=Path(output_dir)
                )
            
            # Build model
            modeler = TopicModeler(config)
            model, topics, probs = modeler.fit(docs, embeddings)
            
            # Evaluate
            evaluator = ModelEvaluator()
            coherence = evaluator.evaluate(docs, model)
            
            # Get average coherence (u_mass or c_v, whichever is available)
            if isinstance(coherence, dict):
                score = coherence.get('c_v', coherence.get('u_mass', 0.0))
            else:
                score = coherence
            
            # Track results
            trial_results.append({
                'nr_topics': nr_topics,
                'coherence': coherence,
                'score': score
            })
            
            # Update best
            if score > best_score:
                best_score = score
                best_nr_topics = nr_topics
                if not quiet:
                    print(f"  ✓ New best! Topics: {nr_topics}, Score: {score:.4f}")
            elif not quiet:
                print(f"  Score: {score:.4f}")
            
            # Cleanup
            del model, topics, probs
            EnvironmentSetup.cleanup_memory()
            
            # Return negative score for minimization
            return -score
            
        except Exception as e:
            if not quiet:
                print(f"  ✗ Error: {e}")
            trial_results.append({
                'nr_topics': nr_topics,
                'coherence': None,
                'score': float('-inf'),
                'error': str(e)
            })
            return float('inf')  # Large penalty for errors
    
    # Run optimization
    result = gp_minimize(
        objective,
        search_space,
        n_calls=n_trials,
        random_state=42,
        verbose=False
    )
    
    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    results_file = output_path / f"optimization_results_{db_name}_{collection_name}.json"
    with open(results_file, 'w') as f:
        json.dump({
            'best_nr_topics': best_nr_topics,
            'best_score': best_score,
            'search_range': [min_topics, max_topics],
            'n_trials': n_trials,
            'all_trials': trial_results,
            'optimization_method': 'bayesian'
        }, f, indent=2)
    
    if not quiet:
        print("\n" + "="*80)
        print("OPTIMIZATION COMPLETE")
        print("="*80)
        print(f"Best number of topics: {best_nr_topics}")
        print(f"Best coherence score: {best_score:.4f}")
        print(f"Results saved to: {results_file}")
        print("="*80 + "\n")
    
    return {
        'best_nr_topics': best_nr_topics,
        'best_score': best_score,
        'all_trials': trial_results,
        'results_file': results_file
    }


def optimize_topic_number_grid(
    db_name: str,
    collection_name: str,
    subcorpus_file: str,
    s2orc_path: str,
    milvus_host: str = 'localhost',
    milvus_port: int = 19530,
    mongo_host: str = 'localhost',
    mongo_port: int = 27017,
    limit: Optional[int] = None,
    filter_expr: Optional[str] = None,
    output_dir: str = './topic_models',
    min_topics: int = 5,
    max_topics: int = 100,
    step: int = 5,
    base_config: Optional[ModelConfig] = None,
    use_gpu: bool = True,
    quiet: bool = False
) -> Dict[str, Any]:
    """
    Find optimal number of topics using adaptive grid search.
    
    This uses a coarse-to-fine strategy:
    1. Coarse search with larger step size
    2. Fine search around the best region
    
    Args:
        db_name: Name of the Milvus database
        collection_name: Name of the collection
        s2orc_path: Path to S2ORC files
        milvus_host: Milvus server host
        milvus_port: Milvus server port
        mongo_host: MongoDB server host
        mongo_port: MongoDB server port
        limit: Maximum number of documents to use
        filter_expr: Optional Milvus filter expression
        output_dir: Output directory for results
        min_topics: Minimum number of topics to try
        max_topics: Maximum number of topics to try
        step: Step size for grid search
        base_config: Base ModelConfig to use
        use_gpu: Whether to use GPU
        quiet: Suppress progress messages
        
    Returns:
        Dictionary with optimization results
    """
    if not quiet:
        print("="*80)
        print("TOPIC NUMBER OPTIMIZATION (Grid Search)")
        print("="*80)
        print(f"Search range: {min_topics} to {max_topics} topics (step={step})")
        print("="*80)
        print()
    
    # Load data once
    if not quiet:
        print("Loading embeddings and documents...")
    
    loader = MilvusEmbeddingLoader(
        db_name=db_name,
        collection_name=collection_name,
        subcorpus_file=subcorpus_file,
        milvus_host=milvus_host,
        milvus_port=milvus_port,
        mongo_host=mongo_host,
        mongo_port=mongo_port
    )
    
    try:
        loader.connect()
        embeddings, metadata_df = loader.load_embeddings(limit=limit, filter_expr=filter_expr)
        
        # Check for cached documents
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        cache_file = output_path / f"docs_cache_{db_name}_{collection_name}.pkl"
        
        if cache_file.exists():
            if not quiet:
                print(f"Loading cached documents from {cache_file}...")
            import pickle
            with open(cache_file, 'rb') as f:
                cached_data = pickle.load(f)
            
            # Verify cache matches current metadata
            cache_valid = False
            if 'metadata_hash' in cached_data:
                import hashlib
                metadata_str = str(sorted(metadata_df[['corpusid']].values.tolist()))
                current_hash = hashlib.md5(metadata_str.encode()).hexdigest()
                cache_valid = (cached_data['metadata_hash'] == current_hash)
            
            if cache_valid:
                if not quiet:
                    print("✓ Cache is valid, using cached documents")
                docs = cached_data['docs']
            else:
                if not quiet:
                    print("✗ Cache is outdated, reloading documents...")
                docs = None
        else:
            docs = None
        
        # Load documents if not cached
        if docs is None:
            docs = loader.get_documents_from_mongo(metadata_df, s2orc_path=s2orc_path)
            
            # Save to cache
            if not quiet:
                print(f"Saving documents to cache: {cache_file}")
            import pickle
            import hashlib
            metadata_str = str(sorted(metadata_df[['corpusid']].values.tolist()))
            metadata_hash = hashlib.md5(metadata_str.encode()).hexdigest()
            
            with open(cache_file, 'wb') as f:
                pickle.dump({
                    'docs': docs,
                    'metadata_hash': metadata_hash
                }, f)
            if not quiet:
                print("✓ Cache saved")
    finally:
        loader.close()
    
    # Filter empty documents
    valid_indices = [i for i, doc in enumerate(docs) if doc.strip()]
    if len(valid_indices) < len(docs):
        if not quiet:
            print(f"Filtered {len(docs) - len(valid_indices)} empty documents")
        docs = [docs[i] for i in valid_indices]
        embeddings = embeddings[valid_indices]
        metadata_df = metadata_df.iloc[valid_indices].reset_index(drop=True)
    
    if not quiet:
        print(f"Using {len(docs)} documents for optimization\n")
    
    # Import required modules
    from topic_modeling_analysis import TopicModeler, ModelEvaluator
    import numpy as np
    
    # Track all trials
    trial_results = []
    best_score = float('-inf')
    best_nr_topics = None
    
    # Coarse search
    topic_range = list(range(min_topics, max_topics + 1, step))
    
    for nr_topics in topic_range:
        if not quiet:
            print(f"\nTesting {nr_topics} topics...")
        
        try:
            # Create config
            if base_config:
                config = ModelConfig(
                    embedding_model_name=base_config.embedding_model_name,
                    nr_topics=nr_topics,
                    hdbscan_min_cluster_size=base_config.hdbscan_min_cluster_size,
                    umap_n_neighbors=base_config.umap_n_neighbors,
                    umap_n_components=base_config.umap_n_components,
                    use_gpu=use_gpu,
                    output_dir=Path(output_dir)
                )
            else:
                config = ModelConfig(
                    nr_topics=nr_topics,
                    use_gpu=use_gpu,
                    output_dir=Path(output_dir)
                )
            
            # Build model
            modeler = TopicModeler(config)
            model, topics, probs = modeler.fit(docs, embeddings)
            
            # Evaluate
            evaluator = ModelEvaluator()
            coherence = evaluator.evaluate(docs, model)
            
            # Get average coherence
            if isinstance(coherence, dict):
                score = coherence.get('c_v', coherence.get('u_mass', 0.0))
            else:
                score = coherence
            
            # Track results
            trial_results.append({
                'nr_topics': nr_topics,
                'coherence': coherence,
                'score': score
            })
            
            # Update best
            if score > best_score:
                best_score = score
                best_nr_topics = nr_topics
                if not quiet:
                    print(f"  ✓ New best! Score: {score:.4f}")
            elif not quiet:
                print(f"  Score: {score:.4f}")
            
            # Cleanup
            del model, topics, probs
            EnvironmentSetup.cleanup_memory()
            
        except Exception as e:
            if not quiet:
                print(f"  ✗ Error: {e}")
            trial_results.append({
                'nr_topics': nr_topics,
                'coherence': None,
                'score': float('-inf'),
                'error': str(e)
            })
    
    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    results_file = output_path / f"optimization_results_{db_name}_{collection_name}.json"
    with open(results_file, 'w') as f:
        json.dump({
            'best_nr_topics': best_nr_topics,
            'best_score': best_score,
            'search_range': [min_topics, max_topics],
            'step': step,
            'all_trials': trial_results,
            'optimization_method': 'grid_search'
        }, f, indent=2)
    
    if not quiet:
        print("\n" + "="*80)
        print("OPTIMIZATION COMPLETE")
        print("="*80)
        print(f"Best number of topics: {best_nr_topics}")
        print(f"Best coherence score: {best_score:.4f}")
        print(f"Results saved to: {results_file}")
        print("="*80 + "\n")
    
    return {
        'best_nr_topics': best_nr_topics,
        'best_score': best_score,
        'all_trials': trial_results,
        'results_file': results_file
    }


def main():
    parser = argparse.ArgumentParser(
        description="Build topic models from Milvus database collections",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Model entire paragraph collection (full subcorpus)
  python topic_model_milvus.py \\
    --db-name my_subcorpus \\
    --collection paragraphs \\
    --subcorpus-file subcorpus_results.pkl \\
    --s2orc-path /data/s2orc
  
  # Model query results (from query_subcorpus.py)
  python topic_model_milvus.py \\
    --db-name my_subcorpus \\
    --collection paragraphs \\
    --subcorpus-file query_results_quantum.pkl \\
    --s2orc-path /data/s2orc \\
    --output-dir ./models/quantum
  
  # Model sentences with limit
  python topic_model_milvus.py \\
    --db-name my_subcorpus \\
    --collection sentences \\
    --subcorpus-file query_results.pkl \\
    --s2orc-path /data/s2orc \\
    --limit 10000 \\
    --nr-topics 30
  
  # Model with RRF score filter (high-relevance query results only)
  python topic_model_milvus.py \\
    --db-name my_subcorpus \\
    --collection paragraphs \\
    --subcorpus-file query_results.pkl \\
    --s2orc-path /data/s2orc \\
    --filter "rrf_score > 0.3"
  
  # Large collection with forced document info export
  python topic_model_milvus.py \\
    --db-name my_subcorpus \\
    --collection paragraphs \\
    --subcorpus-file subcorpus_results.pkl \\
    --s2orc-path /data/s2orc \\
    --export-doc-info
  
  # Optimize topic number for query results
  python topic_model_milvus.py \\
    --db-name my_subcorpus \\
    --collection paragraphs \\
    --subcorpus-file query_results.pkl \\
    --s2orc-path /data/s2orc \\
    --optimize \\
    --min-topics 10 \\
    --max-topics 50 \\
    --n-trials 15
  
  # Optimize and build best model from query results
  python topic_model_milvus.py \\
    --db-name my_subcorpus \\
    --collection paragraphs \\
    --subcorpus-file query_results.pkl \\
    --s2orc-path /data/s2orc \\
    --optimize \\
    --build-best \\
    --min-topics 10 \\
    --max-topics 50
  
  # Grid search optimization on full subcorpus
  python topic_model_milvus.py \\
    --db-name my_subcorpus \\
    --collection paragraphs \\
    --subcorpus-file subcorpus_results.pkl \\
    --s2orc-path /data/s2orc \\
    --optimize \\
    --optimize-method grid \\
    --min-topics 10 \\
    --max-topics 50 \\
    --step 5

Note on document_info file:
  By default, the document_info CSV file is only created when processing
  ≤10,000 documents to avoid very large output files. Use --export-doc-info
  to force creation regardless of document count, or --no-doc-info to
  always skip it.
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
        help='Name of the collection (e.g., sentences, paragraphs)'
    )
    
    parser.add_argument(
        '--subcorpus-file',
        type=str,
        required=True,
        help='Path to subcorpus file: .json from query_subcorpus.py (default) or .pkl from query_milvus_rrf.py'
    )
    
    parser.add_argument(
        '--s2orc-path',
        type=str,
        required=True,
        help='Path to S2ORC gzipped JSONL files directory'
    )
    
    # Connection parameters
    parser.add_argument(
        '--milvus-host',
        type=str,
        default='localhost',
        help='Milvus server host (default: localhost)'
    )
    
    parser.add_argument(
        '--milvus-port',
        type=int,
        default=19530,
        help='Milvus server port (default: 19530)'
    )
    
    parser.add_argument(
        '--mongo-host',
        type=str,
        default='localhost',
        help='MongoDB server host (default: localhost)'
    )
    
    parser.add_argument(
        '--mongo-port',
        type=int,
        default=27017,
        help='MongoDB server port (default: 27017)'
    )
    
    # Model parameters
    parser.add_argument(
        '--embedding-model',
        type=str,
        default='sentence-transformers/all-MiniLM-L6-v2',
        help='Embedding model name (default: all-MiniLM-L6-v2). Note: Embeddings from Milvus will be used; this is only for model metadata'
    )
    
    parser.add_argument(
        '--nr-topics',
        type=int,
        default=None,
        help='Number of topics to generate (default: auto-detect)'
    )
    
    parser.add_argument(
        '--min-cluster-size',
        type=int,
        default=50,
        help='Minimum cluster size for HDBSCAN (default: 50)'
    )
    
    parser.add_argument(
        '--umap-neighbors',
        type=int,
        default=15,
        help='Number of neighbors for UMAP (default: 15)'
    )
    
    parser.add_argument(
        '--umap-components',
        type=int,
        default=10,
        help='Number of components for UMAP (default: 10)'
    )
    
    # Data filtering
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Maximum number of documents to process (default: no limit)'
    )
    
    parser.add_argument(
        '--filter',
        type=str,
        default=None,
        help='Milvus filter expression (e.g., "rrf_score > 0.5", "corpusid in [123, 456]")'
    )
    
    # Output control
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./topic_models',
        help='Output directory for models and visualizations (default: ./topic_models)'
    )
    
    parser.add_argument(
        '--export-doc-info',
        action='store_true',
        help='Force export of document_info CSV even for large datasets (>10k docs)'
    )
    
    parser.add_argument(
        '--no-doc-info',
        action='store_true',
        help='Never export document_info CSV regardless of dataset size'
    )
    
    # GPU
    parser.add_argument(
        '--no-gpu',
        action='store_true',
        help='Disable GPU acceleration'
    )
    
    # Verbosity
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress progress messages'
    )
    
    # Optimization mode
    parser.add_argument(
        '--optimize',
        action='store_true',
        help='Run topic number optimization instead of building a single model'
    )
    
    parser.add_argument(
        '--optimize-method',
        type=str,
        default='bayesian',
        choices=['bayesian', 'grid'],
        help='Optimization method: bayesian (default, more efficient) or grid (exhaustive)'
    )
    
    parser.add_argument(
        '--min-topics',
        type=int,
        default=5,
        help='Minimum number of topics for optimization (default: 5)'
    )
    
    parser.add_argument(
        '--max-topics',
        type=int,
        default=100,
        help='Maximum number of topics for optimization (default: 100)'
    )
    
    parser.add_argument(
        '--n-trials',
        type=int,
        default=20,
        help='Number of trials for Bayesian optimization (default: 20)'
    )
    
    parser.add_argument(
        '--step',
        type=int,
        default=5,
        help='Step size for grid search optimization (default: 5)'
    )
    
    parser.add_argument(
        '--build-best',
        action='store_true',
        help='After optimization, build a full model with the best number of topics'
    )
    
    args = parser.parse_args()
    
    # Setup environment
    if not args.quiet:
        env_info = EnvironmentSetup.setup_local_environment()
    
    # OPTIMIZATION MODE
    if args.optimize:
        if not args.quiet:
            print("="*80)
            print("TOPIC NUMBER OPTIMIZATION MODE")
            print("="*80)
            print()
        
        # Create base config for optimization
        base_config = ModelConfig(
            embedding_model_name=args.embedding_model,
            nr_topics=None,  # Will be set during optimization
            hdbscan_min_cluster_size=args.min_cluster_size,
            umap_n_neighbors=args.umap_neighbors,
            umap_n_components=args.umap_components,
            use_gpu=not args.no_gpu,
            output_dir=Path(args.output_dir)
        )
        
        try:
            # Run optimization
            if args.optimize_method == 'bayesian':
                opt_results = optimize_topic_number(
                    db_name=args.db_name,
                    collection_name=args.collection,
                    subcorpus_file=args.subcorpus_file,
                    s2orc_path=args.s2orc_path,
                    milvus_host=args.milvus_host,
                    milvus_port=args.milvus_port,
                    mongo_host=args.mongo_host,
                    mongo_port=args.mongo_port,
                    limit=args.limit,
                    filter_expr=args.filter,
                    output_dir=args.output_dir,
                    min_topics=args.min_topics,
                    max_topics=args.max_topics,
                    n_trials=args.n_trials,
                    base_config=base_config,
                    use_gpu=not args.no_gpu,
                    quiet=args.quiet
                )
            else:  # grid
                opt_results = optimize_topic_number_grid(
                    db_name=args.db_name,
                    collection_name=args.collection,
                    subcorpus_file=args.subcorpus_file,
                    s2orc_path=args.s2orc_path,
                    milvus_host=args.milvus_host,
                    milvus_port=args.milvus_port,
                    mongo_host=args.mongo_host,
                    mongo_port=args.mongo_port,
                    limit=args.limit,
                    filter_expr=args.filter,
                    output_dir=args.output_dir,
                    min_topics=args.min_topics,
                    max_topics=args.max_topics,
                    step=args.step,
                    base_config=base_config,
                    use_gpu=not args.no_gpu,
                    quiet=args.quiet
                )
            
            # Optionally build full model with best parameters
            if args.build_best:
                if not args.quiet:
                    print("\n" + "="*80)
                    print("BUILDING FULL MODEL WITH BEST PARAMETERS")
                    print("="*80)
                    print()
                
                best_config = ModelConfig(
                    embedding_model_name=args.embedding_model,
                    nr_topics=opt_results['best_nr_topics'],
                    hdbscan_min_cluster_size=args.min_cluster_size,
                    umap_n_neighbors=args.umap_neighbors,
                    umap_n_components=args.umap_components,
                    use_gpu=not args.no_gpu,
                    output_dir=Path(args.output_dir)
                )
                
                results = build_topic_model_from_milvus(
                    db_name=args.db_name,
                    collection_name=args.collection,
                    subcorpus_file=args.subcorpus_file,
                    config=best_config,
                    s2orc_path=args.s2orc_path,
                    milvus_host=args.milvus_host,
                    milvus_port=args.milvus_port,
                    mongo_host=args.mongo_host,
                    mongo_port=args.mongo_port,
                    limit=args.limit,
                    filter_expr=args.filter,
                    output_dir=args.output_dir,
                    export_doc_info=args.export_doc_info,
                    skip_doc_info=args.no_doc_info
                )
                
                if not args.quiet:
                    print("\n" + "="*80)
                    print("OPTIMIZATION + MODEL BUILD COMPLETE!")
                    print("="*80)
                    print(f"\nBest model statistics:")
                    print(f"  Topics: {opt_results['best_nr_topics']}")
                    print(f"  Optimization score: {opt_results['best_score']:.4f}")
                    print(f"  Documents processed: {len(results['docs'])}")
                    print(f"  Model coherence: {results['coherence']}")
            
            return 0
            
        except KeyboardInterrupt:
            print(f"\n✗ Interrupted by user", file=sys.stderr)
            return 130
        except Exception as e:
            print(f"\n✗ Error: {e}", file=sys.stderr)
            if not args.quiet:
                import traceback
                traceback.print_exc()
            return 1
    
    # REGULAR MODE - Build single model
    if not args.quiet:
        print("="*80)
        print("TOPIC MODELING FROM MILVUS COLLECTION")
        print("="*80)
        print()
    
    # Create config
    config = ModelConfig(
        embedding_model_name=args.embedding_model,
        nr_topics=args.nr_topics,
        hdbscan_min_cluster_size=args.min_cluster_size,
        umap_n_neighbors=args.umap_neighbors,
        umap_n_components=args.umap_components,
        use_gpu=not args.no_gpu,
        output_dir=Path(args.output_dir)
    )
    
    if not args.quiet:
        print(f"\\nConfiguration:")
        print(f"  Database: {args.db_name}")
        print(f"  Collection: {args.collection}")
        print(f"  Subcorpus file: {args.subcorpus_file}")
        print(f"  S2ORC path: {args.s2orc_path}")
        if args.limit:
            print(f"  Limit: {args.limit} documents")
        else:
            print(f"  Limit: No limit (process all documents)")
        if args.filter:
            print(f"  Filter: {args.filter}")
        print(f"  Topics: {args.nr_topics if args.nr_topics else 'auto-detect'}")
        print(f"  Min cluster size: {args.min_cluster_size}")
        print(f"  GPU: {not args.no_gpu}")
        print(f"  Output: {args.output_dir}")
        
        # Document info export logic
        if args.no_doc_info:
            print(f"  Document info: Disabled")
        elif args.export_doc_info:
            print(f"  Document info: Forced export (regardless of size)")
        else:
            print(f"  Document info: Auto (only if ≤10,000 docs)")
        print()
    
    try:
        # Build model
        results = build_topic_model_from_milvus(
            db_name=args.db_name,
            collection_name=args.collection,
            subcorpus_file=args.subcorpus_file,
            config=config,
            s2orc_path=args.s2orc_path,
            milvus_host=args.milvus_host,
            milvus_port=args.milvus_port,
            mongo_host=args.mongo_host,
            mongo_port=args.mongo_port,
            limit=args.limit,
            filter_expr=args.filter,
            output_dir=args.output_dir,
            export_doc_info=args.export_doc_info,
            skip_doc_info=args.no_doc_info
        )
        
        if not args.quiet:
            print("\\n" + "="*80)
            print("SUCCESS!")
            print("="*80)
            print(f"\\nModel statistics:")
            print(f"  Documents processed: {len(results['docs'])}")
            print(f"  Topics discovered: {len(results['model'].get_topic_info()) - 1}")  # -1 for outlier topic
            print(f"  Coherence scores: {results['coherence']}")
            print(f"\\nOutput files saved to: {args.output_dir}")
            print(f"  Model: topic_model_{results['model_suffix']}.safetensors")
            print(f"  Topic info: topic_info_{results['model_suffix']}.csv")
            
            # Document info status
            if 'doc_info_exported' in results:
                if results['doc_info_exported']:
                    print(f"  Document info: document_info_{results['model_suffix']}.csv")
                else:
                    print(f"  Document info: Skipped (>10,000 docs, use --export-doc-info to force)")
            
            print(f"  Topic distributions: topic_distributions_{results['model_suffix']}.csv")
            print(f"  Visualizations: *_{results['model_suffix']}.html")
        
        return 0
        
    except KeyboardInterrupt:
        print(f"\\n✗ Interrupted by user", file=sys.stderr)
        return 130
    except Exception as e:
        print(f"\\n✗ Error: {e}", file=sys.stderr)
        if not args.quiet:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
