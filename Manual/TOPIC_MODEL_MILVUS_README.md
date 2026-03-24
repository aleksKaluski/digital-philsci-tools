# Topic Modeling from Milvus Collections - CLI Tool

This CLI tool builds topic models directly from Milvus database collections using pre-computed embeddings, avoiding the need to recompute embeddings and providing 2-3x speedup for topic modeling workflows.

## Requirements

- pymilvus: `pip install pymilvus`
- pymongo: `pip install pymongo`
- scikit-optimize (optional, for Bayesian optimization): `pip install scikit-optimize`
- Other dependencies from `environment_bertopic.yml`

**Note:** If scikit-optimize is not installed and you use `--optimize`, the script will automatically fall back to grid search.

## Usage

### Basic Usage

Model an entire collection:

```bash
python topic_model_milvus.py \
    --db-name my_subcorpus \
    --collection paragraphs \
    --s2orc-path /path/to/s2orc
```

### With Custom Parameters

```bash
python topic_model_milvus.py \
    --db-name my_subcorpus \
    --collection sentences \
    --s2orc-path /path/to/s2orc \
    --nr-topics 20 \
    --min-cluster-size 30 \
    --limit 5000 \
    --output-dir ./my_models
```

### With Filtering

Filter documents using Milvus filter expressions:

```bash
python topic_model_milvus.py \
    --db-name my_subcorpus \
    --collection paragraphs \
    --s2orc-path /path/to/s2orc \
    --filter "rrf_score > 0.5"
```

### Remote Server

Connect to Milvus and MongoDB on remote servers:

```bash
python topic_model_milvus.py \
    --db-name my_subcorpus \
    --collection paragraphs \
    --s2orc-path /path/to/s2orc \
    --milvus-host 192.168.1.100 \
    --milvus-port 19530 \
    --mongo-host 192.168.1.100 \
    --mongo-port 27017
```

## Command-Line Options

### Required Arguments

- `--db-name`: Name of the Milvus database
- `--collection`: Name of the collection (`sentences`, `paragraphs`, or custom)
- `--s2orc-path`: Path to S2ORC gzipped JSONL files directory

### Connection Parameters

- `--milvus-host`: Milvus server host (default: `localhost`)
- `--milvus-port`: Milvus server port (default: `19530`)
- `--mongo-host`: MongoDB server host (default: `localhost`)
- `--mongo-port`: MongoDB server port (default: `27017`)

### Model Parameters

- `--embedding-model`: Embedding model name (default: `sentence-transformers/all-MiniLM-L6-v2`)
  - Note: This is only used for metadata; actual embeddings are loaded from Milvus
- `--nr-topics`: Number of topics to generate (default: auto-detect)
- `--min-cluster-size`: Minimum cluster size for HDBSCAN (default: `50`)
- `--umap-neighbors`: Number of neighbors for UMAP (default: `15`)
- `--umap-components`: Number of components for UMAP (default: `10`)

### Optimization Parameters

- `--optimize`: Enable topic number optimization mode
- `--optimize-method`: Optimization method: `bayesian` (default) or `grid`
- `--min-topics`: Minimum number of topics to try (default: `5`)
- `--max-topics`: Maximum number of topics to try (default: `100`)
- `--n-trials`: Number of trials for Bayesian optimization (default: `20`)
- `--step`: Step size for grid search (default: `5`)
- `--build-best`: After optimization, build full model with best parameters

### Data Filtering

- `--limit`: Maximum number of documents to process (default: no limit)
  - Example: `--limit 5000`
- `--filter`: Milvus filter expression (default: none)
  - Example: `--filter "rrf_score > 0.5"`
  - Example: `--filter "corpusid in [123, 456, 789]"`

### Output Control

- `--output-dir`: Output directory for models and visualizations (default: `./topic_models`)
- `--export-doc-info`: Force export of `document_info.csv` even for large datasets (>10k docs)
- `--no-doc-info`: Never export `document_info.csv` regardless of dataset size

### Other Options

- `--no-gpu`: Disable GPU acceleration
- `--quiet`: Suppress progress messages

## Document Info Export Behavior

The `document_info.csv` file contains detailed per-document information including:
- Document text
- Assigned topic
- Topic probability
- Topic name
- All metadata fields

**Size Management:**
- By default, `document_info.csv` is only created when processing ≤10,000 documents
- This file can become very large (several GB) with extensive collections
- Use `--export-doc-info` to force export for large datasets
- Use `--no-doc-info` to always skip this file
- The `topic_distributions.csv` file is always created regardless of size

## Examples

### Example 1: Quick Analysis

Model up to 5000 paragraphs from a collection:

```bash
python topic_model_milvus.py \
    --db-name quantum_mechanics \
    --collection paragraphs \
    --s2orc-path /data/s2orc \
    --limit 5000
```

### Example 2: High-Quality Subset

Model only high-scoring results with custom topics:

```bash
python topic_model_milvus.py \
    --db-name quantum_mechanics \
    --collection paragraphs \
    --s2orc-path /data/s2orc \
    --filter "rrf_score > 0.7" \
    --nr-topics 30 \
    --min-cluster-size 20
```

### Example 3: Large Collection

Model entire collection without document info export:

```bash
python topic_model_milvus.py \
    --db-name quantum_mechanics \
    --collection sentences \
    --s2orc-path /data/s2orc \
    --no-doc-info
```

### Example 4: Server Deployment

Run on a remote server with Jupyter unavailable:

```bash
python topic_model_milvus.py \
    --db-name my_subcorpus \
    --collection paragraphs \
    --s2orc-path /mnt/data/s2orc \
    --milvus-host localhost \
    --mongo-host localhost \
    --output-dir /mnt/outputs/topic_models \
    --limit 20000 \
    --export-doc-info  # Force export even though >10k
```

### Example 5: Find Optimal Topic Number (Bayesian)

Use Bayesian optimization to find the best number of topics:

```bash
python topic_model_milvus.py \
    --db-name my_subcorpus \
    --collection paragraphs \
    --s2orc-path /data/s2orc \
    --optimize \
    --min-topics 10 \
    --max-topics 50 \
    --n-trials 20
```

This will:
- Intelligently sample 20 different topic numbers between 10 and 50
- Use Bayesian optimization to focus on promising regions
- Save optimization results to `optimization_results_*.json`
- Much faster than grid search

### Example 6: Optimize and Build Best Model

Find optimal topics then build the full model:

```bash
python topic_model_milvus.py \
    --db-name my_subcorpus \
    --collection paragraphs \
    --s2orc-path /data/s2orc \
    --optimize \
    --build-best \
    --min-topics 15 \
    --max-topics 40 \
    --n-trials 15
```

### Example 7: Grid Search Optimization

For exhaustive search across all values:

```bash
python topic_model_milvus.py \
    --db-name my_subcorpus \
    --collection paragraphs \
    --s2orc-path /data/s2orc \
    --optimize \
    --optimize-method grid \
    --min-topics 10 \
    --max-topics 50 \
    --step 5
```

This tests every value: 10, 15, 20, 25, 30, 35, 40, 45, 50

## Output Files

The script generates the following files in the output directory:

### Always Created

- `topic_model_{suffix}.safetensors`: Trained BERTopic model
- `topic_info_{suffix}.csv`: Topic information with top words
- `topic_distributions_{suffix}.csv`: Per-document topic probability distributions
- `documents_visualization_{suffix}.html`: Interactive 2D/3D document visualization
- `hierarchy_visualization_{suffix}.html`: Hierarchical topic structure
- `barchart_visualization_{suffix}.html`: Top topics bar chart
- `heatmap_visualization_{suffix}.html`: Topic similarity heatmap

### Conditionally Created

- `document_info_{suffix}.csv`: Detailed per-document information
  - Created by default when ≤10,000 documents
  - Skipped by default when >10,000 documents
  - Use `--export-doc-info` to force creation
  - Use `--no-doc-info` to always skip

## Integration with Other Tools

This CLI tool works seamlessly with:
- `build_subcorpus_milvus.py`: Use collections created by this tool
- `query_milvus_rrf.py`: Model the query results stored in collections
- `topic_modeling_notebook.ipynb`: Same functionality, interactive interface

## Topic Number Optimization

Finding the optimal number of topics is crucial for meaningful topic modeling. This tool provides two optimization methods:

### Bayesian Optimization (Recommended)

**Advantages:**
- **Efficient**: Intelligently samples the parameter space, focusing on promising regions
- **Fast**: Typically needs only 15-25 trials to find good solutions
- **Smart**: Uses Gaussian Process to model the objective function
- **Adaptive**: Learns from previous evaluations

**When to use:**
- Large search spaces (e.g., 5-100 topics)
- Limited computational budget
- When you want good results quickly

**Requirements:**
```bash
pip install scikit-optimize
```

**Example:**
```bash
python topic_model_milvus.py \
    --db-name my_subcorpus \
    --collection paragraphs \
    --s2orc-path /data/s2orc \
    --optimize \
    --min-topics 10 \
    --max-topics 60 \
    --n-trials 20
```

### Grid Search

**Advantages:**
- **Exhaustive**: Tests every value in the range
- **Simple**: Easy to understand and interpret
- **Deterministic**: Always produces the same results

**Disadvantages:**
- **Slow**: Tests every value, can be time-consuming
- **Inefficient**: Wastes computation on clearly bad values

**When to use:**
- Small search spaces (e.g., 10-30 topics with step=5)
- When you need to test every value
- When computational resources are not a concern

**Example:**
```bash
python topic_model_milvus.py \
    --db-name my_subcorpus \
    --collection paragraphs \
    --s2orc-path /data/s2orc \
    --optimize \
    --optimize-method grid \
    --min-topics 10 \
    --max-topics 40 \
    --step 5
```

### Optimization Output

Both methods produce `optimization_results_*.json` containing:
- Best number of topics found
- Best coherence score
- All trial results for analysis
- Optimization method used

You can analyze this file to:
- Verify the optimization found a good solution
- Understand the coherence landscape
- Make informed decisions about topic numbers

### Tips for Optimization

1. **Start with a reasonable range**: 
   - Sentences: 10-50 topics typical
   - Paragraphs: 15-60 topics typical
   - Very large corpora: up to 100+ topics

2. **Use `--limit` for faster iteration**:
   - Sample 5,000-10,000 documents for optimization
   - Build final model on full dataset with `--build-best`

3. **Consider your corpus size**:
   - Too many topics for small corpora → sparse, unstable topics
   - Too few topics for large corpora → mixed, incoherent topics

4. **Use Bayesian optimization by default**:
   - Grid search only if you need exhaustive search
   - Bayesian is 2-5x faster for similar results

## Performance Notes

- Using pre-computed embeddings from Milvus provides 2-3x speedup vs. recomputing
- GPU acceleration significantly speeds up UMAP dimensionality reduction
- Bayesian optimization is 2-5x faster than grid search for finding optimal topics
- For large collections (>50k documents), consider:
  - Using `--limit` to sample documents for optimization
  - Using `--filter` to select relevant subsets
  - Using `--no-doc-info` to avoid large output files
  - Running with `--quiet` for cleaner logs
  - Optimizing on a sample, then building final model with `--build-best`

## Troubleshooting

### Connection Errors

If you get connection errors to Milvus or MongoDB:
- Verify the servers are running
- Check host and port parameters
- Ensure network connectivity
- Check firewall rules

### Memory Errors

If you run out of memory:
- Use `--limit` to reduce dataset size
- Use `--no-doc-info` to skip large output file
- Increase system memory
- Use a machine with more RAM

### Empty Results

If no documents are loaded:
- Check that the collection exists in Milvus
- Verify the database name is correct
- Check filter expression syntax if using `--filter`
- Ensure MongoDB has corresponding paper metadata

## See Also

- `topic_modeling_notebook.ipynb`: Interactive Jupyter notebook interface
- `topic_modeling_analysis.py`: Core topic modeling library
- `SUBCORPUS_WORKFLOW.md`: Complete workflow documentation
- `QUERY_SUBCORPUS_README.md`: Query and filtering documentation
