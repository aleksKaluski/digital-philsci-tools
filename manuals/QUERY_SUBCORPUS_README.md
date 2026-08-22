# Query Subcorpus - Usage Guide

## Overview

`query_subcorpus.py` is a modular script for querying sentence- and paragraph-level Milvus databases created by `build_subcorpus_milvus.py`.

## Key Features

- **Flexible input**: Read queries from text file (one per line)
- **Configurable**: Specify database and collection names as parameters
- **Multiple output formats**: JSON, CSV, or Parquet
- **Reciprocal Rank Fusion (RRF)**: Aggregate results across multiple queries for improved ranking
- **Extensible**: Clean class-based architecture for easy customization
- **Batch processing**: Efficient encoding and querying
- **Rich results**: Formatted output with rankings and distance scores

## Quick Start

### 1. Prepare Your Queries

Create a text file with one query per line:

```txt
# queries.txt
The meaning of an expression is a function of its parts
How does compositionality apply to natural language?
Neural network approaches to compositional semantics
```

Lines starting with `#` are treated as comments and ignored.

### 2. Run the Query

```bash
# Basic usage - query sentence collection
python query_subcorpus.py \
    --db-name my_subcorpus \
    --collection sentences \
    --queries research_queries.txt \
    --output results.json

# Query paragraph collection
python query_subcorpus.py \
    --db-name my_subcorpus \
    --collection paragraphs \
    --queries research_queries.txt \
    --output results.json
```

## Command-Line Arguments

### Required Arguments

- `--db-name`: Name of the Milvus database to query
- `--collection`: Name of the collection (e.g., `sentences`, `paragraphs`)
- `--queries`: Path to queries text file (one query per line)
- `--output`: Path to output file (`.json`, `.csv`, or `.parquet`)

### Optional Arguments

- `--model`: Sentence transformer model (default: `multi-qa-MiniLM-L6-cos-v1`)
- `--limit`: Maximum results per query (default: 1000)
- `--milvus-host`: Milvus server host (default: `localhost`)
- `--milvus-port`: Milvus server port (default: 19530)
- `--metric-type`: Distance metric (`COSINE`, `L2`, `IP`) (default: `COSINE`)
- `--no-gpu`: Disable GPU for encoding
- `--output-fields`: Custom fields to retrieve (default: `corpus_id`, `sentence_idx`, `text`)
- `--no-summary`: Skip printing results summary
- `--use-rrf`: Use Reciprocal Rank Fusion to aggregate results across queries
- `--rrf-k`: RRF constant k (default: 60)
- `--rrf-output-size`: Number of top results to return when using RRF (default: 1000)

## Examples

### Example 1: Basic Query

```bash
python query_subcorpus.py \
    --db-name compositionality_subcorpus \
    --collection sentences \
    --queries sample_queries.txt \
    --output results.json
```

### Example 2: Large-Scale Query with More Results

```bash
python query_subcorpus.py \
    --db-name my_subcorpus \
    --collection paragraphs \
    --queries research_queries.txt \
    --limit 5000 \
    --output results.csv
```

### Example 3: Custom Model and Settings

```bash
python query_subcorpus.py \
    --db-name my_subcorpus \
    --collection sentences \
    --queries research_queries.txt \
    --model sentence-transformers/all-mpnet-base-v2 \
    --metric-type IP \
    --output results.parquet
```

### Example 4: Query with Custom Output Fields

```bash
python query_subcorpus.py \
    --db-name my_subcorpus \
    --collection paragraphs \
    --queries research_queries.txt \
    --output-fields corpus_id paragraph_idx start_sent end_sent text \
    --output results.json
```

### Example 5: CPU-Only Mode

```bash
python query_subcorpus.py \
    --db-name my_subcorpus \
    --collection sentences \
    --queries research_queries.txt \
    --no-gpu \
    --output results.json
```

### Example 6: Reciprocal Rank Fusion (RRF) Mode

Use RRF to aggregate results across multiple queries, finding documents that are relevant across all queries:

```bash
python query_subcorpus.py \
    --db-name my_subcorpus \
    --collection sentences \
    --queries research_queries.txt \
    --use-rrf \
    --rrf-output-size 1000 \
    --rrf-k 60 \
    --output results_rrf.json
```

**What is RRF?**

Reciprocal Rank Fusion (RRF) is a state-of-the-art rank aggregation method that combines rankings from multiple queries. Instead of getting separate results for each query, RRF produces a single unified ranking of documents that are relevant across all queries.

**RRF Formula:** `RRF_score(d) = Σ(1 / (k + rank_i(d)))`

Where:
- `d` is a document
- `k` is the RRF constant (default: 60, lower values emphasize top-ranked results more)
- `rank_i(d)` is the rank of document `d` in query `i`

**When to use RRF:**
- You have multiple related queries exploring different aspects of a topic
- You want to find documents relevant across all queries (consensus ranking)
- You need a single aggregated result set rather than per-query results

## Output Format

### Standard Mode Output

Results are saved as a table with the following columns:

- `query_idx`: Index of the query (0-based)
- `query`: Original query text
- `rank`: Rank of this result for the query (1-based)
- `distance`: Similarity distance (higher = more similar for COSINE)
- `corpus_id`: S2ORC corpus ID of the paper
- `collection`: Name of the queried collection
- Additional fields from the collection (e.g., `sentence_idx`, `text`, `paragraph_idx`)

### JSON Output Example (Standard Mode)

```json
[
  {
    "query_idx": 0,
    "query": "The meaning of an expression is a function of its parts",
    "rank": 1,
    "distance": 0.8543,
    "corpus_id": 12345678,
    "collection": "sentences",
    "sentence_idx": 42,
    "text": "Compositionality states that the meaning of complex expressions..."
  },
  ...
]
```

### RRF Mode Output

When using `--use-rrf`, the output format changes to show aggregated rankings:

- `rank`: Overall rank in the aggregated results (1-based)
- `rrf_score`: Reciprocal Rank Fusion score (higher = more relevant across queries)
- `corpusid`: S2ORC corpus ID of the paper
- Additional fields from the collection (e.g., `sentence_number`, `sentence_indices`)

Note: Query-specific information (query_idx, query text) is not included in RRF output since results are aggregated.

### JSON Output Example (RRF Mode)

```json
[
  {
    "rank": 1,
    "rrf_score": 0.0342,
    "corpusid": 12345678,
    "sentence_number": 42,
    "sentence_indices": [1523, 1687]
  },
  {
    "rank": 2,
    "rrf_score": 0.0298,
    "corpusid": 87654321,
    "sentence_number": 15,
    "sentence_indices": [892, 1056]
  },
  ...
]
```

## Integration with Build Pipeline

This script is designed to work seamlessly with `build_subcorpus_milvus.py`:

```bash
# Step 1: Build subcorpus database
python build_subcorpus_milvus.py \
    --subcorpus subcorpus_20251105.pkl \
    --s2orc-path /path/to/s2orc/ \
    --db-name my_subcorpus \
    --sentence-collection sentences \
    --paragraph-collection paragraphs

# Step 2: Query the database
python query_subcorpus.py \
    --db-name my_subcorpus \
    --collection sentences \
    --queries research_queries.txt \
    --output results.json
```

## Extending the Script

The script is designed for easy extension:

### Custom Result Processing

```python
from query_subcorpus import SubcorpusQueryClient, load_queries_from_file

# Initialize client
client = SubcorpusQueryClient(db_name="my_subcorpus")
client.connect()
client.load_encoder()

# Load and encode queries
queries = load_queries_from_file("research_queries.txt")
embeddings = client.encode_queries(queries)

# Query collection
results = client.query_collection("sentences", embeddings)

# Custom processing
for query_idx, query_results in enumerate(results):
    print(f"Query {query_idx}: {queries[query_idx]}")
    for result in query_results[:5]:  # Top 5
        print(f"  - {result['entity']['text'][:100]}...")
```

### Custom Filtering

```python
# Override format_results to add filtering
def custom_format_results(queries, results, collection_name):
    df = client.format_results(queries, results, collection_name)
    
    # Filter by distance threshold
    df = df[df['distance'] > 0.7]
    
    # Group by corpus_id
    df = df.groupby('corpus_id').first().reset_index()
    
    return df
```

## Tips for Best Results

1. **Query Design**: Use complete, well-formed sentences rather than keywords
2. **Model Selection**: Ensure the model matches the one used to build the database
3. **Result Limit**: Start with default (1000) and adjust based on needs
4. **Output Format**: Use Parquet for large results (better compression and performance)
5. **Batch Processing**: For many queries, consider splitting into multiple files

## Troubleshooting

### Connection Issues

```bash
# Check if Milvus is running
curl http://localhost:19530/health

# Verify database exists
python -c "from pymilvus import connections, db; connections.connect(); print(db.list_database())"
```

### Model Issues

```bash
# Pre-download model
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')"
```

### Memory Issues

```bash
# Use CPU mode to save GPU memory
python query_subcorpus.py --no-gpu ...

# Reduce batch size (edit BATCH_SIZE in script)
# Process queries in smaller batches
```

## Performance Notes

- **GPU vs CPU**: GPU encoding is ~10-50x faster for large query batches
- **Collection Loading**: First query loads collection into memory (may take a few seconds)
- **Result Limit**: Higher limits increase query time linearly
- **Output Format**: Parquet is faster for writing large results than JSON

## Related Scripts

- `build_subcorpus_milvus.py`: Build the subcorpus databases
- `query_milvus_rrf.py`: Query full corpus with RRF ranking
- `visualize_subcorpus.py`: Visualize query results
