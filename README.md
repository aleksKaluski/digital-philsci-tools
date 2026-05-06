# Digital Philosophical Tools Guide (Linux)

This repository contains the codebase designed for Wiktor Rorot's PhD project: _Scale-Free Communication? An investigation of the use of the concept of "communication" in biology and cognitive sciences._ 

The project features a text-processing pipeline based on **BERTopic**, integrated with **MongoDB** and **Milvus**. To run the analysis locally, we use Docker to create mock instances of these databases.

## 1. Create Initial Python Environment

First, ensure you are using **Python 3.11**. Check your version with:
`python3 --version`. 

Follow these steps to set up your virtual environment:

1. **Create the environment:** `python3 -m venv .venv`
2. **Activate it:** `source .venv/bin/activate`
3. **Verify Python:** `python --version`
4. **Upgrade core tools:** `pip install --upgrade pip wheel`
5. **Install specific setuptools:** `pip install "setuptools<82"`
6. **Install requirements:** `pip install -r requirements.txt`

### Sanity Check
Run these commands to ensure the environment is correctly configured:
* `python -c "import sys; print(sys.executable)"`
* `python -c "import pymongo; print('pymongo ok')"`
* `python -c "import pymilvus; print('pymilvus ok')"`

---

## 2. Working with Docker Containers
#### MongoDB and Milvus Setup

1. **Start the containers:** `docker-compose up -d`
2. **Check status:** `docker-compose ps`
3. **Verify MongoDB connectivity:**
   `python -c "from pymongo import MongoClient; client = MongoClient('localhost', 27017); client.server_info(); print('MongoDB connected!')"`
4. **Verify Milvus connectivity:**
   `python -c "from pymilvus import connections; connections.connect(host='localhost', port='19530'); print('Milvus connected!')"`

#### Mock Testing
Populate the database with a mock dataset of neopositivistic papers to verify functionality:

* **Test MongoDB:** `python mock_testing/mongo_simple_test.py`
* **Test Milvus:** `python mock_testing/milvus_simple_test.py`

> **Note:** To check if the ports are open on Ubuntu, you can use `nc` (netcat):
> * `nc -zv localhost 19530`
> * `nc -zv localhost 27017`

#### Database Cleanup
To drop the mock databases and start fresh:
* **Drop MongoDB:** `python -c "from pymongo import MongoClient; client = MongoClient('localhost', 27017); client.drop_database('papers_db'); print('Database dropped')"`
* **Drop Milvus:** `python -c "from pymilvus import connections, utility; connections.connect(host='localhost', port='19530'); utility.drop_collection('neopositivistic_papers'); print('Collection dropped')"`

---

## 3. Creating the BERT Environment

You will need **Conda** installed for this section (`conda --version`). 

1. **Create a directory for your environments:**
   `mkdir -p ~/conda_envs`
2. **Create the environment:**
   `conda env create -f environment_bertopic.yml --prefix ~/conda_envs/bertopic`
   *(This typically takes 5–15 minutes.)*

**Important:** To avoid conflicts between your `.venv` and your Conda environment, ensure you `deactivate` the standard virtual environment before activating Conda:

```bash
deactivate
conda activate ~/conda_envs/bertopic
# Verify the setup
python -c "from topic_modeling_analysis import *; print('✅ Success')"
```

---

## 4. Running the Analysis

If you’ve made it this far, you’ve done the hard part! Now we can run the analysis on a mock corpus.

### Prepare the Mock Data
Switch back to your original environment for database preparation:
1. `conda deactivate`
2. `source .venv/bin/activate`
3. **Initialize MongoDB:** `python mock_testing/create_mock_mongo_corpus.py`
4. **Initialize Milvus:** `python mock_testing/create_mock_milvus_corpus.py`

### Build the Subcorpus
Run the following command to initialize the Milvus subcorpus. Note the use of `\` for line continuations in Ubuntu/Bash:

```bash
python build_subcorpus_milvus.py \
  --subcorpus mock_testing/data/micro_subcorpus.pkl \
  --s2orc-path . \
  --db-name micro_subcorpus \
  --sentence-collection sentences \
  --paragraph-collection paragraphs \
  --model multi-qa-MiniLM-L6-cos-v1 \
  --no-gpu \
  --mongo-db-name mock_philsci \
  --mongo-collection-name papers
```

### Verification & Troubleshooting
To check if the papers were loaded correctly:
```bash
python -c "from pymilvus import connections, db, MilvusClient; connections.connect(alias='default', host='localhost', port='19530'); db.using_database('micro_subcorpus'); c=MilvusClient(uri='http://localhost:19530', token='root:Milvus', db_name='micro_subcorpus'); print('sentences', c.get_collection_stats('sentences')); print('paragraphs', c.get_collection_stats('paragraphs'))"
```

If you need to restart from scratch because a checkpoint failed, delete the checkpoint and drop the empty collections:
```bash
rm -f subcorpus_checkpoint.pkl
python -c "from pymilvus import connections, db, MilvusClient; connections.connect(alias='default', host='localhost', port='19530'); db.using_database('micro_subcorpus'); c=MilvusClient(uri='http://localhost:19530', token='root:Milvus', db_name='micro_subcorpus'); [c.drop_collection(name) for name in ['sentences','paragraphs'] if c.has_collection(name)]; print('done')"
```

### Final Query
Once the corpus is ready, use `query_subcorpus.py` to run your analysis:

```bash
python query_subcorpus.py \
  --db-name micro_subcorpus \
  --collection paragraphs \
  --queries queries.txt \
  --output micro_query_results.json \
  --limit 50 \
  --model multi-qa-MiniLM-L6-cos-v1 \
  --no-gpu
```

## Real Processing Workflow (in Linux)

#### Step 1: Identify Seed Papers
Manually curate or automatically identify seed papers that represent your research topic of interest.

**Input formats:**
- List of S2ORC corpus IDs
- List of paper titles/abstracts
- Bibliography file (BibTeX, RIS)

**Example seed papers file** (`seed_papers.txt`):
```
34474804
3066540
16056957
14559809
10509722
```
The script `s2_api_requests.py` offers functionalities to download the metadata and embeddings from the Semantic Scholar API for further processing. Alternatively, a local database from the `embeddings` dataset can be used.



#### Step 2: Identify Seed Papers
Use Reciprocal Rank Fusion (RRF) to identify papers semantically related to your seed papers.

**Script:** `query_milvus_rrf.py`

```pycon
python query_milvus_rrf.py \
  --output-size 100 \
  --output-prefix files/data/dp_mock \
  --top-k 100 \
  --rrf-k 60 \
  --embeddings-file files/operational_files/paper_embeddings.pkl \
  --log-dir logs
```

**Output:** Pickle file containing:
- `corpus_ids`: List of S2ORC corpus IDs in the subcorpus
- `scores`: RRF scores for each paper
- `embeddings`: Full-paper embeddings (if available)
- `metadata`: Additional information

**Key Parameters:**`
- `--output-size`: Number of results to output (default: 1000)
- `--top-k`: Number of papers to retrieve per query (default: 10000)
- `--rrf-k`: RRF constant (default: 60, lower = more emphasis on top ranks)
- `--queries`: File with queries (one per line)
- `--embeddings-file`: File with full paper embeddings for the queries, can downloaded from S2 API with `s2_api_requests.py`


#### Step 3: Create Milvus Database
Build searchable vector databases at sentence and paragraph level.

**Script:** `build_subcorpus_milvus.py`

```pycon
python build_subcorpus_milvus.py \
    --subcorpus files/data/dp_mock_20260506_172313.pkl \
    --s2orc-path /path/to/s2orc/corpus/2024-08-06/s2orc/ \
    --db-name dp_mock_subcorpus \
    --sentence-collection dp_sentences \
    --paragraph-collection dp_paragraphs \
    --paragraph-size 10 \
    --checkpoint-file files\data\dp_mock_checkpoint.pkl \
    --model multi-qa-MiniLM-L6-cos-v1 \
    --log-dir logs 

```

**Key Parameters:**
- `--subcorpus`: Pickle file from Step 2
- `--s2orc-path`: Path to S2ORC corpus directory
- `--db-name`: Name for Milvus database
- `--sentence-collection`: Collection name for sentences
- `--paragraph-collection`: Collection name for paragraphs
- `--paragraph-size`: Sentences per paragraph if no annotations (default: 10)
- `--model`: Sentence transformer model (default: multi-qa-MiniLM-L6-cos-v1)
- `--index-type`: Vector index type (IVF_PQ, IVF_FLAT, HNSW)
- `--checkpoint-file`: File for resumable processing
- `--no-language-filter`: Disable English-only filtering

**What it does:**
1. Loads subcorpus corpus IDs from pickle file
2. Retrieves papers from S2ORC using MongoDB indices and indexed gzip
3. Segments papers into sentences (using spaCy or custom sentencizer)
4. Groups sentences into paragraphs (using annotations or fixed-size chunks)
5. Generates embeddings using sentence-transformers
6. **Stores character indices** (not text) for each sentence/paragraph to save space
7. Inserts into Milvus collections with vector indices


**Database Schema:**

**Sentence Collection:**
```
- id (primary key, auto-generated)
- corpusid (int64)
- sentence_number (int64)
- sentence_indices (array[int64, 2])  # [start_char, end_char]
- vector (float vector, dim=384)
- rrf_score (float, optional)
```

**Paragraph Collection:**
```
- id (primary key, auto-generated)
- corpusid (int64)
- paragraph_number (int64)
- paragraph_indices (array[int64, 2])  # [start_char, end_char]
- sentence_start (int64)  # First sentence number
- sentence_end (int64)    # Last sentence number
- vector (float vector, dim=384)
- rrf_score (float, optional)
```


### Step 4: Query the Subcorpus
Search the subcorpus using natural language queries.

**Script:** `query_subcorpus.py`

```pycon
python query_subcorpus.py \
    --db-name dp_mock_subcorpus \
    --collection dp_sentences \
    --queries files/operational_files/research_queries.txt \
    --output files/operational_files/query_results.json \
    --model multi-qa-MiniLM-L6-cos-v1 \
    --limit 100 \
    --metric-type COSINE \
    --use-rrf
    --log-dir logs 
```

**Queries file format** (`research_queries.txt`):
```
# Research questions (lines starting with # are comments)
How does compositionality emerge in neural language models?
What computational mechanisms support semantic composition?
Evidence for compositional processing in the human brain

# Hypothesis statements
Neural networks learn compositional representations through hierarchical processing.
```

**Output format** (JSON):
```json
[
  {
    "query_idx": 0,
    "query": "How does compositionality emerge in neural language models?",
    "rank": 1,
    "distance": 0.8543,
    "corpus_id": 12345678,
    "collection": "sentences",
    "sentence_number": 42,
    "sentence_indices": [1523, 1687]
  },
  ...
]
```

## Citing and Credits
```bibtex
@phdthesis{rorot_scalefree_2025,
  title = {Scale-{{Free Communication}}? {{An}} Investigation of the Use of the Concept "Communication" in Biology and Cognitive Sciences},
  author = {Rorot, Wiktor},
  year = 2025,
  month = dec,
  address = {Warsaw},
  langid = {english},
  school = {University of Warsaw}
}
```

The development of these tools was funded by the National Science Center (Poland) as part of Preludium grant 
"Investigation of the use of the concept “communication” in biology and cognitive sciences" (2022/45/N/HS1/02434), 
awarded to Wiktor Rorot (supervisor: Marcin Miłkowski) (project begun in February 2023, planned conclusion: July 2026).

Published in 2025 by Wiktor Rorot, small improvements introduced by Aleksander Kałuski. 

Code licensed under GNU GPL v3.

