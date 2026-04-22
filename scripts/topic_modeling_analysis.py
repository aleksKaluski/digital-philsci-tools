"""
Topic Modeling Analysis Module

A modular, extensible framework for BERTopic-based topic modeling analysis of scientific literature.
Supports both legacy formats (nested paragraph lists) and new formats (flat text with context).

Key Features:
- Dual format support (legacy JSON and retrieve_query_texts.py output)
- Multiple text modes: analyze query results alone or with full context
- Automatic detection of query level (sentence vs paragraph)
- **Milvus integration: Reuse embeddings from Milvus databases**
- **Model entire Milvus collections without recomputing embeddings**
- Standardized model naming based on data characteristics
- Modular architecture with separate classes for each concern
- GPU acceleration support (CUDA/MPS for embeddings, cuML for UMAP/HDBSCAN)
- Checkpoint/resume functionality
- Comprehensive coherence evaluation
- Interactive visualizations
- Type hints and documentation throughout

Text Modes:
- "result": Analyze only the matched query text
- "result-with-context": Analyze text with context_before and context_after
- "milvus": Load from Milvus database collection

Milvus Integration:
    The module can now load pre-computed embeddings from Milvus databases
    (e.g., created by build_subcorpus_milvus.py), avoiding the need to
    recompute embeddings and significantly speeding up topic modeling.
    
    Two main use cases:
    
    1. Reuse embeddings from existing query results:
       # Load data from JSON, but use Milvus embeddings if available
       loader = DataLoader("results.json")
       docs, metadata = loader.load()
       
       # Load embeddings from Milvus instead of computing
       milvus_loader = MilvusEmbeddingLoader(
           db_name='my_subcorpus',
           collection_name='paragraphs'
       )
       embeddings, _ = milvus_loader.load_embeddings()
       
       # Build model with pre-computed embeddings
       modeler = TopicModeler(config)
       model, topics, probs = modeler.fit(docs, embeddings)
    
    2. Model entire Milvus collections:
       # Build topic model directly from Milvus collection
       # Requires subcorpus pickle file from query_milvus_rrf.py
       results = build_topic_model_from_milvus(
           db_name='my_subcorpus',
           collection_name='paragraphs',
           subcorpus_file='subcorpus_results.pkl',  # From query_milvus_rrf.py
           config=config,
           s2orc_path='/path/to/s2orc',
           limit=1000  # Optional: limit number of documents
       )

Usage:
    # See accompanying Jupyter notebook: topic_modeling_notebook.ipynb
    
    from topic_modeling_analysis import (
        DataLoader, 
        TopicModeler, 
        ModelConfig,
        build_all_text_mode_models
    )
    
    # Load data (specify text mode)
    loader = DataLoader("results.json", text_mode="result")
    docs, metadata = loader.load()
    
    # Build model with configuration
    config = ModelConfig(
        query_level=loader.query_level,
        text_mode=loader.text_mode,
        use_gpu=True
    )
    modeler = TopicModeler(config)
    model, topics, probs = modeler.fit(docs)
    
    # Or build models for all text modes
    results = build_all_text_mode_models("results.json", config)
"""

import json
import pickle
import os
import gc
import time
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union, Any
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import plotly.io as pio

from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired, PartOfSpeech, MaximalMarginalRelevance
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP
from hdbscan import HDBSCAN
import gensim.corpora as corpora
from gensim.models.coherencemodel import CoherenceModel

from scripts.model_adapter import UnifiedEmbedder

# Milvus support (optional)
try:
    from pymilvus import connections, MilvusClient, db
    from pymongo import MongoClient
    MILVUS_AVAILABLE = True
except ImportError:
    MILVUS_AVAILABLE = False
    print("Milvus/MongoDB not available. Install with: pip install pymilvus pymongo")


# Configuration
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Plotly template setup
plt.rcParams['text.usetex'] = True
pio.renderers.default = "png+browser"

fig = go.Figure(layout={
    'title': 'Figure Title',
    'font': {'family': 'Noto Sans'}
})
fig.update_layout(
    font_family="Noto Sans",
    font_color="black",
    title_font_family="Noto Sans",
    title_font_color="black",
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
)
templated_fig = pio.to_templated(fig)
pio.templates['sans_figure'] = templated_fig.layout.template
pio.templates.default = 'sans_figure'


@dataclass
class ModelConfig:
    """Configuration for topic modeling."""
    
    # Model parameters
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    nr_topics: Optional[int] = None
    calculate_probabilities: bool = False
    
    # Data characteristics (set automatically by DataLoader)
    query_level: Optional[str] = None  # "sentence" or "paragraph"
    text_mode: str = "result"  # "result" or "result-with-context"
    
    # UMAP parameters
    umap_n_neighbors: int = 15
    umap_n_components: int = 10
    umap_min_dist: float = 0.0
    umap_metric: str = 'cosine'
    umap_random_state: int = 42
    
    # HDBSCAN parameters
    hdbscan_min_cluster_size: int = 50
    hdbscan_metric: str = 'euclidean'
    hdbscan_cluster_selection_method: str = 'eom'
    
    # Vectorizer parameters
    vectorizer_stop_words: str = "english"
    vectorizer_min_df: int = 2
    vectorizer_ngram_range: Tuple[int, int] = (1, 2)
    vectorizer_token_pattern: str = r'(?u)\b\w\w\w+\b'  # 3+ character tokens
    
    # Outlier reduction
    outlier_strategy: str = "distributions"
    outlier_threshold: float = 0.1
    
    # GPU settings
    use_gpu: bool = True
    
    # Coherence metrics
    coherence_metrics: List[str] = field(default_factory=lambda: ["c_v", "u_mass"])
    
    # Paths
    output_dir: Path = Path("./topic_models")
    checkpoint_interval: int = 100
    
    def get_model_suffix(self) -> str:
        """
        Generate standardized model name suffix based on configuration.
        
        Returns:
            String suffix like "sentence_result_20topics" or "paragraph_context_auto"
        """
        parts = []
        
        # Add query level
        if self.query_level:
            parts.append(self.query_level)
        
        # Add text mode (shortened)
        if self.text_mode == "result":
            parts.append("result")
        elif self.text_mode == "result-with-context":
            parts.append("context")
        
        # Add number of topics
        if self.nr_topics:
            parts.append(f"{self.nr_topics}topics")
        else:
            parts.append("auto")
        
        return "_".join(parts)


class EnvironmentSetup:
    """Setup and manage computational environment."""
    
    @staticmethod
    def setup_local_environment() -> Dict[str, Any]:
        """Setup local working environment for remote filesystems and GPU."""
        import torch
        
        env_info = {
            "working_dir": None,
            "gpu_available": False,
            "gpu_name": None,
            "gpu_memory": None
        }
        
        # Working directory setup
        try:
            cwd = os.getcwd()
            env_info["working_dir"] = cwd
            print(f"Working directory: {cwd}")
        except (FileNotFoundError, OSError):
            local_dir = "/tmp/bertopic_processing"
            os.makedirs(local_dir, exist_ok=True)
            os.chdir(local_dir)
            env_info["working_dir"] = local_dir
            print(f"Changed to local working directory: {local_dir}")
        
        # GPU setup (check both CUDA for NVIDIA and MPS for Apple Silicon)
        if torch.cuda.is_available():
            env_info["gpu_available"] = True
            env_info["gpu_name"] = torch.cuda.get_device_name(0)
            env_info["gpu_memory"] = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"GPU available: {env_info['gpu_name']} (CUDA)")
            print(f"GPU memory: {env_info['gpu_memory']:.1f} GB")
            os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        elif torch.backends.mps.is_available():
            env_info["gpu_available"] = True
            env_info["gpu_name"] = "Apple Silicon (MPS)"
            # MPS doesn't expose memory info the same way
            env_info["gpu_memory"] = None
            print(f"GPU available: {env_info['gpu_name']}")
        else:
            print("No GPU available, using CPU")
        
        # Set joblib environment
        os.environ['JOBLIB_TEMP_FOLDER'] = '/tmp'
        os.environ['LOKY_MAX_CPU_COUNT'] = '4'
        
        return env_info
    
    @staticmethod
    def cleanup_memory():
        """Force cleanup of GPU and system memory."""
        import torch
        
        # Clear Python garbage
        gc.collect()
        
        # Clear GPU memory if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        elif torch.backends.mps.is_available():
            # MPS cache clearing (PyTorch 2.0+)
            try:
                torch.mps.empty_cache()
            except AttributeError:
                pass  # Older PyTorch versions don't have mps.empty_cache()
        
        # Print memory status
        import psutil
        memory = psutil.virtual_memory()
        print(f"Memory usage after cleanup: {memory.percent}%")
        
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory
            gpu_allocated = torch.cuda.memory_allocated(0)
            print(f"GPU memory: {gpu_allocated / gpu_memory * 100:.1f}%")


class DataLoader:
    """Load and prepare data from various formats."""
    
    def __init__(self, data_path: Union[str, Path], text_mode: str = "result"):
        """
        Initialize data loader.
        
        Args:
            data_path: Path to input JSON file
            text_mode: How to extract text from retrieve_query_texts format:
                - "result": Use only the matched text (default)
                - "result-with-context": Combine text with context_before and context_after
        """
        self.data_path = Path(data_path)
        self.text_mode = text_mode
        self.format_type: Optional[str] = None
        self.query_level: Optional[str] = None  # "sentence" or "paragraph"
        
    def load(self) -> Tuple[List[str], pd.DataFrame]:
        """
        Load data and return documents with metadata.
        
        Returns:
            Tuple of (documents, metadata_df)
            - documents: List of text strings
            - metadata_df: DataFrame with metadata for each document
        """
        print(f"Loading data from: {self.data_path}")
        print(f"Text mode: {self.text_mode}")
        
        with open(self.data_path, 'r') as f:
            data = json.load(f)
        
        # Detect format and query level
        self.format_type = self._detect_format(data)
        print(f"Detected format: {self.format_type}")
        
        if self.query_level:
            print(f"Query level: {self.query_level}")
        
        if self.format_type == "legacy":
            return self._load_legacy_format(data)
        elif self.format_type == "retrieve_query_texts":
            return self._load_retrieve_format(data)
        else:
            raise ValueError(f"Unknown data format: {self.format_type}")
    
    def _detect_format(self, data: List[Dict]) -> str:
        """
        Detect data format from structure and query level.
        
        Args:
            data: Loaded JSON data
            
        Returns:
            Format type string
        """
        if not data:
            raise ValueError("Empty data file")
        
        first_item = data[0]
        
        # Legacy format: has "paragraph" field as list
        if "paragraph" in first_item:
            # Check if paragraph is a list or string representation
            para = first_item["paragraph"]
            if isinstance(para, (list, str)):
                self.query_level = "paragraph"  # Legacy format is paragraph-based
                return "legacy"
        
        # New format: has "text" field as single string
        if "text" in first_item and isinstance(first_item["text"], str):
            # Detect query level from collection field
            if "collection" in first_item:
                collection = first_item["collection"]
                if collection in ["sentences", "sentence"]:
                    self.query_level = "sentence"
                elif collection in ["paragraphs", "paragraph"]:
                    self.query_level = "paragraph"
                else:
                    self.query_level = collection  # Use whatever is specified
            return "retrieve_query_texts"
        
        raise ValueError(f"Could not detect format. Available keys: {list(first_item.keys())}")
    
    def _load_legacy_format(self, data: List[Dict]) -> Tuple[List[str], pd.DataFrame]:
        """
        Load legacy format (nested paragraph lists).
        
        Args:
            data: Raw data from JSON
            
        Returns:
            Tuple of (documents, metadata)
        """
        print("Loading legacy format...")
        
        # Parse paragraph field if it's a JSON string
        for item in data:
            if isinstance(item["paragraph"], str):
                item["paragraph"] = json.loads(item["paragraph"])
        
        # Extract documents (flatten paragraph lists)
        docs = []
        metadata_records = []
        
        for item in data:
            paragraphs = item["paragraph"]
            if not isinstance(paragraphs, list):
                paragraphs = [paragraphs]
            
            for para_text in paragraphs:
                docs.append(para_text)
                
                # Create metadata record (exclude paragraph field)
                meta = {k: v for k, v in item.items() if k != "paragraph"}
                meta["paragraph_text"] = para_text
                metadata_records.append(meta)
        
        metadata_df = pd.DataFrame(metadata_records)
        
        print(f"Loaded {len(docs)} documents from {len(data)} records")
        return docs, metadata_df
    
    def _load_retrieve_format(self, data: List[Dict]) -> Tuple[List[str], pd.DataFrame]:
        """
        Load retrieve_query_texts.py format (flat structure with text field).
        
        Args:
            data: Raw data from JSON
            
        Returns:
            Tuple of (documents, metadata)
        """
        print(f"Loading retrieve_query_texts format (mode: {self.text_mode})...")
        
        # Extract documents based on text_mode
        if self.text_mode == "result":
            # Use only the matched text
            docs = [item["text"] for item in data]
        elif self.text_mode == "result-with-context":
            # Combine text with context
            docs = []
            for item in data:
                parts = []
                if "context_before" in item and item["context_before"]:
                    parts.append(item["context_before"])
                parts.append(item["text"])
                if "context_after" in item and item["context_after"]:
                    parts.append(item["context_after"])
                docs.append(" ".join(parts))
        else:
            raise ValueError(f"Unknown text_mode: {self.text_mode}. Use 'result' or 'result-with-context'")
        
        # Create metadata DataFrame
        metadata_df = pd.DataFrame(data)
        
        print(f"Loaded {len(docs)} documents")
        return docs, metadata_df


class EmbeddingGenerator:
    """
    Generate and manage document embeddings.
    
    Supports both standard sentence-transformers models and SPECTER2 models
    through the unified model adapter. The adapter automatically detects the
    model type and uses the appropriate backend.
    
    Supported Models:
        - Standard sentence-transformers: all-MiniLM-L6-v2 (default), all-mpnet-base-v2, etc.
        - SPECTER2: allenai/specter2_base (requires 'adapters' library: pip install adapters)
    
    Example:
        # Standard model
        embedder = EmbeddingGenerator('sentence-transformers/all-MiniLM-L6-v2')
        embeddings = embedder.generate(docs)
        
        # SPECTER2 model
        embedder = EmbeddingGenerator('allenai/specter2_base')
        embeddings = embedder.generate(docs)
    """
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize embedding generator.
        
        Args:
            model_name: Name of sentence transformer or SPECTER2 model.
                       SPECTER2 models automatically use the proximity adapter.
        """
        self.model_name = model_name
        self.model = UnifiedEmbedder(model_name=model_name)
        print(f"Loaded embedding model: {model_name}")
    
    def generate(
        self, 
        docs: List[str], 
        batch_size: int = 128,
        show_progress: bool = True
    ) -> np.ndarray:
        """
        Generate embeddings for documents.
        
        Args:
            docs: List of document strings
            batch_size: Batch size for encoding
            show_progress: Show progress bar
            
        Returns:
            Numpy array of embeddings
        """
        print(f"Generating embeddings for {len(docs)} documents...")
        
        embeddings = self.model.encode(
            docs, 
            normalize_embeddings=True, 
            show_progress_bar=show_progress,
            batch_size=batch_size
        )
        
        # Clean up GPU memory
        if hasattr(self.model, 'device'):
            device_str = str(self.model.device)
            if 'cuda' in device_str:
                import torch
                torch.cuda.empty_cache()
            elif 'mps' in device_str:
                import torch
                try:
                    torch.mps.empty_cache()
                except AttributeError:
                    pass  # Older PyTorch versions
        
        print(f"Generated embeddings with shape: {embeddings.shape}")
        return embeddings
    
    def save(self, embeddings: np.ndarray, filepath: Union[str, Path]):
        """Save embeddings to file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump(embeddings, f)
        print(f"Saved embeddings to: {filepath}")
    
    @staticmethod
    def load(filepath: Union[str, Path]) -> np.ndarray:
        """Load embeddings from file."""
        with open(filepath, 'rb') as f:
            embeddings = pickle.load(f)
        print(f"Loaded embeddings from: {filepath}")
        return embeddings


class MilvusEmbeddingLoader:
    """
    Load embeddings from Milvus database collections.
    
    This allows reusing embeddings already stored in Milvus databases
    (e.g., from build_subcorpus_milvus.py) instead of recomputing them.
    
    Example:
        # Load embeddings from a Milvus collection
        loader = MilvusEmbeddingLoader(
            db_name='my_subcorpus',
            collection_name='paragraphs',
            milvus_host='localhost',
            milvus_port=19530
        )
        embeddings, metadata = loader.load_embeddings()
        docs = loader.get_documents_from_mongo(metadata)
    """
    
    def __init__(self,
                 db_name: str,
                 collection_name: str,
                 subcorpus_file: Optional[Union[str, Path]] = None,
                 milvus_host: str = 'localhost',
                 milvus_port: int = 19530,
                 mongo_host: str = 'localhost',
                 mongo_port: int = 27017):
        """
        Initialize Milvus embedding loader.
        
        Args:
            db_name: Name of the Milvus database
            collection_name: Name of the collection (e.g., 'sentences', 'paragraphs')
            subcorpus_file: Path to subcorpus file (.pkl from query_milvus_rrf.py or .json from query_subcorpus.py)
                          Required for large collections (>1M items). Supports both formats:
                          - JSON: Contains 'corpusid' field in each record
                          - Pickle: Contains 'corpus_ids' list
            milvus_host: Milvus server host
            milvus_port: Milvus server port
            mongo_host: MongoDB server host (for fetching text)
            mongo_port: MongoDB server port
        """
        if not MILVUS_AVAILABLE:
            raise ImportError(
                "Milvus support requires pymilvus and pymongo. "
                "Install with: pip install pymilvus pymongo"
            )
        
        self.db_name = db_name
        self.collection_name = collection_name
        self.subcorpus_file = Path(subcorpus_file) if subcorpus_file else None
        self.milvus_host = milvus_host
        self.milvus_port = milvus_port
        self.mongo_host = mongo_host
        self.mongo_port = mongo_port
        
        self.milvus_client = None
        self.mongo_client = None
    
    def connect(self):
        """Connect to Milvus and MongoDB."""
        print(f"Connecting to Milvus at {self.milvus_host}:{self.milvus_port}...")
        
        # Connect to Milvus
        connections.connect(host=self.milvus_host, port=self.milvus_port)
        
        # Enable the database
        db.using_database(self.db_name)
        
        self.milvus_client = MilvusClient(
            uri=f'http://{self.milvus_host}:{self.milvus_port}',
            token='root:Milvus',
            db_name=self.db_name
        )
        
        print(f"✓ Connected to Milvus database '{self.db_name}'")
        
        # Load collection
        print(f"Loading collection '{self.collection_name}'...")
        self.milvus_client.load_collection(collection_name=self.collection_name)
        print(f"✓ Collection '{self.collection_name}' loaded")
        
        # Connect to MongoDB
        print(f"Connecting to MongoDB at {self.mongo_host}:{self.mongo_port}...")
        self.mongo_client = MongoClient(self.mongo_host, self.mongo_port)
        self.mongo_client.admin.command('ping')
        print("✓ Connected to MongoDB")
    
    def load_embeddings(self,
                       limit: Optional[int] = None,
                       filter_expr: Optional[str] = None) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        Load embeddings and metadata from Milvus collection.
        
        For JSON files (from query_subcorpus.py): Loads only the specific sentences/paragraphs
        identified by corpusid + sentence_number/paragraph_number combinations.
        
        For pickle files (from query_milvus_rrf.py): Loads all sentences/paragraphs for each
        corpus_id in the file.
        
        Args:
            limit: Maximum number of embeddings to load (None = all)
            filter_expr: Optional Milvus filter expression (e.g., 'rrf_score > 0.5')
            
        Returns:
            Tuple of (embeddings array, metadata DataFrame)
        """
        if self.milvus_client is None:
            self.connect()
        
        print(f"Loading embeddings from collection '{self.collection_name}'...")
        
        # Load corpus IDs from subcorpus file (JSON or pickle)
        if self.subcorpus_file is None:
            raise ValueError(
                "subcorpus_file is required for loading embeddings. "
                "Provide path to .json file from query_subcorpus.py or .pkl file from query_milvus_rrf.py"
            )
        
        print(f"Loading data from {self.subcorpus_file}...")
        
        # Detect format by file extension
        if self.subcorpus_file.suffix == '.json':
            # JSON format from query_subcorpus.py - contains specific sentence/paragraph references
            import json
            with open(self.subcorpus_file, 'r') as f:
                subcorpus_data = json.load(f)
            
            if not isinstance(subcorpus_data, list):
                raise ValueError("Invalid JSON format - expected list of records")
            
            # Determine if we're working with sentences or paragraphs
            if subcorpus_data and 'sentence_number' in subcorpus_data[0]:
                number_field = 'sentence_number'
                print(f"JSON file contains {len(subcorpus_data)} specific sentences")
            elif subcorpus_data and 'paragraph_number' in subcorpus_data[0]:
                number_field = 'paragraph_number'
                print(f"JSON file contains {len(subcorpus_data)} specific paragraphs")
            else:
                raise ValueError("JSON records must contain 'sentence_number' or 'paragraph_number' field")
            
            # Query for each specific sentence/paragraph
            all_results = []
            processed_items = 0
            skipped_items = 0
            
            from tqdm import tqdm
            
            for record in tqdm(subcorpus_data, desc=f"Loading specific {number_field}s"):
                corpus_id = record.get('corpusid')
                number = record.get(number_field)
                
                if corpus_id is None or number is None:
                    skipped_items += 1
                    continue
                
                try:
                    # Query for this specific sentence/paragraph
                    query_filter = f"corpusid == {corpus_id} && {number_field} == {number}"
                    
                    # Add user filter if provided
                    if filter_expr:
                        # Note: filter_expr might reference fields from JSON (like rrf_score)
                        # that don't exist in Milvus. We'll apply those filters later.
                        pass
                    
                    query_params = {
                        "collection_name": self.collection_name,
                        "output_fields": ["*"],
                        "filter": query_filter,
                        "limit": 1  # Should only be one match
                    }
                    
                    batch_results = self.milvus_client.query(**query_params)
                    
                    if batch_results:
                        # Add metadata from JSON to the result
                        result = batch_results[0]
                        # Preserve fields from JSON that aren't in Milvus (e.g., rrf_score, rank)
                        for key, value in record.items():
                            if key not in result:
                                result[key] = value
                        all_results.append(result)
                        processed_items += 1
                    else:
                        skipped_items += 1
                    
                    # Check if we've reached user-specified limit
                    if limit and len(all_results) >= limit:
                        all_results = all_results[:limit]
                        print(f"\nReached limit of {limit} embeddings")
                        break
                        
                except Exception as e:
                    skipped_items += 1
                    continue
            
            print(f"\nLoaded {len(all_results)} specific {number_field}s from {processed_items} records")
            if skipped_items > 0:
                print(f"Skipped {skipped_items} records (not found in Milvus)")
                
        else:
            # Pickle format from query_milvus_rrf.py - contains corpus_ids, load all sentences/paragraphs
            import pickle
            with open(self.subcorpus_file, 'rb') as f:
                subcorpus_data = pickle.load(f)
            
            if isinstance(subcorpus_data, dict) and 'corpus_ids' in subcorpus_data:
                corpus_ids = [int(cid) for cid in subcorpus_data['corpus_ids']]
            else:
                raise ValueError("Invalid pickle format - expected dict with 'corpus_ids' key")
            
            print(f"Found {len(corpus_ids)} papers in subcorpus")
            
            all_results = []
            processed_papers = 0
            skipped_papers = 0
            
            # Process papers in batches to avoid memory issues
            from tqdm import tqdm
            
            for corpus_id in tqdm(corpus_ids, desc="Loading embeddings by paper"):
                try:
                    # Query all sentences/paragraphs for this paper
                    query_params = {
                        "collection_name": self.collection_name,
                        "output_fields": ["*"],
                        "filter": f"corpusid == {corpus_id}",
                        "limit": 16384  # Max for safety, but should never hit this per paper
                    }
                    
                    batch_results = self.milvus_client.query(**query_params)
                    
                    if batch_results:
                        all_results.extend(batch_results)
                        processed_papers += 1
                    else:
                        skipped_papers += 1
                    
                    # Check if we've reached user-specified limit
                    if limit and len(all_results) >= limit:
                        all_results = all_results[:limit]
                        print(f"\nReached limit of {limit} embeddings")
                        break
                        
                except Exception as e:
                    skipped_papers += 1
                    continue
            
            print(f"\nLoaded {len(all_results)} entries from {processed_papers} papers")
            print(f"Skipped {skipped_papers} papers (excluded/invalid/non-English)")
        
        if not all_results:
            raise ValueError(f"No data found in collection '{self.collection_name}'")
        
        # Extract embeddings and metadata
        embeddings = []
        metadata_records = []
        
        for result in all_results:
            # Extract vector
            if 'vector' in result:
                embeddings.append(result['vector'])
            
            # Extract metadata (all non-vector fields)
            metadata = {k: v for k, v in result.items() if k != 'vector'}
            metadata_records.append(metadata)
        
        embeddings_array = np.array(embeddings)
        metadata_df = pd.DataFrame(metadata_records)
        
        # Apply filter_expr if provided (for JSON files, this filters on metadata like rrf_score)
        if filter_expr and self.subcorpus_file.suffix == '.json':
            print(f"Applying filter: {filter_expr}")
            # Simple filter evaluation (supports basic comparisons)
            try:
                filtered_mask = metadata_df.eval(filter_expr)
                embeddings_array = embeddings_array[filtered_mask]
                metadata_df = metadata_df[filtered_mask].reset_index(drop=True)
                print(f"After filtering: {len(metadata_df)} entries remain")
            except Exception as e:
                print(f"Warning: Could not apply filter '{filter_expr}': {e}")
        
        print(f"Embeddings shape: {embeddings_array.shape}")
        print(f"Metadata columns: {list(metadata_df.columns)}")
        
        return embeddings_array, metadata_df
    
    def get_documents_from_mongo(self,
                                 metadata_df: pd.DataFrame,
                                 s2orc_path: Optional[Union[str, Path]] = None) -> List[str]:
        """
        Retrieve document texts from MongoDB and S2ORC using metadata.
        
        Args:
            metadata_df: DataFrame with corpusid and index information
            s2orc_path: Path to S2ORC files (required for loading paper text)
            
        Returns:
            List of document text strings
        """
        if self.mongo_client is None:
            self.connect()
        
        if s2orc_path is None:
            raise ValueError("s2orc_path is required to load document texts")
        
        from pathlib import Path
        import json
        
        s2orc_path = Path(s2orc_path)
        papers_collection = self.mongo_client.papers_db.papers
        
        print(f"Loading {len(metadata_df)} documents from S2ORC...")
        
        docs = []
        unique_corpus_ids = metadata_df['corpusid'].unique()
        
        # Cache loaded papers
        paper_cache = {}
        
        from tqdm import tqdm
        
        for idx, row in tqdm(metadata_df.iterrows(), total=len(metadata_df), desc="Loading texts"):
            corpus_id = row['corpusid']
            
            # Load paper if not cached
            if corpus_id not in paper_cache:
                # Get metadata from MongoDB
                mongo_result = papers_collection.find_one({"corpusid": corpus_id})
                
                if mongo_result is None:
                    docs.append("")
                    continue
                
                file_location = mongo_result.get('file_location')
                if not file_location:
                    docs.append("")
                    continue
                
                filename, offset = file_location
                filepath = s2orc_path / filename
                
                # Load paper from gzipped file
                try:
                    import indexed_gzip as igzip
                    with igzip.IndexedGzipFile(str(filepath)) as f:
                        f.seek(offset)
                        line = f.readline()
                        paper = json.loads(line)
                except ImportError:
                    import gzip
                    with gzip.open(filepath, 'rt') as f:
                        f.seek(offset)
                        line = f.readline()
                        paper = json.loads(line)
                
                paper_cache[corpus_id] = paper
            else:
                paper = paper_cache[corpus_id]
            
            # Extract text based on collection type
            text = paper.get("content", {}).get("text", "")
            
            if self.collection_name in ['sentences', 'sentence']:
                # Extract sentence text using indices
                if 'sentence_indices' in row:
                    indices = row['sentence_indices']
                    if isinstance(indices, list) and len(indices) == 2:
                        start, end = indices
                        text = text[start:end].strip()
            elif self.collection_name in ['paragraphs', 'paragraph']:
                # Extract paragraph text using indices
                if 'paragraph_indices' in row:
                    indices = row['paragraph_indices']
                    if isinstance(indices, list) and len(indices) == 2:
                        start, end = indices
                        text = text[start:end].strip()
            
            docs.append(text if text else "")
        
        print(f"Loaded {len(docs)} document texts")
        return docs
    
    def close(self):
        """Close connections."""
        if self.mongo_client:
            self.mongo_client.close()
        if self.milvus_client:
            connections.disconnect(alias="default")
        print("✓ Connections closed")


class TopicModeler:
    """Build and manage BERTopic models."""
    
    def __init__(self, config: Optional[ModelConfig] = None):
        """
        Initialize topic modeler.
        
        Args:
            config: Model configuration (uses defaults if None)
        """
        self.config = config or ModelConfig()
        self.model: Optional[BERTopic] = None
        self.embedding_generator = None  # Lazy initialization in fit()
        
    def _setup_dimensionality_reduction(self):
        """Setup UMAP dimensionality reduction (GPU or CPU)."""
        if self.config.use_gpu:
            try:
                from cuml import UMAP as cuUMAP
                print("Using GPU-accelerated UMAP")
                return cuUMAP(
                    n_neighbors=self.config.umap_n_neighbors,
                    n_components=self.config.umap_n_components,
                    min_dist=self.config.umap_min_dist,
                    random_state=self.config.umap_random_state
                )
            except ImportError:
                print("cuML not available, using CPU UMAP")
        
        return UMAP(
            n_neighbors=self.config.umap_n_neighbors,
            n_components=self.config.umap_n_components,
            min_dist=self.config.umap_min_dist,
            metric=self.config.umap_metric,
            low_memory=False,
            random_state=self.config.umap_random_state
        )
    
    def _setup_clustering(self):
        """Setup HDBSCAN clustering (GPU or CPU)."""
        if self.config.use_gpu:
            try:
                from cuml.cluster import HDBSCAN as cuHDBSCAN
                print("Using GPU-accelerated HDBSCAN")
                return cuHDBSCAN(
                    min_cluster_size=self.config.hdbscan_min_cluster_size,
                    cluster_selection_method=self.config.hdbscan_cluster_selection_method
                )
            except ImportError:
                print("cuML not available, using CPU HDBSCAN")
        
        return HDBSCAN(
            min_cluster_size=self.config.hdbscan_min_cluster_size,
            metric=self.config.hdbscan_metric,
            cluster_selection_method=self.config.hdbscan_cluster_selection_method,
            prediction_data=True
        )
    
    def _setup_representation_models(self) -> Dict[str, Any]:
        """Setup representation models for topic words."""
        pos_patterns_nouns = [
            [{'POS': 'ADJ'}, {'POS': 'NOUN'}],
            [{'POS': 'NOUN'}, {'POS': 'ADJ'}],
            [{'POS': 'NOUN'}],
            [{'POS': 'ADJ'}]
        ]
        
        pos_patterns_verbs = [
            [{'POS': 'VERB'}],
            [{'POS': 'ADV'}, {'POS': 'VERB'}],
            [{'POS': 'VERB'}, {'POS': 'ADV'}]
        ]
        
        main_representation = PartOfSpeech(
            "en_core_web_sm", 
            pos_patterns=pos_patterns_nouns, 
            top_n_words=30
        )
        
        keybert_mmr = [
            KeyBERTInspired(top_n_words=50),
            MaximalMarginalRelevance(diversity=0.5, top_n_words=30)
        ]
        
        verb_representation = PartOfSpeech(
            "en_core_web_sm",
            pos_patterns=pos_patterns_verbs,
            top_n_words=30
        )
        
        return {
            "Representation": main_representation,
            "Nouns and modifiers": main_representation,
            "KeyBERT + MMR": keybert_mmr,
            "Verbs and modifiers": verb_representation,
        }
    
    def _setup_vectorizer(self) -> CountVectorizer:
        """Setup count vectorizer."""
        return CountVectorizer(
            stop_words=self.config.vectorizer_stop_words,
            min_df=self.config.vectorizer_min_df,
            ngram_range=self.config.vectorizer_ngram_range,
            token_pattern=self.config.vectorizer_token_pattern
        )
    
    def fit(
        self, 
        docs: List[str], 
        embeddings: Optional[np.ndarray] = None
    ) -> Tuple[BERTopic, List[int], Optional[np.ndarray]]:
        """
        Fit topic model to documents.
        
        Args:
            docs: List of document strings
            embeddings: Pre-computed embeddings (optional)
            
        Returns:
            Tuple of (model, topics, probabilities)
        """
        print("Building BERTopic model...")
        
        # Generate embeddings if not provided
        if embeddings is None:
            # Lazy initialization of embedding generator
            if self.embedding_generator is None:
                self.embedding_generator = EmbeddingGenerator(self.config.embedding_model_name)
            
            embeddings = self.embedding_generator.generate(docs)
            # Use the embedding model in BERTopic
            embedding_model_to_use = self.embedding_generator.model
        else:
            # Pre-computed embeddings provided - create a dummy embedding model
            # BERTopic may need this for certain operations like update_topics
            print("Using pre-computed embeddings")
            
            class DummyEmbedder:
                """Dummy embedder that returns zeros - should not be called in normal flow."""
                def embed_documents(self, docs):
                    print("WARNING: Dummy embedder called - this shouldn't happen with pre-computed embeddings")
                    # Return zero vectors of the same dimension as the pre-computed embeddings
                    return np.zeros((len(docs), embeddings.shape[1]))
                
                def embed_query(self, query):
                    return np.zeros(embeddings.shape[1])
            
            embedding_model_to_use = DummyEmbedder()
        
        # Setup components
        umap_model = self._setup_dimensionality_reduction()
        hdbscan_model = self._setup_clustering()
        representation_model = self._setup_representation_models()
        vectorizer_model = self._setup_vectorizer()
        
        # Create BERTopic model
        self.model = BERTopic(
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            representation_model=representation_model,
            vectorizer_model=vectorizer_model,
            nr_topics=self.config.nr_topics,
            calculate_probabilities=self.config.calculate_probabilities,
            embedding_model=embedding_model_to_use,
            top_n_words=10,
        )
        
        print("Fitting model...")
        self.model = self.model.fit(documents=docs, embeddings=embeddings)
        
        # Fix outlier topic handling
        self.model = self._fix_outlier_topic(self.model)
        
        # Transform to get topics and probabilities
        topics, probs = self.model.transform(documents=docs, embeddings=embeddings)
        
        print("Reducing outliers...")
        # Outlier reduction
        new_topics = self.model.reduce_outliers(
            docs, 
            topics, 
            strategy=self.config.outlier_strategy,
            threshold=self.config.outlier_threshold
        )
        
        # Update topics
        self.model.update_topics(docs, topics=new_topics, vectorizer_model=vectorizer_model)
        
        print(f"Model fitted with {len(self.model.get_topic_info())} topics")
        
        # Clean up embeddings from model
        if hasattr(self.model, '_embeddings'):
            self.model._embeddings = None
        
        gc.collect()
        
        return self.model, topics, probs
    
    @staticmethod
    def _fix_outlier_topic(model: BERTopic) -> BERTopic:
        """Fix BERTopic's get_topic method to handle outlier topics properly."""
        original_get_topic = model.get_topic
        
        def patched_get_topic(topic_id):
            result = original_get_topic(topic_id)
            # If get_topic returns False (for outlier topic -1), return empty list
            if result is False:
                return [[]]
            return result
        
        model.get_topic = patched_get_topic
        return model
    
    def save(self, filepath: Union[str, Path]):
        """Save model to file."""
        if self.model is None:
            raise ValueError("No model to save. Call fit() first.")
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        self.model.save(
            str(filepath),
            serialization="safetensors",
            save_ctfidf=True,
            save_embedding_model=self.config.embedding_model_name
        )
        print(f"Saved model to: {filepath}")
    
    @staticmethod
    def load(filepath: Union[str, Path]) -> BERTopic:
        """Load model from file."""
        model = BERTopic.load(str(filepath))
        print(f"Loaded model from: {filepath}")
        return model


class ModelEvaluator:
    """Evaluate topic models using coherence metrics."""
    
    def __init__(self, coherence_metrics: Optional[List[str]] = None):
        """
        Initialize evaluator.
        
        Args:
            coherence_metrics: List of coherence metrics to calculate
        """
        self.coherence_metrics = coherence_metrics or ["c_v", "u_mass"]
    
    def evaluate(
        self, 
        docs: List[str], 
        model: BERTopic
    ) -> Dict[str, float]:
        """
        Evaluate topic model using coherence metrics.
        
        Args:
            docs: List of document strings
            model: Fitted BERTopic model
            
        Returns:
            Dictionary of coherence scores
        """
        print("Evaluating model coherence...")
        
        # Preprocess documents
        cleaned_docs = model._preprocess_text(docs)
        
        # Extract vectorizer and analyzer
        vectorizer = model.vectorizer_model
        analyzer = vectorizer.build_analyzer()
        
        # Tokenize
        tokens = [analyzer(doc) for doc in cleaned_docs]
        dictionary = corpora.Dictionary(tokens)
        corpus = [dictionary.doc2bow(token) for token in tokens]
        
        # Get topics
        topics_dict = model.get_topics()
        print(f"Number of topics: {len(topics_dict)}")
        
        # Remove outlier topic (-1) if exists
        topics_dict.pop(-1, None)
        
        # Extract valid topic IDs and words
        valid_topic_ids = [tid for tid in topics_dict.keys() if tid >= 0]
        topic_words = []
        
        for topic_id in valid_topic_ids:
            try:
                topic_data = model.get_topic(topic_id)
                if topic_data and topic_data != [[]]:  # Check if topic is not empty
                    words = [word for word, _ in topic_data]
                    topic_words.append(words)
            except Exception as e:
                print(f"Warning: Error getting topic {topic_id}: {e}")
                continue
        
        print(f"Evaluating {len(topic_words)} valid topics")
        
        if len(topic_words) == 0:
            print("Error: No valid topics found for coherence calculation!")
            return {"error": "No valid topics"}
        
        # Calculate coherence for each metric
        coherence_scores = {}
        for metric in self.coherence_metrics:
            try:
                print(f"Calculating {metric} coherence...")
                coherence_model = CoherenceModel(
                    topics=topic_words,
                    texts=tokens,
                    corpus=corpus,
                    dictionary=dictionary,
                    coherence=metric,
                    processes=1
                )
                coherence_scores[metric] = coherence_model.get_coherence()
                print(f"{metric} coherence = {coherence_scores[metric]:.4f}")
                
                # Cleanup
                del coherence_model
                gc.collect()
                
            except Exception as e:
                print(f"Error calculating {metric} coherence: {e}")
                coherence_scores[metric] = None
        
        # Cleanup
        del tokens, dictionary, corpus, topic_words
        gc.collect()
        
        return coherence_scores


class Visualizer:
    """Generate visualizations for topic models."""
    
    def __init__(self, output_dir: Union[str, Path]):
        """
        Initialize visualizer.
        
        Args:
            output_dir: Directory to save visualizations
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_all(
        self,
        model: BERTopic,
        docs: List[str],
        embeddings: np.ndarray,
        name_suffix: str = ""
    ):
        """
        Generate all available visualizations.
        
        Args:
            model: Fitted BERTopic model
            docs: List of document strings
            embeddings: Document embeddings
            name_suffix: Suffix for output filenames
        """
        print("Generating visualizations...")
        
        # Document visualization
        self._visualize_documents(model, docs, embeddings, name_suffix)
        
        # Hierarchical topics
        self._visualize_hierarchy(model, docs, name_suffix)
        
        # Barchart
        self._visualize_barchart(model, name_suffix)
        
        # Heatmap
        self._visualize_heatmap(model, name_suffix)
        
        print("Visualization generation completed")
    
    def _visualize_documents(
        self, 
        model: BERTopic, 
        docs: List[str], 
        embeddings: np.ndarray,
        name_suffix: str
    ):
        """Generate custom document visualizations using UMAP and t-SNE with tabbed interface."""
        try:
            import plotly.graph_objects as go
            from umap import UMAP
            from sklearn.manifold import TSNE
            
            print("Creating custom document visualizations...")
            print(f"  Documents: {len(docs)}")
            print(f"  Embeddings shape: {embeddings.shape}")
            
            # Get topics and topic info
            topics_list = model.topics_
            topic_info = model.get_topic_info()
            
            # Count outliers
            outlier_count = sum(1 for t in topics_list if t == -1)
            print(f"  Outliers: {outlier_count}")
            print(f"  Topics: {len(set(topics_list)) - (1 if -1 in topics_list else 0)}")
            
            # Create color mapping for topics with extended palette
            unique_topics = sorted([t for t in set(topics_list) if t != -1])
            # Use Plotly's extended color palettes for unlimited topics
            import plotly.express as px
            color_palette = (
                px.colors.qualitative.Light24 +
                px.colors.qualitative.Dark24 +
                px.colors.qualitative.Alphabet
            )
            
            topic_colors = {}
            for idx, topic_id in enumerate(unique_topics):
                topic_colors[topic_id] = color_palette[idx % len(color_palette)]
            topic_colors[-1] = '#cccccc'  # Gray for outliers
            
            # Get topic names
            topic_names = {}
            for _, row in topic_info.iterrows():
                topic_id = row['Topic']
                # Handle both string and non-string Name values
                if 'Name' in row.index:
                    name = str(row['Name']) if row['Name'] is not None else f"Topic {topic_id}"
                else:
                    name = f"Topic {topic_id}"
                topic_names[topic_id] = name
            
            print(f"  Sample topic names: {list(topic_names.items())[:3]}")
            
            # Reduce embeddings to 2D and 3D using UMAP and t-SNE
            print("  Reducing dimensions with UMAP...")
            print(f"    Input embeddings shape: {embeddings.shape}")
            
            umap_2d = UMAP(n_components=2, n_neighbors=15, min_dist=0.0, metric='cosine', random_state=42)
            coords_umap_2d = umap_2d.fit_transform(embeddings)
            print(f"    ✓ UMAP 2D coords shape: {coords_umap_2d.shape}")
            print(f"    ✓ UMAP 2D range: X=[{coords_umap_2d[:, 0].min():.2f}, {coords_umap_2d[:, 0].max():.2f}], Y=[{coords_umap_2d[:, 1].min():.2f}, {coords_umap_2d[:, 1].max():.2f}]")
            
            umap_3d = UMAP(n_components=3, n_neighbors=15, min_dist=0.0, metric='cosine', random_state=42)
            coords_umap_3d = umap_3d.fit_transform(embeddings)
            print(f"    ✓ UMAP 3D coords shape: {coords_umap_3d.shape}")
            print(f"    ✓ UMAP 3D range: X=[{coords_umap_3d[:, 0].min():.2f}, {coords_umap_3d[:, 0].max():.2f}], Y=[{coords_umap_3d[:, 1].min():.2f}, {coords_umap_3d[:, 1].max():.2f}], Z=[{coords_umap_3d[:, 2].min():.2f}, {coords_umap_3d[:, 2].max():.2f}]")
            
            print("  Reducing dimensions with t-SNE...")
            tsne_2d = TSNE(n_components=2, perplexity=min(30, len(docs)-1), random_state=42, max_iter=1000)
            coords_tsne_2d = tsne_2d.fit_transform(embeddings)
            print(f"    ✓ t-SNE 2D coords shape: {coords_tsne_2d.shape}")
            print(f"    ✓ t-SNE 2D range: X=[{coords_tsne_2d[:, 0].min():.2f}, {coords_tsne_2d[:, 0].max():.2f}], Y=[{coords_tsne_2d[:, 1].min():.2f}, {coords_tsne_2d[:, 1].max():.2f}]")
            
            tsne_3d = TSNE(n_components=3, perplexity=min(30, len(docs)-1), random_state=42, max_iter=1000)
            coords_tsne_3d = tsne_3d.fit_transform(embeddings)
            print(f"    ✓ t-SNE 3D coords shape: {coords_tsne_3d.shape}")
            print(f"    ✓ t-SNE 3D range: X=[{coords_tsne_3d[:, 0].min():.2f}, {coords_tsne_3d[:, 0].max():.2f}], Y=[{coords_tsne_3d[:, 1].min():.2f}, {coords_tsne_3d[:, 1].max():.2f}], Z=[{coords_tsne_3d[:, 2].min():.2f}, {coords_tsne_3d[:, 2].max():.2f}]")
            
            # Helper function to create scatter plot
            def create_scatter_plot(coords, is_3d=False):
                traces = []
                print(f"    Creating {'3D' if is_3d else '2D'} scatter plot with coords shape {coords.shape}")
                
                # Include all topics, not just first 20
                for topic_id in unique_topics + ([-1] if outlier_count > 0 else []):
                    mask = [t == topic_id for t in topics_list]
                    num_points = sum(mask)
                    
                    if num_points == 0:
                        continue
                        
                    topic_name = topic_names.get(topic_id, f"Topic {topic_id}")
                    color = topic_colors.get(topic_id, '#cccccc')
                    
                    # Prepare hover text
                    hover_texts = []
                    for i, is_topic in enumerate(mask):
                        if is_topic:
                            doc_preview = docs[i][:100] + "..." if len(docs[i]) > 100 else docs[i]
                            hover_texts.append(f"Topic: {topic_name}<br>Doc: {doc_preview}")
                    
                    if is_3d:
                        trace = go.Scatter3d(
                            x=coords[mask, 0],
                            y=coords[mask, 1],
                            z=coords[mask, 2],
                            mode='markers',
                            name=topic_name,
                            marker=dict(size=3, color=color, opacity=0.6),
                            text=hover_texts,
                            hovertemplate='%{text}<extra></extra>',
                            showlegend=True  # Show all topics in legend
                        )
                    else:
                        trace = go.Scatter(
                            x=coords[mask, 0],
                            y=coords[mask, 1],
                            mode='markers',
                            name=topic_name,
                            marker=dict(size=5, color=color, opacity=0.6),
                            text=hover_texts,
                            hovertemplate='%{text}<extra></extra>',
                            showlegend=True  # Show all topics in legend
                        )
                    traces.append(trace)
                    
                    if topic_id in unique_topics[:3] or topic_id == -1:
                        print(f"      Topic {topic_id} ({topic_name[:30]}): {num_points} points")
                
                print(f"    ✓ Created {len(traces)} traces")
                return traces
            
            # Create 4 separate figures for tabs
            print("  Creating plotly figures...")
            fig_umap_2d = go.Figure(data=create_scatter_plot(coords_umap_2d, is_3d=False))
            fig_umap_2d.update_layout(
                title=f"UMAP 2D - Document Clustering ({len(docs)} docs, {len(unique_topics)} topics)",
                xaxis_title="UMAP-1",
                yaxis_title="UMAP-2",
                height=700,
                hovermode='closest'
            )
            print(f"    ✓ UMAP 2D figure created with {len(fig_umap_2d.data)} traces")
            
            fig_umap_3d = go.Figure(data=create_scatter_plot(coords_umap_3d, is_3d=True))
            fig_umap_3d.update_layout(
                title=f"UMAP 3D - Document Clustering ({len(docs)} docs, {len(unique_topics)} topics)",
                scene=dict(
                    xaxis_title="UMAP-1",
                    yaxis_title="UMAP-2",
                    zaxis_title="UMAP-3"
                ),
                height=700,
                hovermode='closest'
            )
            print(f"    ✓ UMAP 3D figure created with {len(fig_umap_3d.data)} traces")
            
            fig_tsne_2d = go.Figure(data=create_scatter_plot(coords_tsne_2d, is_3d=False))
            fig_tsne_2d.update_layout(
                title=f"t-SNE 2D - Document Clustering ({len(docs)} docs, {len(unique_topics)} topics)",
                xaxis_title="t-SNE-1",
                yaxis_title="t-SNE-2",
                height=700,
                hovermode='closest'
            )
            print(f"    ✓ t-SNE 2D figure created with {len(fig_tsne_2d.data)} traces")
            
            fig_tsne_3d = go.Figure(data=create_scatter_plot(coords_tsne_3d, is_3d=True))
            fig_tsne_3d.update_layout(
                title=f"t-SNE 3D - Document Clustering ({len(docs)} docs, {len(unique_topics)} topics)",
                scene=dict(
                    xaxis_title="t-SNE-1",
                    yaxis_title="t-SNE-2",
                    zaxis_title="t-SNE-3"
                ),
                height=700,
                hovermode='closest'
            )
            print(f"    ✓ t-SNE 3D figure created with {len(fig_tsne_3d.data)} traces")
            
            # Save each figure as a separate HTML file
            print("  Saving individual HTML files...")
            
            output_base = self.output_dir / f"vis_documents{name_suffix}"
            
            # UMAP 2D
            umap_2d_file = str(output_base) + "_umap_2d.html"
            fig_umap_2d.write_html(umap_2d_file)
            file_size = Path(umap_2d_file).stat().st_size / 1024
            print(f"    ✓ UMAP 2D saved to {Path(umap_2d_file).name} ({file_size:.1f} KB)")
            
            # UMAP 3D
            umap_3d_file = str(output_base) + "_umap_3d.html"
            fig_umap_3d.write_html(umap_3d_file)
            file_size = Path(umap_3d_file).stat().st_size / 1024
            print(f"    ✓ UMAP 3D saved to {Path(umap_3d_file).name} ({file_size:.1f} KB)")
            
            # t-SNE 2D
            tsne_2d_file = str(output_base) + "_tsne_2d.html"
            fig_tsne_2d.write_html(tsne_2d_file)
            file_size = Path(tsne_2d_file).stat().st_size / 1024
            print(f"    ✓ t-SNE 2D saved to {Path(tsne_2d_file).name} ({file_size:.1f} KB)")
            
            # t-SNE 3D
            tsne_3d_file = str(output_base) + "_tsne_3d.html"
            fig_tsne_3d.write_html(tsne_3d_file)
            file_size = Path(tsne_3d_file).stat().st_size / 1024
            print(f"    ✓ t-SNE 3D saved to {Path(tsne_3d_file).name} ({file_size:.1f} KB)")
            
            print(f"✅ Document visualizations saved as 4 separate HTML files")
                    
        except Exception as e:
            import traceback
            import sys
            print(f"❌ Document visualization failed: {e}")
            print("  Full stack trace:")
            traceback.print_exc(file=sys.stdout)
    
    def _visualize_hierarchy(
        self, 
        model: BERTopic, 
        docs: List[str],
        name_suffix: str
    ):
        """Generate hierarchical topics visualization."""
        try:
            print("Creating hierarchical topics visualization...")
            hierarchical_topics = model.hierarchical_topics(docs)
            
            if hierarchical_topics is not None and not isinstance(hierarchical_topics, bool):
                fig = model.visualize_hierarchy(hierarchical_topics=hierarchical_topics)
                
                output_file = self.output_dir / f"vis_hierarchy{name_suffix}"
                fig.write_html(str(output_file.with_suffix('.html')))
                fig.write_image(str(output_file.with_suffix('.png')), scale=3)
                
                print("✅ Hierarchical visualization complete")
            else:
                print("⚠️ Hierarchical topics returned invalid data, skipping")
        except Exception as e:
            import traceback
            print(f"❌ Hierarchical visualization failed: {e}")
            print("  Full traceback:")
            traceback.print_exc()
    
    def _visualize_barchart(self, model: BERTopic, name_suffix: str):
        """Generate barchart visualization."""
        try:
            print("Creating barchart visualization...")
            fig = model.visualize_barchart(top_n_topics=20, n_words=7)
            
            output_file = self.output_dir / f"vis_barchart{name_suffix}"
            fig.write_html(str(output_file.with_suffix('.html')))
            fig.write_image(str(output_file.with_suffix('.png')), scale=3)
            
            print("✅ Barchart visualization complete")
        except Exception as e:
            import traceback
            print(f"❌ Barchart visualization failed: {e}")
            print("  Full traceback:")
            traceback.print_exc()
    
    def _visualize_heatmap(self, model: BERTopic, name_suffix: str):
        """Generate heatmap visualization."""
        try:
            if len(model.get_topic_info()) > 8:
                print("Creating heatmap visualization...")
                fig = model.visualize_heatmap(n_clusters=8)
                
                output_file = self.output_dir / f"vis_heatmap{name_suffix}"
                fig.write_html(str(output_file.with_suffix('.html')))
                fig.write_image(str(output_file.with_suffix('.png')), scale=3)
                
                print("✅ Heatmap visualization complete")
            else:
                print("⚠️ Too few topics for heatmap (need > 8)")
        except Exception as e:
            import traceback
            print(f"❌ Heatmap visualization failed: {e}")
            print("  Full traceback:")
            traceback.print_exc()


class TopicDistributionAnalyzer:
    """Analyze and export topic distributions."""
    
    def __init__(self, output_dir: Union[str, Path]):
        """
        Initialize analyzer.
        
        Args:
            output_dir: Directory to save outputs
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def compute_document_info(
        self,
        model: BERTopic,
        docs: List[str],
        metadata_df: pd.DataFrame,
        name_suffix: str = ""
    ) -> pd.DataFrame:
        """
        Compute document-level topic information.
        
        Args:
            model: Fitted BERTopic model
            docs: List of document strings
            metadata_df: DataFrame with document metadata
            name_suffix: Suffix for output filename
            
        Returns:
            DataFrame with document info and topic assignments
        """
        print("Computing document-level topic information...")
        
        # Align lengths
        min_length = min(len(docs), len(metadata_df))
        if len(docs) != len(metadata_df):
            print(f"⚠️ Length mismatch. Truncating to {min_length} entries.")
        
        docs_subset = docs[:min_length]
        metadata_subset = metadata_df.iloc[:min_length].reset_index(drop=True)
        
        try:
            # Try BERTopic's built-in method first
            if hasattr(model, "get_document_info"):
                doc_info = model.get_document_info(docs_subset, df=metadata_subset)
                if isinstance(doc_info, pd.DataFrame):
                    output_file = self.output_dir / f"document_info{name_suffix}.csv"
                    doc_info.to_csv(output_file, index=False)
                    print(f"✅ Saved document info to: {output_file}")
                    return doc_info
        except Exception as e:
            print(f"Built-in method failed: {e}")
        
        # Fallback: manual creation
        print("Creating document info manually...")
        
        doc_topics, doc_probs = model.transform(docs_subset)
        
        # Safe probability extractor
        def _prob_value(p):
            if p is None:
                return 0.0
            if isinstance(p, (np.floating, np.integer, float, int)):
                return float(p)
            try:
                if hasattr(p, "__len__") and len(p) > 0:
                    return float(max(p))
            except Exception:
                pass
            return 0.0
        
        # Build document info
        doc_info_data = {
            'Document': docs_subset,
            'Topic': list(doc_topics),
            'Probability': [_prob_value(p) for p in (doc_probs if doc_probs is not None else [None]*len(docs_subset))]
        }
        
        # Add topic names
        topic_names = {}
        for topic_id in set(doc_topics):
            try:
                if topic_id >= 0:
                    topic_words = model.get_topic(topic_id)
                    if topic_words and topic_words != [[]]:
                        top_words = [w for w, _ in topic_words[:3]]
                        topic_names[topic_id] = "_".join(top_words)
                    else:
                        topic_names[topic_id] = f"Topic_{topic_id}"
                else:
                    topic_names[topic_id] = "Outlier"
            except Exception:
                topic_names[topic_id] = f"Topic_{topic_id}"
        
        doc_info_data['Name'] = [topic_names.get(t, f"Unknown_{t}") for t in doc_info_data['Topic']]
        
        # Add metadata
        for col in metadata_subset.columns:
            if col not in doc_info_data:
                doc_info_data[col] = metadata_subset[col].tolist()
        
        doc_info = pd.DataFrame(doc_info_data)
        
        output_file = self.output_dir / f"document_info{name_suffix}.csv"
        doc_info.to_csv(output_file, index=False)
        print(f"✅ Saved document info to: {output_file}")
        
        return doc_info
    
    def compute_topic_distributions(
        self,
        model: BERTopic,
        docs: List[str],
        metadata_df: pd.DataFrame,
        name_suffix: str = ""
    ) -> pd.DataFrame:
        """
        Compute topic distribution for each document.
        
        Args:
            model: Fitted BERTopic model
            docs: List of document strings
            metadata_df: DataFrame with document metadata
            name_suffix: Suffix for output filename
            
        Returns:
            DataFrame with topic distributions
        """
        print("Computing topic distributions...")
        
        topic_distr, _ = model.approximate_distribution(docs)
        topic_distr_df = pd.DataFrame(
            topic_distr,
            columns=[f"topic_id={i}" for i in range(topic_distr.shape[1])]
        )
        
        # Combine with metadata
        result_df = pd.concat([metadata_df.reset_index(drop=True), topic_distr_df], axis=1)
        
        output_file = self.output_dir / f"topic_distributions{name_suffix}.csv"
        result_df.to_csv(output_file, index=False)
        print(f"✅ Saved topic distributions to: {output_file}")
        
        return result_df


# Convenience function for testing topic number ranges
def test_topic_numbers(
    docs: List[str],
    embeddings: np.ndarray,
    range_min: int = 5,
    range_max: int = 100,
    step: int = 5,
    output_dir: Union[str, Path] = "./topic_models"
) -> pd.DataFrame:
    """
    Test different numbers of topics and evaluate coherence.
    
    Args:
        docs: List of document strings
        embeddings: Document embeddings
        range_min: Minimum number of topics
        range_max: Maximum number of topics
        step: Step size
        output_dir: Output directory for results
        
    Returns:
        DataFrame with coherence scores for each topic count
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    coherences = {}
    
    for nr_topics in range(range_min, range_max, step):
        start_time = time.time()
        print(f"\n{'='*60}")
        print(f"Testing {nr_topics} topics")
        print(f"{'='*60}")
        
        try:
            # Create config
            config = ModelConfig(nr_topics=nr_topics)
            
            # Build model
            modeler = TopicModeler(config)
            model, topics, probs = modeler.fit(docs, embeddings)
            
            # Evaluate
            evaluator = ModelEvaluator()
            coherence = evaluator.evaluate(docs, model)
            
            coherences[nr_topics] = coherence
            print(f"Coherence for {nr_topics} topics: {coherence}")
            print(f"Elapsed: {time.time() - start_time:.1f}s")
            
            # Cleanup
            del model, topics, probs
            EnvironmentSetup.cleanup_memory()
            
        except Exception as e:
            print(f"Error with {nr_topics} topics: {e}")
            coherences[nr_topics] = {"error": str(e)}
            EnvironmentSetup.cleanup_memory()
    
    # Save results
    results_df = pd.DataFrame.from_dict(coherences, orient="index")
    output_file = output_dir / f"coherence_test_{range_min}-{range_max}-{step}.csv"
    results_df.to_csv(output_file)
    print(f"\n✅ Saved coherence test results to: {output_file}")
    
    return results_df


def build_all_text_mode_models(
    data_path: Union[str, Path],
    config: ModelConfig,
    text_modes: Optional[List[str]] = None,
    output_dir: Optional[Union[str, Path]] = None
) -> Dict[str, Dict[str, Any]]:
    """
    Build topic models for multiple text modes from the same data file.
    
    This is useful for comparing results from analyzing just the query result
    vs. the result with full context.
    
    Args:
        data_path: Path to input JSON file
        config: Base model configuration (will be copied and modified for each mode)
        text_modes: List of text modes to process (default: ["result", "result-with-context"])
        output_dir: Output directory (overrides config.output_dir if provided)
        
    Returns:
        Dictionary mapping text_mode to results dict containing:
            - model: Trained BERTopic model
            - topics: Topic assignments
            - probs: Topic probabilities
            - docs: Document list
            - metadata_df: Metadata DataFrame
            - embeddings: Document embeddings
            - coherence: Coherence scores
    """
    if text_modes is None:
        text_modes = ["result", "result-with-context"]
    
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    for text_mode in text_modes:
        print(f"\n{'='*80}")
        print(f"Building model for text_mode: {text_mode}")
        print(f"{'='*80}\n")
        
        # Load data with this text mode
        loader = DataLoader(data_path, text_mode=text_mode)
        docs, metadata_df = loader.load()
        
        # Create config for this mode
        mode_config = ModelConfig(
            embedding_model_name=config.embedding_model_name,
            nr_topics=config.nr_topics,
            calculate_probabilities=config.calculate_probabilities,
            query_level=loader.query_level,
            text_mode=text_mode,
            umap_n_neighbors=config.umap_n_neighbors,
            umap_n_components=config.umap_n_components,
            umap_min_dist=config.umap_min_dist,
            umap_metric=config.umap_metric,
            umap_random_state=config.umap_random_state,
            hdbscan_min_cluster_size=config.hdbscan_min_cluster_size,
            hdbscan_metric=config.hdbscan_metric,
            hdbscan_cluster_selection_method=config.hdbscan_cluster_selection_method,
            vectorizer_stop_words=config.vectorizer_stop_words,
            vectorizer_min_df=config.vectorizer_min_df,
            vectorizer_ngram_range=config.vectorizer_ngram_range,
            vectorizer_token_pattern=config.vectorizer_token_pattern,
            outlier_strategy=config.outlier_strategy,
            outlier_threshold=config.outlier_threshold,
            use_gpu=config.use_gpu,
            output_dir=output_dir if output_dir else config.output_dir,
            checkpoint_interval=config.checkpoint_interval
        )
        
        # Generate model suffix
        model_suffix = mode_config.get_model_suffix()
        print(f"Model suffix: {model_suffix}\\n")
        
        # Generate embeddings
        embeddings_file = mode_config.output_dir / f"embeddings_{model_suffix}.pkl"
        if embeddings_file.exists():
            print(f"Loading pre-computed embeddings from {embeddings_file}")
            embeddings = EmbeddingGenerator.load(embeddings_file)
        else:
            print("Generating embeddings...")
            embedding_gen = EmbeddingGenerator(mode_config.embedding_model_name)
            embeddings = embedding_gen.generate(docs, batch_size=128, show_progress=True)
            embedding_gen.save(embeddings, embeddings_file)
        
        # Build model
        print("\\nBuilding topic model...")
        modeler = TopicModeler(mode_config)
        model, topics, probs = modeler.fit(docs, embeddings)
        
        # Evaluate
        print("\\nEvaluating model...")
        evaluator = ModelEvaluator()
        coherence = evaluator.evaluate(docs, model)
        
        # Save model
        model_file = mode_config.output_dir / f"topic_model_{model_suffix}.safetensors"
        modeler.save(model_file)
        
        # Save topic info
        topic_info = model.get_topic_info()
        topic_info_file = mode_config.output_dir / f"topic_info_{model_suffix}.csv"
        topic_info.to_csv(topic_info_file, index=False)
        
        # Generate visualizations
        print("\\nGenerating visualizations...")
        visualizer = Visualizer(mode_config.output_dir)
        visualizer.generate_all(model, docs, embeddings, name_suffix=f"_{model_suffix}")
        
        # Export results
        print("\\nExporting results...")
        analyzer = TopicDistributionAnalyzer(mode_config.output_dir)
        doc_info = analyzer.compute_document_info(
            model, docs, metadata_df, name_suffix=f"_{model_suffix}"
        )
        topic_dist = analyzer.compute_topic_distributions(
            model, docs, metadata_df, name_suffix=f"_{model_suffix}"
        )
        
        # Store results
        results[text_mode] = {
            "model": model,
            "topics": topics,
            "probs": probs,
            "docs": docs,
            "metadata_df": metadata_df,
            "embeddings": embeddings,
            "coherence": coherence,
            "config": mode_config,
            "model_suffix": model_suffix,
            "doc_info": doc_info,
            "topic_dist": topic_dist
        }
        
        print(f"\\n✅ Completed model for {text_mode}")
        print(f"   Model saved: {model_file}")
        print(f"   Coherence: {coherence}")
        
        # Cleanup
        EnvironmentSetup.cleanup_memory()
    
    print(f"\\n{'='*80}")
    print("All models completed!")
    print(f"{'='*80}")
    
    return results


def build_topic_model_from_milvus(
    db_name: str,
    collection_name: str,
    subcorpus_file: Union[str, Path],
    config: ModelConfig,
    s2orc_path: Union[str, Path],
    milvus_host: str = 'localhost',
    milvus_port: int = 19530,
    mongo_host: str = 'localhost',
    mongo_port: int = 27017,
    limit: Optional[int] = None,
    filter_expr: Optional[str] = None,
    output_dir: Optional[Union[str, Path]] = None,
    export_doc_info: bool = False,
    skip_doc_info: bool = False
) -> Dict[str, Any]:
    """
    Build a topic model from embeddings stored in a Milvus collection.
    
    This function loads pre-computed embeddings from a Milvus database
    (e.g., created by build_subcorpus_milvus.py) and uses them for
    topic modeling, avoiding the need to recompute embeddings.
    
    Args:
        db_name: Name of the Milvus database
        collection_name: Name of the collection (e.g., 'sentences', 'paragraphs')
        subcorpus_file: Path to subcorpus pickle file from query_milvus_rrf.py
        config: Model configuration
        s2orc_path: Path to S2ORC gzipped files (for loading document texts)
        milvus_host: Milvus server host
        milvus_port: Milvus server port
        mongo_host: MongoDB server host
        mongo_port: MongoDB server port
        limit: Maximum number of documents to process (None = all)
        filter_expr: Optional Milvus filter expression
        output_dir: Output directory (overrides config.output_dir if provided)
        export_doc_info: Force export of document_info CSV even for large datasets
        skip_doc_info: Never export document_info CSV regardless of dataset size
        
    Returns:
        Dictionary containing:
            - model: Trained BERTopic model
            - topics: Topic assignments
            - probs: Topic probabilities
            - docs: Document list
            - metadata_df: Metadata DataFrame
            - embeddings: Document embeddings
            - coherence: Coherence scores
            - config: Model configuration
    
    Example:
        config = ModelConfig(nr_topics=None, use_gpu=True)
        results = build_topic_model_from_milvus(
            db_name='my_subcorpus',
            collection_name='paragraphs',
            subcorpus_file='subcorpus_results.pkl',
            config=config,
            s2orc_path='/path/to/s2orc',
            limit=1000
        )
    """
    print(f"\\n{'='*80}")
    print(f"Building topic model from Milvus collection")
    print(f"  Database: {db_name}")
    print(f"  Collection: {collection_name}")
    print(f"{'='*80}\\n")
    
    if output_dir is not None:
        config.output_dir = Path(output_dir)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set query level from collection name
    if 'sentence' in collection_name.lower():
        config.query_level = 'sentence'
    elif 'paragraph' in collection_name.lower():
        config.query_level = 'paragraph'
    else:
        config.query_level = collection_name
    
    config.text_mode = 'milvus'
    
    # Initialize Milvus loader
    loader = MilvusEmbeddingLoader(
        db_name=db_name,
        collection_name=collection_name,
        subcorpus_file=subcorpus_file,
        milvus_host=milvus_host,
        milvus_port=milvus_port,
        mongo_host=mongo_host,
        mongo_port=mongo_port
    )
    
    # Load embeddings from Milvus
    print("Loading embeddings from Milvus...")
    embeddings, metadata_df = loader.load_embeddings(limit=limit, filter_expr=filter_expr)
    
    # Check for cached documents
    cache_file = config.output_dir / f"docs_cache_{db_name}_{collection_name}.pkl"
    
    if cache_file.exists():
        print(f"\nLoading cached documents from {cache_file}...")
        import pickle
        with open(cache_file, 'rb') as f:
            cached_data = pickle.load(f)
        
        # Verify cache matches current metadata (check if same corpus_ids and indices)
        cache_valid = False
        if 'metadata_hash' in cached_data:
            # Create hash of current metadata
            import hashlib
            metadata_str = str(sorted(metadata_df[['corpusid']].values.tolist()))
            current_hash = hashlib.md5(metadata_str.encode()).hexdigest()
            cache_valid = (cached_data['metadata_hash'] == current_hash)
        
        if cache_valid:
            print("✓ Cache is valid, using cached documents")
            docs = cached_data['docs']
        else:
            print("✗ Cache is outdated, reloading documents...")
            docs = None
    else:
        docs = None
    
    # Load document texts from MongoDB/S2ORC if not cached
    if docs is None:
        print("\nLoading document texts from S2ORC...")
        docs = loader.get_documents_from_mongo(metadata_df, s2orc_path=s2orc_path)
        
        # Save to cache
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
        print("✓ Cache saved")
    
    # Filter out empty documents
    valid_indices = [i for i, doc in enumerate(docs) if doc.strip()]
    if len(valid_indices) < len(docs):
        print(f"Filtered {len(docs) - len(valid_indices)} empty documents")
        docs = [docs[i] for i in valid_indices]
        embeddings = embeddings[valid_indices]
        metadata_df = metadata_df.iloc[valid_indices].reset_index(drop=True)
    
    print(f"\\nProcessing {len(docs)} documents")
    
    # Generate model suffix
    model_suffix = f"{db_name}_{collection_name}_{config.get_model_suffix()}"
    print(f"Model suffix: {model_suffix}\\n")
    
    # Build model
    print("Building topic model...")
    modeler = TopicModeler(config)
    model, topics, probs = modeler.fit(docs, embeddings)
    
    # Evaluate
    print("\\nEvaluating model...")
    evaluator = ModelEvaluator()
    coherence = evaluator.evaluate(docs, model)
    
    # Save model
    model_file = config.output_dir / f"topic_model_{model_suffix}.safetensors"
    modeler.save(model_file)
    
    # Save topic info
    topic_info = model.get_topic_info()
    topic_info_file = config.output_dir / f"topic_info_{model_suffix}.csv"
    topic_info.to_csv(topic_info_file, index=False)
    
    # Generate visualizations
    print("\\nGenerating visualizations...")
    visualizer = Visualizer(config.output_dir)
    visualizer.generate_all(model, docs, embeddings, name_suffix=f"_{model_suffix}")
    
    # Export results
    print("\\nExporting results...")
    analyzer = TopicDistributionAnalyzer(config.output_dir)
    
    # Determine whether to export document info based on size and user preferences
    doc_count = len(docs)
    should_export_doc_info = True
    doc_info_exported = False
    
    if skip_doc_info:
        should_export_doc_info = False
        print(f"Skipping document_info export (user requested)")
    elif not export_doc_info and doc_count > 10000:
        should_export_doc_info = False
        print(f"Skipping document_info export ({doc_count} docs > 10,000 threshold)")
        print(f"  Use export_doc_info=True to force export for large datasets")
    else:
        if doc_count > 10000:
            print(f"Exporting document_info for {doc_count} docs (user forced)...")
        else:
            print(f"Exporting document_info for {doc_count} docs...")
    
    if should_export_doc_info:
        doc_info = analyzer.compute_document_info(
            model, docs, metadata_df, name_suffix=f"_{model_suffix}"
        )
        doc_info_exported = True
    else:
        doc_info = None
    
    topic_dist = analyzer.compute_topic_distributions(
        model, docs, metadata_df, name_suffix=f"_{model_suffix}"
    )
    
    # Close connections
    loader.close()
    
    # Cleanup
    EnvironmentSetup.cleanup_memory()
    
    print(f"\\n✅ Model completed!")
    print(f"   Model saved: {model_file}")
    print(f"   Coherence: {coherence}")
    print(f"\\n{'='*80}\\n")
    
    return {
        "model": model,
        "topics": topics,
        "probs": probs,
        "docs": docs,
        "metadata_df": metadata_df,
        "embeddings": embeddings,
        "coherence": coherence,
        "config": config,
        "model_suffix": model_suffix,
        "doc_info": doc_info,
        "topic_dist": topic_dist,
        "doc_info_exported": doc_info_exported
    }


if __name__ == "__main__":
    print("Topic Modeling Analysis Module")
    print("For usage examples, see: topic_modeling_notebook.ipynb")
