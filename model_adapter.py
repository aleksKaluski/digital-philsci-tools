"""
Unified model adapter for embedding generation.

This module provides a consistent interface for generating embeddings from text,
supporting both standard sentence-transformers models and specialized models
like SPECTER2 that require the adapters library.

Key Differences Between Backends:

sentence-transformers:
    - High-level wrapper around transformers
    - Automatic pooling and embedding extraction
    - Simple .encode() API
    - Works with standard Hugging Face models
    - Cannot load models with custom architectures or adapters

transformers + adapters (for SPECTER2):
    - Lower-level, more flexible
    - manual tokenization and embedding extraction (CLS token)
    - Task-specific adapters (proximity, classification, regression, adhoc_query)
    - Required for SPECTER2 family models with adapter architecture

Installation:
    # Standard models (already installed)
    pip install sentence-transformers
    
    # SPECTER2 support (optional)
    pip install adapters

Usage:
    from model_adapter import UnifiedEmbedder
    
    # Standard sentence-transformers model
    embedder = UnifiedEmbedder('multi-qa-MiniLM-L6-cos-v1')
    embeddings = embedder.encode(['text1', 'text2'])
    
    # SPECTER2 with default proximity adapter
    embedder = UnifiedEmbedder('allenai/specter2_base')
    embeddings = embedder.encode(['text1', 'text2'])
    
    # SPECTER2 with specific adapter
    embedder = UnifiedEmbedder(
        'allenai/specter2_base',
        adapter='allenai/specter2_adhoc_query'
    )
    embeddings = embedder.encode(['short query text'])

SPECTER2 Adapters:
    - allenai/specter2: Proximity/retrieval (default)
    - allenai/specter2_adhoc_query: Short text queries
    - allenai/specter2_classification: Classification features
    - allenai/specter2_regression: Regression features

Backend Selection:
    The embedder automatically detects SPECTER2 models by checking for
    'specter2' or 'specter-2' in the model name. All other models use
    the sentence-transformers backend.
"""

from typing import List, Optional, Union
import numpy as np
import torch


class UnifiedEmbedder:
    """
    Unified interface for generating text embeddings.
    
    Automatically detects model type and uses appropriate backend:
    - sentence-transformers for standard models
    - transformers + adapters for SPECTER2 models
    """
    
    def __init__(self, 
                 model_name: str,
                 adapter: Optional[str] = None,
                 device: Optional[str] = None,
                 use_gpu: bool = True):
        """
        Initialize embedder with automatic backend selection.
        
        Args:
            model_name: Model name/path (e.g., 'multi-qa-MiniLM-L6-cos-v1' or 'allenai/specter2_base')
            adapter: Adapter name for SPECTER2 models (e.g., 'allenai/specter2')
            device: Device to use ('cuda', 'cpu', or None for auto-detection)
            use_gpu: Whether to use GPU if available (ignored if device is specified)
        """
        self.model_name = model_name
        self.adapter = adapter
        self.is_specter2 = self._is_specter2_model(model_name)
        
        # Determine device
        if device is not None:
            self.device = device
        elif use_gpu:
            if torch.cuda.is_available():
                self.device = 'cuda'
            elif torch.backends.mps.is_available():
                self.device = 'mps'
            else:
                self.device = 'cpu'
        else:
            self.device = 'cpu'
        
        # Load model
        if self.is_specter2:
            self._load_specter2_model()
        else:
            self._load_sentence_transformer()
    
    def _is_specter2_model(self, model_name: str) -> bool:
        """Check if model is a SPECTER2 variant."""
        specter2_patterns = [
            'specter2',
            'specter-2',
        ]
        model_lower = model_name.lower()
        return any(pattern in model_lower for pattern in specter2_patterns)
    
    def _load_sentence_transformer(self):
        """Load model using sentence-transformers library."""
        from sentence_transformers import SentenceTransformer
        
        self.backend = 'sentence-transformers'
        self.model = SentenceTransformer(self.model_name, device=self.device)
        self.tokenizer = None  # Not needed for sentence-transformers
        
        print(f"Loaded {self.model_name} using sentence-transformers (device: {self.device})")
    
    def _load_specter2_model(self):
        """Load SPECTER2 model using transformers + adapters library."""
        try:
            from transformers import AutoTokenizer
            from adapters import AutoAdapterModel
        except ImportError as e:
            raise ImportError(
                "SPECTER2 models require the 'adapters' library. "
                "Install it with: pip install adapters"
            ) from e
        
        self.backend = 'adapters'
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoAdapterModel.from_pretrained(self.model_name)
        
        # Load adapter if specified
        if self.adapter:
            print(f"Loading adapter: {self.adapter}")
            self.model.load_adapter(self.adapter, source="hf", set_active=True)
        else:
            # Default to proximity adapter for SPECTER2
            default_adapter = "allenai/specter2"
            print(f"No adapter specified, using default: {default_adapter}")
            self.model.load_adapter(default_adapter, source="hf", set_active=True)
        
        # Move model to device AFTER loading adapter (crucial for proper device placement)
        self.model = self.model.to(self.device)
        self.model.eval()
        
        print(f"Loaded {self.model_name} using adapters (device: {self.device})")
    
    def encode(self,
               texts: Union[str, List[str]],
               batch_size: int = 32,
               show_progress_bar: bool = True,
               convert_to_numpy: bool = True,
               normalize_embeddings: bool = False) -> np.ndarray:
        """
        Encode texts into embeddings.
        
        Args:
            texts: Single text or list of texts to encode
            batch_size: Batch size for encoding
            show_progress_bar: Whether to show progress bar
            convert_to_numpy: Whether to convert to numpy array
            normalize_embeddings: Whether to normalize embeddings to unit length
            
        Returns:
            Embeddings as numpy array or torch tensor
        """
        # Ensure texts is a list
        if isinstance(texts, str):
            texts = [texts]
        
        if self.backend == 'sentence-transformers':
            embeddings = self._encode_sentence_transformer(
                texts,
                batch_size=batch_size,
                show_progress_bar=show_progress_bar,
                convert_to_numpy=convert_to_numpy,
                normalize_embeddings=normalize_embeddings
            )
        else:  # adapters backend
            embeddings = self._encode_specter2(
                texts,
                batch_size=batch_size,
                show_progress_bar=show_progress_bar,
                convert_to_numpy=convert_to_numpy,
                normalize_embeddings=normalize_embeddings
            )
        
        return embeddings
    
    def _encode_sentence_transformer(self,
                                    texts: List[str],
                                    batch_size: int,
                                    show_progress_bar: bool,
                                    convert_to_numpy: bool,
                                    normalize_embeddings: bool) -> np.ndarray:
        """Encode using sentence-transformers backend."""
        return self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            convert_to_numpy=convert_to_numpy,
            normalize_embeddings=normalize_embeddings
        )
    
    def _encode_specter2(self,
                        texts: List[str],
                        batch_size: int,
                        show_progress_bar: bool,
                        convert_to_numpy: bool,
                        normalize_embeddings: bool) -> np.ndarray:
        """Encode using adapters backend for SPECTER2."""
        from tqdm import tqdm
        
        all_embeddings = []
        
        # Process in batches
        iterator = range(0, len(texts), batch_size)
        if show_progress_bar:
            iterator = tqdm(iterator, desc="Encoding", total=len(texts) // batch_size + 1)
        
        with torch.no_grad():
            for i in iterator:
                batch_texts = texts[i:i + batch_size]
                
                # Tokenize
                inputs = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                    return_token_type_ids=False,
                    max_length=512
                )
                
                # Move to device
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Get embeddings
                outputs = self.model(**inputs)
                
                # Take first token (CLS token) as embedding
                embeddings = outputs.last_hidden_state[:, 0, :]
                
                # Normalize if requested
                if normalize_embeddings:
                    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
                
                all_embeddings.append(embeddings)
        
        # Concatenate all batches
        all_embeddings = torch.cat(all_embeddings, dim=0)
        
        # Convert to numpy if requested
        if convert_to_numpy:
            all_embeddings = all_embeddings.cpu().numpy()
        
        return all_embeddings
    
    def get_sentence_embedding_dimension(self) -> int:
        """Get the dimension of embeddings produced by this model."""
        if self.backend == 'sentence-transformers':
            return self.model.get_sentence_embedding_dimension()
        else:  # adapters backend
            # Get dimension from model config
            return self.model.config.hidden_size
    
    def cuda(self):
        """Move model to CUDA (for backward compatibility)."""
        if self.device != 'cuda':
            self.device = 'cuda'
            if self.backend == 'sentence-transformers':
                self.model = self.model.to('cuda')
            else:
                self.model = self.model.to('cuda')
        return self
