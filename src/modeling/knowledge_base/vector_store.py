"""
Vector Store Module
Provides vector storage and similarity search functionality.
"""

import torch
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Dict, Any, Union
import os
import pickle


class BaseVectorStore(ABC):
    """
    Abstract base class for vector stores.
    Provides interface for storing and searching embeddings.
    """
    
    def __init__(
        self,
        embedding_dim: int = 768,
        metric: str = 'cosine'
    ):
        """
        Initialize base vector store.
        
        Args:
            embedding_dim: Dimension of embeddings
            metric: Distance metric (cosine, l2, ip)
        """
        self.embedding_dim = embedding_dim
        self.metric = metric
        self.num_vectors = 0
    
    @abstractmethod
    def add(
        self,
        vectors: Union[torch.Tensor, np.ndarray],
        ids: Optional[List[str]] = None
    ) -> List[str]:
        """
        Add vectors to store.
        
        Args:
            vectors: Embedding vectors [num_vectors, embedding_dim]
            ids: Optional vector IDs
            
        Returns:
            List of vector IDs
        """
        pass
    
    @abstractmethod
    def search(
        self,
        query: Union[torch.Tensor, np.ndarray],
        top_k: int = 5
    ) -> Tuple[List[str], List[float]]:
        """
        Search for similar vectors.
        
        Args:
            query: Query vector [embedding_dim] or [num_queries, embedding_dim]
            top_k: Number of results to return
            
        Returns:
            Tuple of (ids, scores)
        """
        pass
    
    @abstractmethod
    def delete(self, ids: List[str]) -> bool:
        """
        Delete vectors by IDs.
        
        Args:
            ids: List of vector IDs to delete
            
        Returns:
            Success status
        """
        pass
    
    @abstractmethod
    def save(self, path: str):
        """Save index to disk."""
        pass
    
    @abstractmethod
    def load(self, path: str):
        """Load index from disk."""
        pass
    
    def _normalize(self, vectors: np.ndarray) -> np.ndarray:
        """
        Normalize vectors for cosine similarity.
        
        Args:
            vectors: Input vectors
            
        Returns:
            Normalized vectors
        """
        norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        return vectors / norms
    
    def _to_numpy(self, vectors: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
        """
        Convert to numpy array.
        
        Args:
            vectors: Input vectors
            
        Returns:
            Numpy array
        """
        if isinstance(vectors, torch.Tensor):
            return vectors.detach().cpu().numpy()
        return vectors


class InMemoryVectorStore(BaseVectorStore):
    """
    Simple in-memory vector store.
    Suitable for small to medium sized knowledge bases.
    """
    
    def __init__(
        self,
        embedding_dim: int = 768,
        metric: str = 'cosine'
    ):
        """
        Initialize in-memory vector store.
        
        Args:
            embedding_dim: Dimension of embeddings
            metric: Distance metric
        """
        super().__init__(embedding_dim, metric)
        
        self.vectors: List[np.ndarray] = []
        self.ids: List[str] = []
        self.id_to_idx: Dict[str, int] = {}
    
    def add(
        self,
        vectors: Union[torch.Tensor, np.ndarray],
        ids: Optional[List[str]] = None
    ) -> List[str]:
        """
        Add vectors to store.
        
        Args:
            vectors: Embedding vectors
            ids: Optional vector IDs
            
        Returns:
            List of vector IDs
        """
        vectors = self._to_numpy(vectors)
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        
        if self.metric == 'cosine':
            vectors = self._normalize(vectors)
        
        num_new = vectors.shape[0]
        if ids is None:
            ids = [f"vec_{self.num_vectors + i}" for i in range(num_new)]
        
        for i, (vec, vec_id) in enumerate(zip(vectors, ids)):
            idx = len(self.vectors)
            self.vectors.append(vec)
            self.ids.append(vec_id)
            self.id_to_idx[vec_id] = idx
        
        self.num_vectors += num_new
        return ids
    
    def search(
        self,
        query: Union[torch.Tensor, np.ndarray],
        top_k: int = 5
    ) -> Tuple[List[str], List[float]]:
        """
        Search for similar vectors using brute force.
        
        Args:
            query: Query vector
            top_k: Number of results
            
        Returns:
            Tuple of (ids, scores)
        """
        if len(self.vectors) == 0:
            return [], []
        
        query = self._to_numpy(query)
        if query.ndim == 1:
            query = query.reshape(1, -1)
        
        if self.metric == 'cosine':
            query = self._normalize(query)
        
        # Stack all vectors
        all_vectors = np.stack(self.vectors)  # [N, D]
        
        # Compute similarities
        if self.metric == 'cosine':
            scores = np.dot(query, all_vectors.T).squeeze()  # [N]
        elif self.metric == 'l2':
            scores = -np.linalg.norm(all_vectors - query, axis=-1)  # Negative L2
        else:  # Inner product
            scores = np.dot(query, all_vectors.T).squeeze()
        
        # Get top-k
        top_k = min(top_k, len(self.vectors))
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        top_ids = [self.ids[i] for i in top_indices]
        top_scores = [float(scores[i]) for i in top_indices]
        
        return top_ids, top_scores
    
    def delete(self, ids: List[str]) -> bool:
        """
        Delete vectors by IDs.
        Note: This is inefficient for in-memory store.
        
        Args:
            ids: Vector IDs to delete
            
        Returns:
            Success status
        """
        ids_set = set(ids)
        new_vectors = []
        new_ids = []
        
        for vec, vec_id in zip(self.vectors, self.ids):
            if vec_id not in ids_set:
                new_vectors.append(vec)
                new_ids.append(vec_id)
        
        self.vectors = new_vectors
        self.ids = new_ids
        self.id_to_idx = {vid: i for i, vid in enumerate(self.ids)}
        self.num_vectors = len(self.vectors)
        
        return True
    
    def save(self, path: str):
        """Save store to disk."""
        data = {
            'vectors': self.vectors,
            'ids': self.ids,
            'embedding_dim': self.embedding_dim,
            'metric': self.metric
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)
    
    def load(self, path: str):
        """Load store from disk."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        self.vectors = data['vectors']
        self.ids = data['ids']
        self.embedding_dim = data['embedding_dim']
        self.metric = data['metric']
        self.id_to_idx = {vid: i for i, vid in enumerate(self.ids)}
        self.num_vectors = len(self.vectors)


class FAISSVectorStore(BaseVectorStore):
    """
    FAISS-based vector store for efficient similarity search.
    Supports various index types for different use cases.
    """
    
    def __init__(
        self,
        embedding_dim: int = 768,
        metric: str = 'cosine',
        index_type: str = 'flat',
        num_partitions: int = 100,
        use_gpu: bool = False
    ):
        """
        Initialize FAISS vector store.
        
        Args:
            embedding_dim: Dimension of embeddings
            metric: Distance metric
            index_type: Index type (flat, ivf, hnsw)
            num_partitions: Number of IVF partitions
            use_gpu: Whether to use GPU
        """
        super().__init__(embedding_dim, metric)
        
        self.index_type = index_type
        self.num_partitions = num_partitions
        self.use_gpu = use_gpu
        self.index = None
        self.ids: List[str] = []
        self.is_trained = False
        
        self._create_index()
    
    def _create_index(self):
        """Create FAISS index based on configuration."""
        try:
            import faiss
        except ImportError:
            raise ImportError("faiss not installed. Install with: pip install faiss-cpu")
        
        # Determine metric type
        if self.metric == 'cosine' or self.metric == 'ip':
            metric_type = faiss.METRIC_INNER_PRODUCT
        else:
            metric_type = faiss.METRIC_L2
        
        # Create index based on type
        if self.index_type == 'flat':
            if self.metric == 'l2':
                self.index = faiss.IndexFlatL2(self.embedding_dim)
            else:
                self.index = faiss.IndexFlatIP(self.embedding_dim)
            self.is_trained = True
            
        elif self.index_type == 'ivf':
            quantizer = faiss.IndexFlatIP(self.embedding_dim)
            self.index = faiss.IndexIVFFlat(
                quantizer,
                self.embedding_dim,
                self.num_partitions,
                metric_type
            )
            
        elif self.index_type == 'hnsw':
            self.index = faiss.IndexHNSWFlat(self.embedding_dim, 32, metric_type)
            self.is_trained = True
        
        else:
            self.index = faiss.IndexFlatIP(self.embedding_dim)
            self.is_trained = True
        
        # Move to GPU if requested
        if self.use_gpu and faiss.get_num_gpus() > 0:
            self.index = faiss.index_cpu_to_gpu(
                faiss.StandardGpuResources(),
                0,
                self.index
            )
    
    def train(self, vectors: Union[torch.Tensor, np.ndarray]):
        """
        Train index (required for IVF).
        
        Args:
            vectors: Training vectors
        """
        if self.is_trained:
            return
        
        vectors = self._to_numpy(vectors).astype(np.float32)
        
        if self.metric == 'cosine':
            vectors = self._normalize(vectors)
        
        self.index.train(vectors)
        self.is_trained = True
    
    def add(
        self,
        vectors: Union[torch.Tensor, np.ndarray],
        ids: Optional[List[str]] = None
    ) -> List[str]:
        """
        Add vectors to index.
        
        Args:
            vectors: Embedding vectors
            ids: Optional vector IDs
            
        Returns:
            List of vector IDs
        """
        vectors = self._to_numpy(vectors).astype(np.float32)
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        
        if self.metric == 'cosine':
            vectors = self._normalize(vectors)
        
        # Train if needed
        if not self.is_trained:
            self.train(vectors)
        
        num_new = vectors.shape[0]
        if ids is None:
            ids = [f"vec_{self.num_vectors + i}" for i in range(num_new)]
        
        self.index.add(vectors)
        self.ids.extend(ids)
        self.num_vectors += num_new
        
        return ids
    
    def search(
        self,
        query: Union[torch.Tensor, np.ndarray],
        top_k: int = 5
    ) -> Tuple[List[str], List[float]]:
        """
        Search for similar vectors.
        
        Args:
            query: Query vector
            top_k: Number of results
            
        Returns:
            Tuple of (ids, scores)
        """
        if self.num_vectors == 0:
            return [], []
        
        query = self._to_numpy(query).astype(np.float32)
        if query.ndim == 1:
            query = query.reshape(1, -1)
        
        if self.metric == 'cosine':
            query = self._normalize(query)
        
        top_k = min(top_k, self.num_vectors)
        scores, indices = self.index.search(query, top_k)
        
        # Convert to lists
        scores = scores[0].tolist()
        indices = indices[0].tolist()
        
        # Map indices to IDs
        result_ids = []
        result_scores = []
        for idx, score in zip(indices, scores):
            if idx >= 0 and idx < len(self.ids):
                result_ids.append(self.ids[idx])
                result_scores.append(float(score))
        
        return result_ids, result_scores
    
    def delete(self, ids: List[str]) -> bool:
        """
        Delete vectors (rebuilds index).
        
        Args:
            ids: Vector IDs to delete
            
        Returns:
            Success status
        """
        # FAISS does not support deletion, need to rebuild
        import warnings
        warnings.warn("FAISS delete requires index rebuild")
        return False
    
    def save(self, path: str):
        """Save index and metadata."""
        try:
            import faiss
            
            # Save FAISS index
            index_path = path + '.faiss'
            faiss.write_index(self.index, index_path)
            
            # Save metadata
            meta_path = path + '.meta'
            meta = {
                'ids': self.ids,
                'embedding_dim': self.embedding_dim,
                'metric': self.metric,
                'index_type': self.index_type,
                'num_vectors': self.num_vectors
            }
            with open(meta_path, 'wb') as f:
                pickle.dump(meta, f)
                
        except Exception as e:
            raise IOError(f"Failed to save FAISS index: {e}")
    
    def load(self, path: str):
        """Load index and metadata."""
        try:
            import faiss
            
            # Load FAISS index
            index_path = path + '.faiss'
            self.index = faiss.read_index(index_path)
            
            # Load metadata
            meta_path = path + '.meta'
            with open(meta_path, 'rb') as f:
                meta = pickle.load(f)
            
            self.ids = meta['ids']
            self.embedding_dim = meta['embedding_dim']
            self.metric = meta['metric']
            self.index_type = meta['index_type']
            self.num_vectors = meta['num_vectors']
            self.is_trained = True
            
        except Exception as e:
            raise IOError(f"Failed to load FAISS index: {e}")


class ChromaVectorStore(BaseVectorStore):
    """
    ChromaDB-based vector store.
    Provides persistent vector storage with metadata filtering.
    """
    
    def __init__(
        self,
        embedding_dim: int = 768,
        metric: str = 'cosine',
        collection_name: str = 'knowledge_base',
        persist_directory: Optional[str] = None
    ):
        """
        Initialize Chroma vector store.
        
        Args:
            embedding_dim: Dimension of embeddings
            metric: Distance metric
            collection_name: Name of the collection
            persist_directory: Directory for persistence
        """
        super().__init__(embedding_dim, metric)
        
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        
        self._init_client()
    
    def _init_client(self):
        """Initialize ChromaDB client and collection."""
        try:
            import chromadb
            from chromadb.config import Settings
        except ImportError:
            raise ImportError("chromadb not installed. Install with: pip install chromadb")
        
        # Create client
        if self.persist_directory:
            self.client = chromadb.Client(Settings(
                chroma_db_impl="duckdb+parquet",
                persist_directory=self.persist_directory
            ))
        else:
            self.client = chromadb.Client()
        
        # Map metric
        distance_fn = {
            'cosine': 'cosine',
            'l2': 'l2',
            'ip': 'ip'
        }.get(self.metric, 'cosine')
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": distance_fn}
        )
    
    def add(
        self,
        vectors: Union[torch.Tensor, np.ndarray],
        ids: Optional[List[str]] = None,
        metadata: Optional[List[Dict]] = None
    ) -> List[str]:
        """
        Add vectors with optional metadata.
        
        Args:
            vectors: Embedding vectors
            ids: Optional vector IDs
            metadata: Optional metadata list
            
        Returns:
            List of vector IDs
        """
        vectors = self._to_numpy(vectors)
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        
        num_new = vectors.shape[0]
        if ids is None:
            ids = [f"vec_{self.num_vectors + i}" for i in range(num_new)]
        
        if metadata is None:
            metadata = [{}] * num_new
        
        self.collection.add(
            embeddings=vectors.tolist(),
            ids=ids,
            metadatas=metadata
        )
        
        self.num_vectors += num_new
        return ids
    
    def search(
        self,
        query: Union[torch.Tensor, np.ndarray],
        top_k: int = 5,
        where: Optional[Dict] = None
    ) -> Tuple[List[str], List[float]]:
        """
        Search with optional metadata filtering.
        
        Args:
            query: Query vector
            top_k: Number of results
            where: Optional metadata filter
            
        Returns:
            Tuple of (ids, scores)
        """
        query = self._to_numpy(query)
        if query.ndim == 1:
            query = query.reshape(1, -1)
        
        results = self.collection.query(
            query_embeddings=query.tolist(),
            n_results=top_k,
            where=where
        )
        
        ids = results['ids'][0] if results['ids'] else []
        distances = results['distances'][0] if results['distances'] else []
        
        # Convert distances to scores (higher is better)
        if self.metric == 'cosine':
            scores = [1 - d for d in distances]
        elif self.metric == 'l2':
            scores = [-d for d in distances]
        else:
            scores = distances
        
        return ids, scores
    
    def delete(self, ids: List[str]) -> bool:
        """
        Delete vectors by IDs.
        
        Args:
            ids: Vector IDs to delete
            
        Returns:
            Success status
        """
        try:
            self.collection.delete(ids=ids)
            self.num_vectors -= len(ids)
            return True
        except Exception:
            return False
    
    def save(self, path: str):
        """Save (persist) the collection."""
        if self.persist_directory:
            self.client.persist()
    
    def load(self, path: str):
        """Load is automatic with persist directory."""
        pass


def create_vector_store(
    store_type: str,
    embedding_dim: int = 768,
    **kwargs
) -> BaseVectorStore:
    """
    Factory function to create vector store.
    
    Args:
        store_type: Type of vector store
        embedding_dim: Embedding dimension
        **kwargs: Additional arguments
        
    Returns:
        Vector store instance
    """
    stores = {
        'in_memory': InMemoryVectorStore,
        'faiss': FAISSVectorStore,
        'chroma': ChromaVectorStore
    }
    
    if store_type not in stores:
        raise ValueError(f"Unknown store type: {store_type}. Available: {list(stores.keys())}")
    
    return stores[store_type](embedding_dim=embedding_dim, **kwargs)
