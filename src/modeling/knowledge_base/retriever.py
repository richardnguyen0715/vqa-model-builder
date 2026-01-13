"""
Retriever Module
Provides retrieval mechanisms for knowledge base queries.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import math

import torch
import torch.nn as nn


class RetrievalMethod(Enum):
    """Enumeration of retrieval methods."""
    DENSE = 'dense'
    SPARSE = 'sparse'
    HYBRID = 'hybrid'
    MULTIMODAL = 'multimodal'


@dataclass
class RetrievalResult:
    """
    Container for retrieval results.
    
    Attributes:
        documents: Retrieved document contents
        scores: Retrieval scores
        ids: Document IDs
        metadata: Document metadata
        embeddings: Document embeddings
    """
    documents: List[str]
    scores: List[float]
    ids: List[str]
    metadata: Optional[List[Dict[str, Any]]] = None
    embeddings: Optional[torch.Tensor] = None
    
    def __len__(self) -> int:
        """Return number of results."""
        return len(self.documents)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get single result by index."""
        result = {
            'document': self.documents[idx],
            'score': self.scores[idx],
            'id': self.ids[idx]
        }
        if self.metadata:
            result['metadata'] = self.metadata[idx]
        return result


class BaseRetriever(ABC):
    """
    Abstract base class for retrievers.
    """
    
    def __init__(
        self,
        top_k: int = 5,
        score_threshold: Optional[float] = None
    ):
        """
        Initialize base retriever.
        
        Args:
            top_k: Number of documents to retrieve
            score_threshold: Minimum score threshold
        """
        self.top_k = top_k
        self.score_threshold = score_threshold
    
    @abstractmethod
    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        **kwargs
    ) -> RetrievalResult:
        """
        Retrieve documents for query.
        
        Args:
            query: Query string
            top_k: Number of documents to retrieve
            **kwargs: Additional arguments
            
        Returns:
            Retrieval result
        """
        pass
    
    @abstractmethod
    def retrieve_batch(
        self,
        queries: List[str],
        top_k: Optional[int] = None,
        **kwargs
    ) -> List[RetrievalResult]:
        """
        Retrieve documents for batch of queries.
        
        Args:
            queries: List of query strings
            top_k: Number of documents to retrieve
            **kwargs: Additional arguments
            
        Returns:
            List of retrieval results
        """
        pass
    
    def _filter_by_threshold(
        self,
        result: RetrievalResult
    ) -> RetrievalResult:
        """
        Filter results by score threshold.
        
        Args:
            result: Retrieval result
            
        Returns:
            Filtered result
        """
        if self.score_threshold is None:
            return result
        
        filtered_docs = []
        filtered_scores = []
        filtered_ids = []
        filtered_metadata = []
        
        for i, score in enumerate(result.scores):
            if score >= self.score_threshold:
                filtered_docs.append(result.documents[i])
                filtered_scores.append(score)
                filtered_ids.append(result.ids[i])
                if result.metadata:
                    filtered_metadata.append(result.metadata[i])
        
        return RetrievalResult(
            documents=filtered_docs,
            scores=filtered_scores,
            ids=filtered_ids,
            metadata=filtered_metadata if filtered_metadata else None
        )


class DenseRetriever(BaseRetriever):
    """
    Dense retriever using vector similarity search.
    """
    
    def __init__(
        self,
        encoder: Any,
        vector_store: Any,
        top_k: int = 5,
        score_threshold: Optional[float] = None,
        normalize_scores: bool = True
    ):
        """
        Initialize dense retriever.
        
        Args:
            encoder: Text encoder for queries
            vector_store: Vector store for document embeddings
            top_k: Number of documents to retrieve
            score_threshold: Minimum score threshold
            normalize_scores: Whether to normalize scores
        """
        super().__init__(top_k, score_threshold)
        
        self.encoder = encoder
        self.vector_store = vector_store
        self.normalize_scores = normalize_scores
    
    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        **kwargs
    ) -> RetrievalResult:
        """
        Retrieve documents using dense vectors.
        
        Args:
            query: Query string
            top_k: Number of documents to retrieve
            **kwargs: Additional arguments
            
        Returns:
            Retrieval result
        """
        k = top_k or self.top_k
        
        # Encode query
        encoding_result = self.encoder.encode(query)
        query_embedding = encoding_result.pooled_output
        
        # Convert to numpy for vector store
        if isinstance(query_embedding, torch.Tensor):
            query_embedding = query_embedding.cpu().numpy()
        
        if query_embedding.ndim == 2:
            query_embedding = query_embedding[0]
        
        # Search vector store
        search_results = self.vector_store.search(query_embedding, k=k)
        
        # Normalize scores if needed
        if self.normalize_scores and search_results:
            max_score = max(r['score'] for r in search_results)
            min_score = min(r['score'] for r in search_results)
            score_range = max_score - min_score
            
            if score_range > 0:
                for r in search_results:
                    r['score'] = (r['score'] - min_score) / score_range
        
        # Build result
        result = RetrievalResult(
            documents=[r.get('document', r.get('text', '')) for r in search_results],
            scores=[r['score'] for r in search_results],
            ids=[r['id'] for r in search_results],
            metadata=[r.get('metadata', {}) for r in search_results]
        )
        
        return self._filter_by_threshold(result)
    
    def retrieve_batch(
        self,
        queries: List[str],
        top_k: Optional[int] = None,
        **kwargs
    ) -> List[RetrievalResult]:
        """
        Retrieve documents for batch of queries.
        
        Args:
            queries: List of query strings
            top_k: Number of documents to retrieve
            **kwargs: Additional arguments
            
        Returns:
            List of retrieval results
        """
        return [self.retrieve(q, top_k, **kwargs) for q in queries]


class SparseRetriever(BaseRetriever):
    """
    Sparse retriever using BM25 or TF-IDF.
    """
    
    def __init__(
        self,
        documents: Optional[List[str]] = None,
        document_ids: Optional[List[str]] = None,
        method: str = 'bm25',
        top_k: int = 5,
        score_threshold: Optional[float] = None,
        k1: float = 1.5,
        b: float = 0.75
    ):
        """
        Initialize sparse retriever.
        
        Args:
            documents: List of documents
            document_ids: Document IDs
            method: Retrieval method (bm25, tfidf)
            top_k: Number of documents to retrieve
            score_threshold: Minimum score threshold
            k1: BM25 k1 parameter
            b: BM25 b parameter
        """
        super().__init__(top_k, score_threshold)
        
        self.method = method
        self.k1 = k1
        self.b = b
        
        self.documents = documents or []
        self.document_ids = document_ids or []
        
        # Build index
        self.inverted_index: Dict[str, List[Tuple[int, int]]] = {}
        self.doc_lengths: List[int] = []
        self.avg_doc_length: float = 0.0
        
        if documents:
            self._build_index()
    
    def _tokenize(self, text: str) -> List[str]:
        """
        Simple tokenization.
        
        Args:
            text: Input text
            
        Returns:
            List of tokens
        """
        return text.lower().split()
    
    def _build_index(self):
        """Build inverted index."""
        self.inverted_index = {}
        self.doc_lengths = []
        
        for doc_idx, doc in enumerate(self.documents):
            tokens = self._tokenize(doc)
            self.doc_lengths.append(len(tokens))
            
            # Count term frequencies
            term_freq: Dict[str, int] = {}
            for token in tokens:
                term_freq[token] = term_freq.get(token, 0) + 1
            
            # Add to inverted index
            for term, freq in term_freq.items():
                if term not in self.inverted_index:
                    self.inverted_index[term] = []
                self.inverted_index[term].append((doc_idx, freq))
        
        self.avg_doc_length = sum(self.doc_lengths) / len(self.doc_lengths) if self.doc_lengths else 0
    
    def add_documents(
        self,
        documents: List[str],
        document_ids: Optional[List[str]] = None
    ):
        """
        Add documents to the index.
        
        Args:
            documents: List of documents
            document_ids: Document IDs
        """
        start_idx = len(self.documents)
        
        if document_ids is None:
            document_ids = [f"doc_{start_idx + i}" for i in range(len(documents))]
        
        self.documents.extend(documents)
        self.document_ids.extend(document_ids)
        
        # Rebuild index
        self._build_index()
    
    def _bm25_score(
        self,
        query_tokens: List[str],
        doc_idx: int
    ) -> float:
        """
        Calculate BM25 score.
        
        Args:
            query_tokens: Query tokens
            doc_idx: Document index
            
        Returns:
            BM25 score
        """
        score = 0.0
        doc_length = self.doc_lengths[doc_idx]
        n_docs = len(self.documents)
        
        for term in query_tokens:
            if term not in self.inverted_index:
                continue
            
            postings = self.inverted_index[term]
            df = len(postings)  # Document frequency
            
            # Find term frequency in this document
            tf = 0
            for d_idx, freq in postings:
                if d_idx == doc_idx:
                    tf = freq
                    break
            
            if tf == 0:
                continue
            
            # IDF component
            idf = math.log((n_docs - df + 0.5) / (df + 0.5) + 1)
            
            # TF component with length normalization
            tf_component = (tf * (self.k1 + 1)) / (
                tf + self.k1 * (1 - self.b + self.b * doc_length / self.avg_doc_length)
            )
            
            score += idf * tf_component
        
        return score
    
    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        **kwargs
    ) -> RetrievalResult:
        """
        Retrieve documents using sparse matching.
        
        Args:
            query: Query string
            top_k: Number of documents to retrieve
            **kwargs: Additional arguments
            
        Returns:
            Retrieval result
        """
        k = top_k or self.top_k
        query_tokens = self._tokenize(query)
        
        # Get candidate documents
        candidate_docs = set()
        for token in query_tokens:
            if token in self.inverted_index:
                for doc_idx, _ in self.inverted_index[token]:
                    candidate_docs.add(doc_idx)
        
        # Score candidates
        scores = []
        for doc_idx in candidate_docs:
            score = self._bm25_score(query_tokens, doc_idx)
            scores.append((doc_idx, score))
        
        # Sort by score
        scores.sort(key=lambda x: x[1], reverse=True)
        scores = scores[:k]
        
        # Build result
        result = RetrievalResult(
            documents=[self.documents[idx] for idx, _ in scores],
            scores=[score for _, score in scores],
            ids=[self.document_ids[idx] for idx, _ in scores]
        )
        
        return self._filter_by_threshold(result)
    
    def retrieve_batch(
        self,
        queries: List[str],
        top_k: Optional[int] = None,
        **kwargs
    ) -> List[RetrievalResult]:
        """
        Retrieve documents for batch of queries.
        
        Args:
            queries: List of query strings
            top_k: Number of documents to retrieve
            **kwargs: Additional arguments
            
        Returns:
            List of retrieval results
        """
        return [self.retrieve(q, top_k, **kwargs) for q in queries]


class HybridRetriever(BaseRetriever):
    """
    Hybrid retriever combining dense and sparse retrieval.
    """
    
    def __init__(
        self,
        dense_retriever: DenseRetriever,
        sparse_retriever: SparseRetriever,
        top_k: int = 5,
        score_threshold: Optional[float] = None,
        dense_weight: float = 0.5,
        fusion_method: str = 'rrf'
    ):
        """
        Initialize hybrid retriever.
        
        Args:
            dense_retriever: Dense retriever
            sparse_retriever: Sparse retriever
            top_k: Number of documents to retrieve
            score_threshold: Minimum score threshold
            dense_weight: Weight for dense scores (0-1)
            fusion_method: Fusion method (rrf, linear, max)
        """
        super().__init__(top_k, score_threshold)
        
        self.dense_retriever = dense_retriever
        self.sparse_retriever = sparse_retriever
        self.dense_weight = dense_weight
        self.sparse_weight = 1 - dense_weight
        self.fusion_method = fusion_method
    
    def _reciprocal_rank_fusion(
        self,
        dense_result: RetrievalResult,
        sparse_result: RetrievalResult,
        k: int = 60
    ) -> Dict[str, float]:
        """
        Combine results using Reciprocal Rank Fusion.
        
        Args:
            dense_result: Dense retrieval result
            sparse_result: Sparse retrieval result
            k: RRF constant
            
        Returns:
            Dictionary of document IDs to fused scores
        """
        fused_scores: Dict[str, float] = {}
        
        # Add dense results
        for rank, doc_id in enumerate(dense_result.ids):
            score = self.dense_weight / (k + rank + 1)
            fused_scores[doc_id] = fused_scores.get(doc_id, 0) + score
        
        # Add sparse results
        for rank, doc_id in enumerate(sparse_result.ids):
            score = self.sparse_weight / (k + rank + 1)
            fused_scores[doc_id] = fused_scores.get(doc_id, 0) + score
        
        return fused_scores
    
    def _linear_fusion(
        self,
        dense_result: RetrievalResult,
        sparse_result: RetrievalResult
    ) -> Dict[str, float]:
        """
        Combine results using linear combination.
        
        Args:
            dense_result: Dense retrieval result
            sparse_result: Sparse retrieval result
            
        Returns:
            Dictionary of document IDs to fused scores
        """
        fused_scores: Dict[str, float] = {}
        
        # Normalize scores
        dense_max = max(dense_result.scores) if dense_result.scores else 1
        sparse_max = max(sparse_result.scores) if sparse_result.scores else 1
        
        # Add dense results
        for doc_id, score in zip(dense_result.ids, dense_result.scores):
            normalized = score / dense_max if dense_max > 0 else 0
            fused_scores[doc_id] = fused_scores.get(doc_id, 0) + self.dense_weight * normalized
        
        # Add sparse results
        for doc_id, score in zip(sparse_result.ids, sparse_result.scores):
            normalized = score / sparse_max if sparse_max > 0 else 0
            fused_scores[doc_id] = fused_scores.get(doc_id, 0) + self.sparse_weight * normalized
        
        return fused_scores
    
    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        **kwargs
    ) -> RetrievalResult:
        """
        Retrieve documents using hybrid approach.
        
        Args:
            query: Query string
            top_k: Number of documents to retrieve
            **kwargs: Additional arguments
            
        Returns:
            Retrieval result
        """
        k = top_k or self.top_k
        
        # Get results from both retrievers
        dense_result = self.dense_retriever.retrieve(query, top_k=k * 2)
        sparse_result = self.sparse_retriever.retrieve(query, top_k=k * 2)
        
        # Fuse results
        if self.fusion_method == 'rrf':
            fused_scores = self._reciprocal_rank_fusion(dense_result, sparse_result)
        else:
            fused_scores = self._linear_fusion(dense_result, sparse_result)
        
        # Sort by fused score
        sorted_results = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
        sorted_results = sorted_results[:k]
        
        # Build document lookup
        doc_lookup = {}
        for i, doc_id in enumerate(dense_result.ids):
            doc_lookup[doc_id] = {
                'document': dense_result.documents[i],
                'metadata': dense_result.metadata[i] if dense_result.metadata else {}
            }
        for i, doc_id in enumerate(sparse_result.ids):
            if doc_id not in doc_lookup:
                doc_lookup[doc_id] = {
                    'document': sparse_result.documents[i],
                    'metadata': sparse_result.metadata[i] if sparse_result.metadata else {}
                }
        
        # Build result
        result = RetrievalResult(
            documents=[doc_lookup[doc_id]['document'] for doc_id, _ in sorted_results],
            scores=[score for _, score in sorted_results],
            ids=[doc_id for doc_id, _ in sorted_results],
            metadata=[doc_lookup[doc_id]['metadata'] for doc_id, _ in sorted_results]
        )
        
        return self._filter_by_threshold(result)
    
    def retrieve_batch(
        self,
        queries: List[str],
        top_k: Optional[int] = None,
        **kwargs
    ) -> List[RetrievalResult]:
        """
        Retrieve documents for batch of queries.
        
        Args:
            queries: List of query strings
            top_k: Number of documents to retrieve
            **kwargs: Additional arguments
            
        Returns:
            List of retrieval results
        """
        return [self.retrieve(q, top_k, **kwargs) for q in queries]


class MultimodalRetriever(BaseRetriever):
    """
    Multimodal retriever for text and visual queries.
    """
    
    def __init__(
        self,
        multimodal_encoder: Any,
        vector_store: Any,
        top_k: int = 5,
        score_threshold: Optional[float] = None
    ):
        """
        Initialize multimodal retriever.
        
        Args:
            multimodal_encoder: Multimodal encoder
            vector_store: Vector store
            top_k: Number of documents to retrieve
            score_threshold: Minimum score threshold
        """
        super().__init__(top_k, score_threshold)
        
        self.encoder = multimodal_encoder
        self.vector_store = vector_store
    
    def retrieve(
        self,
        query: Optional[str] = None,
        image: Optional[Any] = None,
        top_k: Optional[int] = None,
        **kwargs
    ) -> RetrievalResult:
        """
        Retrieve documents using multimodal query.
        
        Args:
            query: Text query
            image: Image query
            top_k: Number of documents to retrieve
            **kwargs: Additional arguments
            
        Returns:
            Retrieval result
        """
        k = top_k or self.top_k
        
        # Encode query
        encoding_result = self.encoder.encode(text=query, image=image)
        query_embedding = encoding_result.pooled_output
        
        # Convert to numpy
        if isinstance(query_embedding, torch.Tensor):
            query_embedding = query_embedding.cpu().numpy()
        
        if query_embedding.ndim == 2:
            query_embedding = query_embedding[0]
        
        # Search
        search_results = self.vector_store.search(query_embedding, k=k)
        
        # Build result
        result = RetrievalResult(
            documents=[r.get('document', r.get('text', '')) for r in search_results],
            scores=[r['score'] for r in search_results],
            ids=[r['id'] for r in search_results],
            metadata=[r.get('metadata', {}) for r in search_results]
        )
        
        return self._filter_by_threshold(result)
    
    def retrieve_batch(
        self,
        queries: List[str],
        images: Optional[List[Any]] = None,
        top_k: Optional[int] = None,
        **kwargs
    ) -> List[RetrievalResult]:
        """
        Retrieve documents for batch of multimodal queries.
        
        Args:
            queries: List of text queries
            images: List of image queries
            top_k: Number of documents to retrieve
            **kwargs: Additional arguments
            
        Returns:
            List of retrieval results
        """
        results = []
        for i, query in enumerate(queries):
            image = images[i] if images and i < len(images) else None
            results.append(self.retrieve(query=query, image=image, top_k=top_k, **kwargs))
        return results


class RerankerRetriever(BaseRetriever):
    """
    Two-stage retriever with reranking.
    """
    
    def __init__(
        self,
        first_stage_retriever: BaseRetriever,
        reranker: Any,
        top_k: int = 5,
        first_stage_k: int = 50,
        score_threshold: Optional[float] = None
    ):
        """
        Initialize reranker retriever.
        
        Args:
            first_stage_retriever: First stage retriever
            reranker: Reranking model
            top_k: Final number of documents
            first_stage_k: First stage retrieval count
            score_threshold: Minimum score threshold
        """
        super().__init__(top_k, score_threshold)
        
        self.first_stage = first_stage_retriever
        self.reranker = reranker
        self.first_stage_k = first_stage_k
    
    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        **kwargs
    ) -> RetrievalResult:
        """
        Retrieve with reranking.
        
        Args:
            query: Query string
            top_k: Number of documents to retrieve
            **kwargs: Additional arguments
            
        Returns:
            Retrieval result
        """
        k = top_k or self.top_k
        
        # First stage retrieval
        first_stage_result = self.first_stage.retrieve(query, top_k=self.first_stage_k)
        
        if not first_stage_result.documents:
            return first_stage_result
        
        # Rerank
        pairs = [(query, doc) for doc in first_stage_result.documents]
        rerank_scores = self.reranker.predict(pairs)
        
        # Combine with indices
        indexed_scores = list(enumerate(rerank_scores))
        indexed_scores.sort(key=lambda x: x[1], reverse=True)
        indexed_scores = indexed_scores[:k]
        
        # Build result
        result = RetrievalResult(
            documents=[first_stage_result.documents[idx] for idx, _ in indexed_scores],
            scores=[score for _, score in indexed_scores],
            ids=[first_stage_result.ids[idx] for idx, _ in indexed_scores],
            metadata=[first_stage_result.metadata[idx] for idx, _ in indexed_scores] if first_stage_result.metadata else None
        )
        
        return self._filter_by_threshold(result)
    
    def retrieve_batch(
        self,
        queries: List[str],
        top_k: Optional[int] = None,
        **kwargs
    ) -> List[RetrievalResult]:
        """
        Retrieve with reranking for batch.
        
        Args:
            queries: List of queries
            top_k: Number of documents
            **kwargs: Additional arguments
            
        Returns:
            List of retrieval results
        """
        return [self.retrieve(q, top_k, **kwargs) for q in queries]


def create_dense_retriever(
    encoder: Any,
    vector_store: Any,
    **kwargs
) -> DenseRetriever:
    """
    Factory function to create dense retriever.
    
    Args:
        encoder: Text encoder
        vector_store: Vector store
        **kwargs: Additional arguments
        
    Returns:
        Dense retriever instance
    """
    return DenseRetriever(encoder, vector_store, **kwargs)


def create_hybrid_retriever(
    encoder: Any,
    vector_store: Any,
    documents: List[str],
    document_ids: Optional[List[str]] = None,
    **kwargs
) -> HybridRetriever:
    """
    Factory function to create hybrid retriever.
    
    Args:
        encoder: Text encoder
        vector_store: Vector store
        documents: List of documents
        document_ids: Document IDs
        **kwargs: Additional arguments
        
    Returns:
        Hybrid retriever instance
    """
    dense = DenseRetriever(encoder, vector_store)
    sparse = SparseRetriever(documents, document_ids)
    
    return HybridRetriever(dense, sparse, **kwargs)
