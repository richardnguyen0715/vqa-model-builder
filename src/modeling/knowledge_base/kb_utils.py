"""
Knowledge Base Utility Functions
Provides helper functions for knowledge base operations.
"""

import json
import os
import hashlib
import pickle
from typing import List, Dict, Any, Optional, Tuple, Union
from pathlib import Path
from datetime import datetime
import logging

import numpy as np


logger = logging.getLogger(__name__)


def generate_document_id(content: str, prefix: str = 'doc') -> str:
    """
    Generate unique document ID from content hash.
    
    Args:
        content: Document content
        prefix: ID prefix
        
    Returns:
        Unique document ID
    """
    content_hash = hashlib.md5(content.encode()).hexdigest()[:12]
    return f"{prefix}_{content_hash}"


def chunk_documents(
    documents: List[str],
    chunk_size: int = 256,
    overlap: int = 32,
    tokenizer: Optional[Any] = None
) -> List[Dict[str, Any]]:
    """
    Chunk documents into smaller segments.
    
    Args:
        documents: List of document strings
        chunk_size: Maximum chunk size
        overlap: Overlap between chunks
        tokenizer: Optional tokenizer for word-level chunking
        
    Returns:
        List of chunk dictionaries with content and metadata
    """
    chunks = []
    
    for doc_idx, doc in enumerate(documents):
        doc_id = generate_document_id(doc)
        
        if tokenizer:
            tokens = tokenizer.tokenize(doc)
        else:
            tokens = doc.split()
        
        if len(tokens) <= chunk_size:
            chunks.append({
                'content': doc,
                'doc_id': doc_id,
                'chunk_idx': 0,
                'start_token': 0,
                'end_token': len(tokens)
            })
            continue
        
        start = 0
        chunk_idx = 0
        
        while start < len(tokens):
            end = min(start + chunk_size, len(tokens))
            chunk_tokens = tokens[start:end]
            
            chunk_content = ' '.join(chunk_tokens)
            
            chunks.append({
                'content': chunk_content,
                'doc_id': doc_id,
                'chunk_idx': chunk_idx,
                'start_token': start,
                'end_token': end
            })
            
            start = end - overlap
            chunk_idx += 1
    
    return chunks


def batch_encode_documents(
    documents: List[str],
    encoder: Any,
    batch_size: int = 32,
    show_progress: bool = True
) -> np.ndarray:
    """
    Encode documents in batches.
    
    Args:
        documents: List of documents
        encoder: Encoder model
        batch_size: Batch size
        show_progress: Whether to show progress
        
    Returns:
        Document embeddings array
    """
    embeddings = []
    num_batches = (len(documents) + batch_size - 1) // batch_size
    
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i + batch_size]
        
        if show_progress:
            batch_num = i // batch_size + 1
            logger.info(f"Encoding batch {batch_num}/{num_batches}")
        
        result = encoder.encode_batch(batch)
        batch_embeddings = result.pooled_output
        
        if hasattr(batch_embeddings, 'cpu'):
            batch_embeddings = batch_embeddings.cpu().numpy()
        
        embeddings.append(batch_embeddings)
    
    return np.vstack(embeddings)


def save_knowledge_base(
    documents: List[str],
    embeddings: np.ndarray,
    metadata: Optional[List[Dict]] = None,
    save_path: str = 'knowledge_base.pkl'
):
    """
    Save knowledge base to disk.
    
    Args:
        documents: List of documents
        embeddings: Document embeddings
        metadata: Document metadata
        save_path: Save path
    """
    kb_data = {
        'documents': documents,
        'embeddings': embeddings,
        'metadata': metadata,
        'created_at': datetime.now().isoformat()
    }
    
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(save_path, 'wb') as f:
        pickle.dump(kb_data, f)
    
    logger.info(f"Knowledge base saved to {save_path}")


def load_knowledge_base(
    load_path: str
) -> Tuple[List[str], np.ndarray, Optional[List[Dict]]]:
    """
    Load knowledge base from disk.
    
    Args:
        load_path: Load path
        
    Returns:
        Tuple of (documents, embeddings, metadata)
    """
    with open(load_path, 'rb') as f:
        kb_data = pickle.load(f)
    
    logger.info(f"Knowledge base loaded from {load_path}")
    
    return (
        kb_data['documents'],
        kb_data['embeddings'],
        kb_data.get('metadata')
    )


def export_to_json(
    documents: List[str],
    metadata: Optional[List[Dict]] = None,
    output_path: str = 'knowledge_base.json'
):
    """
    Export knowledge base documents to JSON.
    
    Args:
        documents: List of documents
        metadata: Document metadata
        output_path: Output path
    """
    data = []
    
    for i, doc in enumerate(documents):
        entry = {
            'id': generate_document_id(doc),
            'content': doc
        }
        
        if metadata and i < len(metadata):
            entry['metadata'] = metadata[i]
        
        data.append(entry)
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Knowledge base exported to {output_path}")


def import_from_json(
    input_path: str
) -> Tuple[List[str], List[str], List[Dict]]:
    """
    Import knowledge base documents from JSON.
    
    Args:
        input_path: Input path
        
    Returns:
        Tuple of (documents, ids, metadata)
    """
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    documents = []
    ids = []
    metadata = []
    
    for entry in data:
        documents.append(entry['content'])
        ids.append(entry.get('id', generate_document_id(entry['content'])))
        metadata.append(entry.get('metadata', {}))
    
    logger.info(f"Loaded {len(documents)} documents from {input_path}")
    
    return documents, ids, metadata


def compute_embedding_statistics(
    embeddings: np.ndarray
) -> Dict[str, float]:
    """
    Compute statistics for embeddings.
    
    Args:
        embeddings: Embedding array
        
    Returns:
        Statistics dictionary
    """
    return {
        'num_embeddings': len(embeddings),
        'embedding_dim': embeddings.shape[1],
        'mean_norm': float(np.linalg.norm(embeddings, axis=1).mean()),
        'std_norm': float(np.linalg.norm(embeddings, axis=1).std()),
        'mean_value': float(embeddings.mean()),
        'std_value': float(embeddings.std()),
        'min_value': float(embeddings.min()),
        'max_value': float(embeddings.max())
    }


def compute_similarity_matrix(
    embeddings: np.ndarray,
    normalize: bool = True
) -> np.ndarray:
    """
    Compute pairwise similarity matrix.
    
    Args:
        embeddings: Embedding array [N, D]
        normalize: Whether to normalize embeddings
        
    Returns:
        Similarity matrix [N, N]
    """
    if normalize:
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / (norms + 1e-10)
    
    return np.dot(embeddings, embeddings.T)


def find_duplicates(
    documents: List[str],
    embeddings: np.ndarray,
    similarity_threshold: float = 0.95
) -> List[Tuple[int, int, float]]:
    """
    Find duplicate or near-duplicate documents.
    
    Args:
        documents: List of documents
        embeddings: Document embeddings
        similarity_threshold: Similarity threshold for duplicates
        
    Returns:
        List of (idx1, idx2, similarity) tuples
    """
    sim_matrix = compute_similarity_matrix(embeddings)
    
    duplicates = []
    n = len(documents)
    
    for i in range(n):
        for j in range(i + 1, n):
            if sim_matrix[i, j] >= similarity_threshold:
                duplicates.append((i, j, float(sim_matrix[i, j])))
    
    return duplicates


def deduplicate_documents(
    documents: List[str],
    embeddings: np.ndarray,
    similarity_threshold: float = 0.95
) -> Tuple[List[str], np.ndarray, List[int]]:
    """
    Remove duplicate documents.
    
    Args:
        documents: List of documents
        embeddings: Document embeddings
        similarity_threshold: Similarity threshold
        
    Returns:
        Tuple of (unique_documents, unique_embeddings, kept_indices)
    """
    duplicates = find_duplicates(documents, embeddings, similarity_threshold)
    
    # Find indices to remove (keep first occurrence)
    to_remove = set()
    for i, j, _ in duplicates:
        to_remove.add(j)
    
    # Filter
    kept_indices = [i for i in range(len(documents)) if i not in to_remove]
    unique_documents = [documents[i] for i in kept_indices]
    unique_embeddings = embeddings[kept_indices]
    
    logger.info(f"Removed {len(to_remove)} duplicates, kept {len(kept_indices)} documents")
    
    return unique_documents, unique_embeddings, kept_indices


def cluster_documents(
    embeddings: np.ndarray,
    n_clusters: int = 10,
    method: str = 'kmeans'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Cluster documents by embedding similarity.
    
    Args:
        embeddings: Document embeddings
        n_clusters: Number of clusters
        method: Clustering method
        
    Returns:
        Tuple of (cluster_labels, cluster_centers)
    """
    try:
        from sklearn.cluster import KMeans, AgglomerativeClustering
    except ImportError:
        raise ImportError("scikit-learn required for clustering")
    
    if method == 'kmeans':
        clusterer = KMeans(n_clusters=n_clusters, random_state=42)
        labels = clusterer.fit_predict(embeddings)
        centers = clusterer.cluster_centers_
        
    elif method == 'hierarchical':
        clusterer = AgglomerativeClustering(n_clusters=n_clusters)
        labels = clusterer.fit_predict(embeddings)
        
        # Compute cluster centers
        centers = np.zeros((n_clusters, embeddings.shape[1]))
        for i in range(n_clusters):
            mask = labels == i
            if mask.sum() > 0:
                centers[i] = embeddings[mask].mean(axis=0)
    else:
        raise ValueError(f"Unknown clustering method: {method}")
    
    return labels, centers


def retrieve_diverse(
    query_embedding: np.ndarray,
    document_embeddings: np.ndarray,
    documents: List[str],
    k: int = 5,
    diversity_weight: float = 0.3
) -> List[Tuple[int, str, float]]:
    """
    Retrieve diverse results using MMR (Maximum Marginal Relevance).
    
    Args:
        query_embedding: Query embedding [D]
        document_embeddings: Document embeddings [N, D]
        documents: List of documents
        k: Number of results
        diversity_weight: Weight for diversity (lambda)
        
    Returns:
        List of (index, document, score) tuples
    """
    # Normalize
    query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-10)
    doc_norms = document_embeddings / (np.linalg.norm(document_embeddings, axis=1, keepdims=True) + 1e-10)
    
    # Compute query-document similarities
    query_sim = np.dot(doc_norms, query_norm)
    
    selected = []
    selected_indices = []
    
    for _ in range(k):
        if len(selected_indices) == 0:
            # First selection: most similar to query
            idx = np.argmax(query_sim)
        else:
            # MMR selection
            mmr_scores = []
            
            for i in range(len(documents)):
                if i in selected_indices:
                    mmr_scores.append(-np.inf)
                    continue
                
                # Relevance to query
                relevance = query_sim[i]
                
                # Max similarity to already selected
                selected_embs = doc_norms[selected_indices]
                diversity = np.max(np.dot(selected_embs, doc_norms[i]))
                
                # MMR score
                mmr = (1 - diversity_weight) * relevance - diversity_weight * diversity
                mmr_scores.append(mmr)
            
            idx = np.argmax(mmr_scores)
        
        selected.append((idx, documents[idx], float(query_sim[idx])))
        selected_indices.append(idx)
    
    return selected


def format_context_for_prompt(
    contexts: List[str],
    max_length: int = 1024,
    separator: str = '\n---\n'
) -> str:
    """
    Format retrieved contexts for prompt injection.
    
    Args:
        contexts: List of context strings
        max_length: Maximum total length
        separator: Separator between contexts
        
    Returns:
        Formatted context string
    """
    if not contexts:
        return ""
    
    formatted_parts = []
    current_length = 0
    
    for i, ctx in enumerate(contexts):
        prefix = f"[Context {i+1}]: "
        ctx_text = prefix + ctx
        
        if current_length + len(ctx_text) + len(separator) > max_length:
            # Truncate this context
            remaining = max_length - current_length - len(separator)
            if remaining > len(prefix) + 50:
                ctx_text = prefix + ctx[:remaining - len(prefix) - 3] + "..."
                formatted_parts.append(ctx_text)
            break
        
        formatted_parts.append(ctx_text)
        current_length += len(ctx_text) + len(separator)
    
    return separator.join(formatted_parts)


def create_knowledge_base_index(
    documents: List[str],
    encoder: Any,
    vector_store_type: str = 'faiss',
    batch_size: int = 32,
    save_path: Optional[str] = None
) -> Any:
    """
    Create and optionally save a knowledge base index.
    
    Args:
        documents: List of documents
        encoder: Text encoder
        vector_store_type: Vector store type
        batch_size: Encoding batch size
        save_path: Optional save path
        
    Returns:
        Vector store instance
    """
    from .vector_store import FAISSVectorStore, InMemoryVectorStore, ChromaVectorStore
    
    # Encode documents
    logger.info(f"Encoding {len(documents)} documents...")
    embeddings = batch_encode_documents(documents, encoder, batch_size)
    
    # Generate IDs
    doc_ids = [generate_document_id(doc) for doc in documents]
    
    # Create vector store
    embed_dim = embeddings.shape[1]
    
    if vector_store_type == 'faiss':
        vector_store = FAISSVectorStore(
            embed_dim=embed_dim,
            index_type='flat'
        )
    elif vector_store_type == 'chroma':
        vector_store = ChromaVectorStore(
            collection_name='knowledge_base',
            embed_dim=embed_dim
        )
    else:
        vector_store = InMemoryVectorStore(embed_dim=embed_dim)
    
    # Add documents
    for i, (doc_id, doc, emb) in enumerate(zip(doc_ids, documents, embeddings)):
        vector_store.add(
            id=doc_id,
            embedding=emb,
            text=doc,
            metadata={'index': i}
        )
    
    # Save if requested
    if save_path:
        save_knowledge_base(documents, embeddings, None, save_path)
    
    logger.info(f"Created knowledge base with {len(documents)} documents")
    
    return vector_store


def evaluate_retrieval_quality(
    retriever: Any,
    queries: List[str],
    ground_truth: List[List[str]],
    k_values: List[int] = [1, 3, 5, 10]
) -> Dict[str, float]:
    """
    Evaluate retrieval quality using standard metrics.
    
    Args:
        retriever: Retriever instance
        queries: List of queries
        ground_truth: List of relevant document IDs for each query
        k_values: K values for metrics
        
    Returns:
        Dictionary of metrics
    """
    metrics = {f'recall@{k}': 0.0 for k in k_values}
    metrics.update({f'precision@{k}': 0.0 for k in k_values})
    metrics['mrr'] = 0.0
    
    for query, relevant_ids in zip(queries, ground_truth):
        max_k = max(k_values)
        result = retriever.retrieve(query, top_k=max_k)
        retrieved_ids = result.ids
        
        relevant_set = set(relevant_ids)
        
        # Recall and Precision at K
        for k in k_values:
            retrieved_at_k = set(retrieved_ids[:k])
            hits = len(retrieved_at_k & relevant_set)
            
            metrics[f'recall@{k}'] += hits / len(relevant_set) if relevant_set else 0
            metrics[f'precision@{k}'] += hits / k
        
        # MRR
        for rank, doc_id in enumerate(retrieved_ids, 1):
            if doc_id in relevant_set:
                metrics['mrr'] += 1.0 / rank
                break
    
    # Average
    n_queries = len(queries)
    for key in metrics:
        metrics[key] /= n_queries
    
    return metrics
