"""
Knowledge Base Configuration Module
Defines configuration classes for knowledge base components.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum


class VectorStoreType(Enum):
    """Supported vector store types."""
    FAISS = 'faiss'
    IN_MEMORY = 'in_memory'
    CHROMA = 'chroma'
    MILVUS = 'milvus'


class RetrieverType(Enum):
    """Supported retriever types."""
    DENSE = 'dense'
    SPARSE = 'sparse'
    HYBRID = 'hybrid'
    MULTIMODAL = 'multimodal'


class EncoderType(Enum):
    """Supported encoder types."""
    TEXT = 'text'
    VISUAL = 'visual'
    MULTIMODAL = 'multimodal'


@dataclass
class VectorStoreConfig:
    """
    Configuration for vector store.
    
    Attributes:
        store_type: Type of vector store to use
        embedding_dim: Dimension of embeddings
        index_type: Type of index for similarity search
        metric: Distance metric (cosine, l2, ip)
        use_gpu: Whether to use GPU for indexing
        num_partitions: Number of partitions for IVF index
        num_sub_quantizers: Number of sub-quantizers for PQ
        persist_directory: Directory to persist the index
    """
    store_type: str = 'faiss'
    embedding_dim: int = 768
    index_type: str = 'flat'  # flat, ivf, ivfpq, hnsw
    metric: str = 'cosine'
    use_gpu: bool = False
    num_partitions: int = 100
    num_sub_quantizers: int = 8
    persist_directory: Optional[str] = None
    
    def get_faiss_index_string(self) -> str:
        """
        Get FAISS index factory string.
        
        Returns:
            FAISS index factory string
        """
        if self.index_type == 'flat':
            return 'Flat'
        elif self.index_type == 'ivf':
            return f'IVF{self.num_partitions},Flat'
        elif self.index_type == 'ivfpq':
            return f'IVF{self.num_partitions},PQ{self.num_sub_quantizers}'
        elif self.index_type == 'hnsw':
            return 'HNSW32'
        else:
            return 'Flat'


@dataclass
class RetrieverConfig:
    """
    Configuration for retriever.
    
    Attributes:
        retriever_type: Type of retriever
        top_k: Number of documents to retrieve
        similarity_threshold: Minimum similarity score
        use_reranking: Whether to use reranking
        reranker_model: Model for reranking
        max_query_length: Maximum query length
        batch_size: Batch size for encoding
        use_cache: Whether to cache retrieval results
        sparse_weight: Weight for sparse retrieval in hybrid
        dense_weight: Weight for dense retrieval in hybrid
    """
    retriever_type: str = 'dense'
    top_k: int = 5
    similarity_threshold: float = 0.0
    use_reranking: bool = False
    reranker_model: Optional[str] = None
    max_query_length: int = 512
    batch_size: int = 32
    use_cache: bool = True
    sparse_weight: float = 0.3
    dense_weight: float = 0.7


@dataclass
class EncoderConfig:
    """
    Configuration for knowledge encoder.
    
    Attributes:
        encoder_type: Type of encoder
        model_name: Pretrained model name
        embedding_dim: Output embedding dimension
        max_length: Maximum input length
        pooling_strategy: Pooling strategy for sequences
        normalize_embeddings: Whether to normalize embeddings
        use_projection: Whether to use projection layer
        freeze_backbone: Whether to freeze backbone weights
    """
    encoder_type: str = 'text'
    model_name: str = 'vinai/phobert-base'
    embedding_dim: int = 768
    max_length: int = 256
    pooling_strategy: str = 'mean'  # cls, mean, max
    normalize_embeddings: bool = True
    use_projection: bool = False
    freeze_backbone: bool = True


@dataclass
class VietnameseProcessorConfig:
    """
    Configuration for Vietnamese text processor.
    
    Attributes:
        use_word_segmentation: Whether to use word segmentation
        segmenter_model: Word segmentation model
        use_accent_normalization: Whether to normalize accents
        lowercase: Whether to lowercase text
        remove_punctuation: Whether to remove punctuation
        remove_stopwords: Whether to remove stopwords
        stopwords_path: Path to stopwords file
        use_sentence_splitting: Whether to split into sentences
    """
    use_word_segmentation: bool = True
    segmenter_model: str = 'vinai/phobert-base'
    use_accent_normalization: bool = True
    lowercase: bool = True
    remove_punctuation: bool = False
    remove_stopwords: bool = False
    stopwords_path: Optional[str] = None
    use_sentence_splitting: bool = True


@dataclass
class RAGConfig:
    """
    Configuration for RAG module.
    
    Attributes:
        num_retrieved_docs: Number of documents to retrieve
        max_context_length: Maximum context length
        fusion_method: Method to fuse retrieved knowledge
        use_knowledge_attention: Whether to use knowledge attention
        knowledge_dropout: Dropout for knowledge integration
        temperature: Temperature for knowledge selection
        use_query_expansion: Whether to expand queries
        use_iterative_retrieval: Whether to use iterative retrieval
        num_iterations: Number of retrieval iterations
    """
    num_retrieved_docs: int = 5
    max_context_length: int = 1024
    fusion_method: str = 'attention'  # attention, concat, gate
    use_knowledge_attention: bool = True
    knowledge_dropout: float = 0.1
    temperature: float = 1.0
    use_query_expansion: bool = False
    use_iterative_retrieval: bool = False
    num_iterations: int = 2


@dataclass
class KnowledgeBaseConfig:
    """
    Main configuration for Knowledge Base.
    
    Attributes:
        vector_store_config: Vector store configuration
        retriever_config: Retriever configuration
        encoder_config: Encoder configuration
        vietnamese_config: Vietnamese processor configuration
        rag_config: RAG configuration
        knowledge_sources: List of knowledge source paths
        index_name: Name of the knowledge index
        auto_update: Whether to auto-update index
        max_documents: Maximum number of documents
    """
    vector_store_config: VectorStoreConfig = field(default_factory=VectorStoreConfig)
    retriever_config: RetrieverConfig = field(default_factory=RetrieverConfig)
    encoder_config: EncoderConfig = field(default_factory=EncoderConfig)
    vietnamese_config: VietnameseProcessorConfig = field(default_factory=VietnameseProcessorConfig)
    rag_config: RAGConfig = field(default_factory=RAGConfig)
    knowledge_sources: List[str] = field(default_factory=list)
    index_name: str = 'vqa_knowledge'
    auto_update: bool = False
    max_documents: int = 1000000
    
    def __post_init__(self):
        """Initialize nested configs if provided as dicts."""
        if isinstance(self.vector_store_config, dict):
            self.vector_store_config = VectorStoreConfig(**self.vector_store_config)
        if isinstance(self.retriever_config, dict):
            self.retriever_config = RetrieverConfig(**self.retriever_config)
        if isinstance(self.encoder_config, dict):
            self.encoder_config = EncoderConfig(**self.encoder_config)
        if isinstance(self.vietnamese_config, dict):
            self.vietnamese_config = VietnameseProcessorConfig(**self.vietnamese_config)
        if isinstance(self.rag_config, dict):
            self.rag_config = RAGConfig(**self.rag_config)


@dataclass 
class VQAKnowledgeConfig(KnowledgeBaseConfig):
    """
    VQA-specific knowledge base configuration.
    
    Attributes:
        use_visual_knowledge: Whether to include visual knowledge
        use_commonsense: Whether to use commonsense knowledge
        use_factual: Whether to use factual knowledge
        use_scene_graphs: Whether to use scene graph knowledge
        vietnamese_wikipedia_path: Path to Vietnamese Wikipedia
        vietnamese_commonsense_path: Path to Vietnamese commonsense KB
        image_caption_path: Path to image captions
        object_labels_path: Path to object label vocabulary
    """
    use_visual_knowledge: bool = True
    use_commonsense: bool = True
    use_factual: bool = True
    use_scene_graphs: bool = False
    vietnamese_wikipedia_path: Optional[str] = None
    vietnamese_commonsense_path: Optional[str] = None
    image_caption_path: Optional[str] = None
    object_labels_path: Optional[str] = None
