"""
Knowledge Base and RAG Module for Vietnamese VQA
Provides knowledge retrieval and augmentation for enhanced question answering.
"""

from src.modeling.knowledge_base.kb_config import (
    KnowledgeBaseConfig,
    VectorStoreConfig,
    RetrieverConfig,
    RAGConfig,
    VietnameseProcessorConfig,
    EncoderConfig,
    VQAKnowledgeConfig
)
from src.modeling.knowledge_base.vector_store import (
    BaseVectorStore,
    FAISSVectorStore,
    InMemoryVectorStore,
    ChromaVectorStore
)
from src.modeling.knowledge_base.document_store import (
    Document,
    DocumentStore,
    KnowledgeEntry,
    VisualKnowledgeEntry
)
from src.modeling.knowledge_base.knowledge_encoder import (
    BaseKnowledgeEncoder,
    TextKnowledgeEncoder,
    VisualKnowledgeEncoder,
    MultimodalKnowledgeEncoder,
    EncodingResult,
    create_text_encoder,
    create_visual_encoder,
    create_multimodal_encoder
)
from src.modeling.knowledge_base.retriever import (
    BaseRetriever,
    DenseRetriever,
    SparseRetriever,
    HybridRetriever,
    MultimodalRetriever,
    RerankerRetriever,
    RetrievalResult,
    RetrievalMethod,
    create_dense_retriever,
    create_hybrid_retriever
)
from src.modeling.knowledge_base.vietnamese_processor import (
    VietnameseTextProcessor,
    VietnameseTokenizer,
    VietnameseSentenceSplitter,
    ProcessedText,
    normalize_vietnamese_text,
    detect_vietnamese,
    convert_to_ascii_vietnamese,
    VIETNAMESE_STOPWORDS
)
from src.modeling.knowledge_base.rag_module import (
    RAGModule,
    RAGOutput,
    KnowledgeAugmentedVQA,
    KnowledgeAugmentedFusion,
    ContextEncoder,
    ContextAttention,
    RAGLoss,
    create_rag_module,
    create_knowledge_augmented_vqa
)
from src.modeling.knowledge_base.kb_utils import (
    generate_document_id,
    chunk_documents,
    batch_encode_documents,
    save_knowledge_base,
    load_knowledge_base,
    export_to_json,
    import_from_json,
    compute_embedding_statistics,
    compute_similarity_matrix,
    find_duplicates,
    deduplicate_documents,
    cluster_documents,
    retrieve_diverse,
    format_context_for_prompt,
    create_knowledge_base_index,
    evaluate_retrieval_quality
)

__all__ = [
    # Config
    'KnowledgeBaseConfig',
    'VectorStoreConfig',
    'RetrieverConfig',
    'RAGConfig',
    'VietnameseProcessorConfig',
    'EncoderConfig',
    'VQAKnowledgeConfig',
    # Vector Store
    'BaseVectorStore',
    'FAISSVectorStore',
    'InMemoryVectorStore',
    'ChromaVectorStore',
    # Document Store
    'Document',
    'DocumentStore',
    'KnowledgeEntry',
    'VisualKnowledgeEntry',
    # Encoder
    'BaseKnowledgeEncoder',
    'TextKnowledgeEncoder',
    'VisualKnowledgeEncoder',
    'MultimodalKnowledgeEncoder',
    'EncodingResult',
    'create_text_encoder',
    'create_visual_encoder',
    'create_multimodal_encoder',
    # Retriever
    'BaseRetriever',
    'DenseRetriever',
    'SparseRetriever',
    'HybridRetriever',
    'MultimodalRetriever',
    'RerankerRetriever',
    'RetrievalResult',
    'RetrievalMethod',
    'create_dense_retriever',
    'create_hybrid_retriever',
    # Vietnamese
    'VietnameseTextProcessor',
    'VietnameseTokenizer',
    'VietnameseSentenceSplitter',
    'ProcessedText',
    'normalize_vietnamese_text',
    'detect_vietnamese',
    'convert_to_ascii_vietnamese',
    'VIETNAMESE_STOPWORDS',
    # RAG
    'RAGModule',
    'RAGOutput',
    'KnowledgeAugmentedVQA',
    'KnowledgeAugmentedFusion',
    'ContextEncoder',
    'ContextAttention',
    'RAGLoss',
    'create_rag_module',
    'create_knowledge_augmented_vqa',
    # Utils
    'generate_document_id',
    'chunk_documents',
    'batch_encode_documents',
    'save_knowledge_base',
    'load_knowledge_base',
    'export_to_json',
    'import_from_json',
    'compute_embedding_statistics',
    'compute_similarity_matrix',
    'find_duplicates',
    'deduplicate_documents',
    'cluster_documents',
    'retrieve_diverse',
    'format_context_for_prompt',
    'create_knowledge_base_index',
    'evaluate_retrieval_quality'
]
