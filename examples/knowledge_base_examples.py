"""
Knowledge Base Examples
Demonstrates usage of Knowledge Base and RAG components for Vietnamese VQA.
"""

import torch


def example_vietnamese_processor():
    """Demonstrate Vietnamese text processing."""
    from src.modeling.knowledge_base import (
        VietnameseTextProcessor,
        VietnameseTokenizer,
        VietnameseSentenceSplitter,
        normalize_vietnamese_text,
        detect_vietnamese
    )
    
    print("=== Vietnamese Text Processing ===\n")
    
    # Basic normalization
    text = "Hà Nội là thủ đô của Việt Nam."
    normalized = normalize_vietnamese_text(text)
    print(f"Original: {text}")
    print(f"Normalized: {normalized}")
    
    # Language detection
    is_vn, confidence = detect_vietnamese(text)
    print(f"Is Vietnamese: {is_vn}, Confidence: {confidence:.2f}")
    
    # Full processor
    processor = VietnameseTextProcessor(
        use_word_segmentation=True,
        lowercase=True,
        remove_stopwords=True
    )
    
    processed = processor.process(text)
    print(f"Processed: {processed}")
    
    # Extract keywords
    long_text = """
    Việt Nam là một quốc gia nằm ở khu vực Đông Nam Á.
    Thủ đô của Việt Nam là Hà Nội, thành phố lớn nhất là Thành phố Hồ Chí Minh.
    Việt Nam có nhiều di sản văn hóa và thiên nhiên được UNESCO công nhận.
    """
    keywords = processor.extract_keywords(long_text, top_k=5)
    print(f"Keywords: {keywords}")
    
    # Chunk text
    chunks = processor.chunk_text(long_text, chunk_size=20, overlap=5)
    print(f"Number of chunks: {len(chunks)}")
    for i, chunk in enumerate(chunks):
        print(f"  Chunk {i}: {chunk[:50]}...")
    
    print()


def example_document_store():
    """Demonstrate document storage."""
    from src.modeling.knowledge_base import (
        Document,
        DocumentStore,
        KnowledgeEntry
    )
    
    print("=== Document Store ===\n")
    
    # Create document store
    store = DocumentStore()
    
    # Add documents
    doc1 = Document(
        id="doc_1",
        content="Hà Nội là thủ đô của Việt Nam với lịch sử hơn 1000 năm.",
        metadata={"category": "geography", "source": "wikipedia"}
    )
    
    doc2 = Document(
        id="doc_2",
        content="Phở là món ăn truyền thống nổi tiếng của Việt Nam.",
        metadata={"category": "food", "source": "wikipedia"}
    )
    
    store.add_document(doc1)
    store.add_document(doc2)
    
    print(f"Total documents: {store.size}")
    
    # Retrieve document
    retrieved = store.get_document("doc_1")
    print(f"Retrieved: {retrieved.content}")
    
    # Search by metadata
    food_docs = store.get_by_metadata("category", "food")
    print(f"Food documents: {len(food_docs)}")
    
    # Add knowledge entry
    entry = KnowledgeEntry(
        id="knowledge_1",
        content="Việt Nam có 54 dân tộc.",
        category="culture",
        source="statistics"
    )
    store.add_knowledge(entry)
    
    print()


def example_vector_store():
    """Demonstrate vector store operations."""
    import numpy as np
    from src.modeling.knowledge_base import (
        InMemoryVectorStore,
        FAISSVectorStore
    )
    
    print("=== Vector Store ===\n")
    
    # Create vector store
    embed_dim = 768
    store = InMemoryVectorStore(embed_dim=embed_dim)
    
    # Add vectors
    for i in range(10):
        embedding = np.random.randn(embed_dim).astype(np.float32)
        embedding = embedding / np.linalg.norm(embedding)
        
        store.add(
            id=f"doc_{i}",
            embedding=embedding,
            text=f"Document {i} content",
            metadata={"index": i}
        )
    
    print(f"Total vectors: {store.size}")
    
    # Search
    query = np.random.randn(embed_dim).astype(np.float32)
    query = query / np.linalg.norm(query)
    
    results = store.search(query, k=3)
    print("Search results:")
    for r in results:
        print(f"  ID: {r['id']}, Score: {r['score']:.4f}")
    
    # FAISS vector store
    print("\nFAISS Vector Store:")
    try:
        faiss_store = FAISSVectorStore(embed_dim=embed_dim, index_type='flat')
        
        for i in range(100):
            embedding = np.random.randn(embed_dim).astype(np.float32)
            faiss_store.add(
                id=f"faiss_doc_{i}",
                embedding=embedding,
                text=f"FAISS document {i}"
            )
        
        print(f"FAISS store size: {faiss_store.size}")
        
        results = faiss_store.search(query, k=5)
        print(f"FAISS search returned {len(results)} results")
        
    except ImportError:
        print("FAISS not installed. Install with: pip install faiss-cpu")
    
    print()


def example_knowledge_encoder():
    """Demonstrate knowledge encoding."""
    print("=== Knowledge Encoder ===\n")
    
    try:
        from src.modeling.knowledge_base import (
            TextKnowledgeEncoder,
            create_text_encoder
        )
        
        # Create encoder (requires transformers)
        print("Creating PhoBERT encoder...")
        encoder = create_text_encoder(
            model_name='vinai/phobert-base',
            pooling_strategy='mean',
            normalize_embeddings=True
        )
        
        # Encode text
        texts = [
            "Hà Nội là thủ đô của Việt Nam.",
            "Thành phố Hồ Chí Minh là thành phố lớn nhất."
        ]
        
        result = encoder.encode_batch(texts)
        print(f"Embedding shape: {result.pooled_output.shape}")
        print(f"Embedding norm: {result.pooled_output.norm(dim=-1)}")
        
    except ImportError:
        print("transformers not installed. Install with: pip install transformers")
    
    print()


def example_retriever():
    """Demonstrate retrieval mechanisms."""
    import numpy as np
    from src.modeling.knowledge_base import (
        SparseRetriever,
        RetrievalResult
    )
    
    print("=== Retriever ===\n")
    
    # Create sparse retriever (BM25)
    documents = [
        "Hà Nội là thủ đô của Việt Nam.",
        "Phở là món ăn truyền thống của Việt Nam.",
        "Áo dài là trang phục truyền thống của Việt Nam.",
        "Việt Nam nằm ở khu vực Đông Nam Á.",
        "Hạ Long là vịnh đẹp nhất Việt Nam."
    ]
    
    retriever = SparseRetriever(
        documents=documents,
        method='bm25',
        top_k=3
    )
    
    # Search
    query = "thủ đô Việt Nam"
    result = retriever.retrieve(query)
    
    print(f"Query: {query}")
    print("Results:")
    for i, (doc, score) in enumerate(zip(result.documents, result.scores)):
        print(f"  {i+1}. Score: {score:.4f} - {doc}")
    
    print()


def example_rag_module():
    """Demonstrate RAG module (conceptual example)."""
    print("=== RAG Module ===\n")
    
    print("RAG Module integrates retrieval with generation.")
    print("Key components:")
    print("  1. Retriever: Finds relevant documents")
    print("  2. Context Encoder: Encodes retrieved context")
    print("  3. Context Attention: Attends over context")
    print("  4. Fusion: Combines query with context")
    print()
    
    # Show config structure
    from src.modeling.knowledge_base import RAGConfig
    
    config = RAGConfig(
        num_contexts=5,
        context_max_length=512,
        use_context_attention=True,
        context_fusion='attention'
    )
    
    print(f"RAG Config:")
    print(f"  num_contexts: {config.num_contexts}")
    print(f"  context_max_length: {config.context_max_length}")
    print(f"  use_context_attention: {config.use_context_attention}")
    print(f"  context_fusion: {config.context_fusion}")
    
    print()


def example_knowledge_augmented_vqa():
    """Demonstrate Knowledge-Augmented VQA (conceptual example)."""
    print("=== Knowledge-Augmented VQA ===\n")
    
    print("Knowledge-Augmented VQA combines:")
    print("  1. Visual features from image encoder")
    print("  2. Text features from question encoder")
    print("  3. Knowledge features from RAG retrieval")
    print()
    
    from src.modeling.knowledge_base import KnowledgeAugmentedFusion
    
    # Create fusion module
    fusion = KnowledgeAugmentedFusion(
        visual_dim=768,
        text_dim=768,
        knowledge_dim=768,
        output_dim=512,
        fusion_method='trilinear'
    )
    
    # Simulate forward pass
    batch_size = 2
    visual = torch.randn(batch_size, 768)
    text = torch.randn(batch_size, 768)
    knowledge = torch.randn(batch_size, 768)
    
    fused = fusion(visual, text, knowledge)
    print(f"Fused output shape: {fused.shape}")
    
    print()


def example_kb_utilities():
    """Demonstrate knowledge base utilities."""
    import numpy as np
    from src.modeling.knowledge_base import (
        generate_document_id,
        chunk_documents,
        compute_embedding_statistics,
        find_duplicates,
        retrieve_diverse
    )
    
    print("=== KB Utilities ===\n")
    
    # Generate document ID
    content = "Việt Nam là quốc gia đẹp."
    doc_id = generate_document_id(content)
    print(f"Document ID: {doc_id}")
    
    # Chunk documents
    documents = [
        "Đây là một văn bản dài cần được chia thành nhiều phần nhỏ hơn. " * 20
    ]
    chunks = chunk_documents(documents, chunk_size=50, overlap=10)
    print(f"Number of chunks: {len(chunks)}")
    
    # Compute statistics
    embeddings = np.random.randn(100, 768).astype(np.float32)
    stats = compute_embedding_statistics(embeddings)
    print(f"Embedding stats:")
    print(f"  Mean norm: {stats['mean_norm']:.4f}")
    print(f"  Std norm: {stats['std_norm']:.4f}")
    
    # Find duplicates
    embeddings_with_dup = np.vstack([embeddings[:10], embeddings[0:3] + 0.01])
    documents_test = [f"Doc {i}" for i in range(13)]
    dups = find_duplicates(documents_test, embeddings_with_dup, similarity_threshold=0.99)
    print(f"Found {len(dups)} near-duplicates")
    
    # Diverse retrieval
    query = np.random.randn(768).astype(np.float32)
    diverse_results = retrieve_diverse(
        query, embeddings[:20],
        [f"Doc {i}" for i in range(20)],
        k=5, diversity_weight=0.3
    )
    print(f"Diverse results: {len(diverse_results)} documents")
    
    print()


def main():
    """Run all examples."""
    print("=" * 60)
    print("Knowledge Base Examples for Vietnamese VQA")
    print("=" * 60)
    print()
    
    example_vietnamese_processor()
    example_document_store()
    example_vector_store()
    example_retriever()
    example_rag_module()
    example_knowledge_augmented_vqa()
    example_kb_utilities()
    
    print("=" * 60)
    print("All examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
