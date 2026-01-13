# Knowledge Base and RAG Approaches

This document describes the Knowledge Base and Retrieval-Augmented Generation (RAG) implementations for Vietnamese VQA in the AutovivqaModelBuilder project.

## Overview

The Knowledge Base module provides external knowledge retrieval and augmentation capabilities to enhance VQA performance, particularly for questions requiring factual knowledge or context beyond what's visible in the image.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                  Knowledge-Augmented VQA                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────┐  ┌──────────┐  ┌──────────────────────────────┐  │
│  │  Image   │  │ Question │  │      Knowledge Base          │  │
│  │ Encoder  │  │ Encoder  │  │  ┌────────┐  ┌───────────┐  │  │
│  └────┬─────┘  └────┬─────┘  │  │Document│  │  Vector   │  │  │
│       │             │        │  │ Store  │  │   Store   │  │  │
│       ▼             ▼        │  └────────┘  └───────────┘  │  │
│  ┌──────────────────────┐    │                              │  │
│  │  Visual Features     │    └───────────┬──────────────────┘  │
│  └──────────┬───────────┘                │                     │
│             │                            ▼                     │
│             │    ┌─────────────────────────────────────────┐   │
│             │    │              Retriever                   │   │
│             │    │  ┌───────┐ ┌────────┐ ┌────────────┐    │   │
│             │    │  │ Dense │ │ Sparse │ │  Hybrid    │    │   │
│             │    │  └───────┘ └────────┘ └────────────┘    │   │
│             │    └─────────────────┬───────────────────────┘   │
│             │                      │                           │
│             │                      ▼                           │
│             │    ┌─────────────────────────────────────────┐   │
│             │    │           RAG Module                     │   │
│             │    │  ┌─────────────────────────────────────┐│   │
│             │    │  │     Context Encoder & Attention     ││   │
│             │    │  └─────────────────────────────────────┘│   │
│             │    └─────────────────┬───────────────────────┘   │
│             │                      │                           │
│             ▼                      ▼                           │
│       ┌────────────────────────────────────────────────────┐   │
│       │          Knowledge-Augmented Fusion                 │   │
│       │   (Visual + Text + Knowledge → Fused Features)     │   │
│       └────────────────────────────┬───────────────────────┘   │
│                                    │                           │
│                                    ▼                           │
│                          ┌───────────────┐                     │
│                          │  Answer Head  │                     │
│                          └───────────────┘                     │
└─────────────────────────────────────────────────────────────────┘
```

## Components

### 1. Vietnamese Text Processor

Located in `vietnamese_processor.py`, provides Vietnamese-specific text processing:

```python
from src.modeling.knowledge_base import VietnameseTextProcessor

processor = VietnameseTextProcessor(
    use_word_segmentation=True,  # Use underthesea for word segmentation
    lowercase=True,
    remove_stopwords=True
)

# Process text
processed = processor.process("Hà Nội là thủ đô của Việt Nam")

# Extract keywords
keywords = processor.extract_keywords(text, top_k=10)

# Chunk long documents
chunks = processor.chunk_text(text, chunk_size=256, overlap=32)
```

Features:
- Unicode normalization (NFC form)
- Vietnamese word segmentation (via underthesea)
- Vietnamese stopword removal
- Sentence splitting
- Keyword extraction
- Text chunking with overlap

### 2. Document Store

Located in `document_store.py`, manages document storage:

```python
from src.modeling.knowledge_base import DocumentStore, Document

store = DocumentStore()

# Add document
doc = Document(
    id="doc_1",
    content="Phở là món ăn truyền thống của Việt Nam.",
    metadata={"category": "food", "source": "wikipedia"}
)
store.add_document(doc)

# Retrieve
doc = store.get_document("doc_1")

# Search by metadata
food_docs = store.get_by_metadata("category", "food")
```

### 3. Vector Store

Located in `vector_store.py`, provides vector similarity search:

#### In-Memory Store
```python
from src.modeling.knowledge_base import InMemoryVectorStore

store = InMemoryVectorStore(embed_dim=768, metric='cosine')
store.add(id="doc_1", embedding=embedding, text="content")
results = store.search(query_embedding, k=5)
```

#### FAISS Store
```python
from src.modeling.knowledge_base import FAISSVectorStore

store = FAISSVectorStore(
    embed_dim=768,
    index_type='ivf',  # 'flat', 'ivf', 'hnsw'
    metric='cosine',
    nlist=100
)
```

#### ChromaDB Store
```python
from src.modeling.knowledge_base import ChromaVectorStore

store = ChromaVectorStore(
    collection_name="vqa_knowledge",
    embed_dim=768,
    persist_directory="./chroma_db"
)
```

### 4. Knowledge Encoders

Located in `knowledge_encoder.py`:

#### Text Encoder (PhoBERT)
```python
from src.modeling.knowledge_base import create_text_encoder

encoder = create_text_encoder(
    model_name='vinai/phobert-base',
    pooling_strategy='mean',  # 'cls', 'mean', 'max'
    normalize_embeddings=True
)

result = encoder.encode_batch(["Câu hỏi 1", "Câu hỏi 2"])
embeddings = result.pooled_output  # [batch, 768]
```

#### Visual Encoder
```python
from src.modeling.knowledge_base import create_visual_encoder

encoder = create_visual_encoder(
    model_name='openai/clip-vit-base-patch32',
    pooling_strategy='cls'
)

result = encoder.encode_batch(images)
```

#### Multimodal Encoder
```python
from src.modeling.knowledge_base import create_multimodal_encoder

encoder = create_multimodal_encoder(
    text_model='vinai/phobert-base',
    visual_model='openai/clip-vit-base-patch32',
    fusion_method='attention'  # 'concat', 'add', 'multiply', 'attention'
)
```

### 5. Retrievers

Located in `retriever.py`:

#### Dense Retriever
Uses vector similarity for semantic search:
```python
from src.modeling.knowledge_base import DenseRetriever

retriever = DenseRetriever(
    encoder=text_encoder,
    vector_store=faiss_store,
    top_k=5
)

result = retriever.retrieve("Thủ đô Việt Nam là gì?")
```

#### Sparse Retriever (BM25)
Uses term-based matching:
```python
from src.modeling.knowledge_base import SparseRetriever

retriever = SparseRetriever(
    documents=documents,
    method='bm25',
    k1=1.5,
    b=0.75
)
```

#### Hybrid Retriever
Combines dense and sparse:
```python
from src.modeling.knowledge_base import HybridRetriever

retriever = HybridRetriever(
    dense_retriever=dense,
    sparse_retriever=sparse,
    dense_weight=0.6,
    fusion_method='rrf'  # Reciprocal Rank Fusion
)
```

### 6. RAG Module

Located in `rag_module.py`:

```python
from src.modeling.knowledge_base import RAGModule, create_rag_module

rag = RAGModule(
    retriever=retriever,
    context_encoder=encoder,
    embed_dim=768,
    num_contexts=5,
    use_context_attention=True,
    context_fusion='attention'  # 'attention', 'concat', 'gated', 'add'
)

output = rag(query_text="Hà Nội có gì nổi tiếng?", query_embedding=embedding)
# output.output: Enhanced embedding
# output.context: Retrieved documents
# output.attention_weights: Context attention
```

### 7. Knowledge-Augmented Fusion

Trilinear fusion of visual, text, and knowledge:

```python
from src.modeling.knowledge_base import KnowledgeAugmentedFusion

fusion = KnowledgeAugmentedFusion(
    visual_dim=768,
    text_dim=768,
    knowledge_dim=768,
    output_dim=512,
    fusion_method='trilinear'  # 'trilinear', 'cross_attention', 'gated'
)

fused = fusion(visual_features, text_features, knowledge_features)
```

### 8. Complete VQA Model

```python
from src.modeling.knowledge_base import create_knowledge_augmented_vqa

model = create_knowledge_augmented_vqa(
    visual_encoder=visual_encoder,
    text_encoder=text_encoder,
    retriever=retriever,
    context_encoder=context_encoder,
    num_answers=3000
)

outputs = model(images, questions)
# outputs['logits']: Answer predictions
# outputs['rag_outputs']: RAG information
```

## Configuration

Use dataclass configurations for easy setup:

```python
from src.modeling.knowledge_base import (
    KnowledgeBaseConfig,
    VectorStoreConfig,
    RetrieverConfig,
    RAGConfig
)

kb_config = KnowledgeBaseConfig(
    vector_store=VectorStoreConfig(
        type='faiss',
        embed_dim=768,
        metric='cosine'
    ),
    retriever=RetrieverConfig(
        type='hybrid',
        top_k=5,
        dense_weight=0.6
    ),
    rag=RAGConfig(
        num_contexts=5,
        context_fusion='attention'
    )
)
```

## Utilities

Located in `kb_utils.py`:

```python
from src.modeling.knowledge_base import (
    chunk_documents,
    batch_encode_documents,
    save_knowledge_base,
    load_knowledge_base,
    find_duplicates,
    deduplicate_documents,
    retrieve_diverse,
    evaluate_retrieval_quality
)

# Chunk documents
chunks = chunk_documents(documents, chunk_size=256, overlap=32)

# Batch encode
embeddings = batch_encode_documents(documents, encoder, batch_size=32)

# Save/load
save_knowledge_base(documents, embeddings, metadata, "kb.pkl")
documents, embeddings, metadata = load_knowledge_base("kb.pkl")

# Remove duplicates
unique_docs, unique_embs, indices = deduplicate_documents(
    documents, embeddings, similarity_threshold=0.95
)

# Diverse retrieval (MMR)
results = retrieve_diverse(query_emb, doc_embs, documents, k=5, diversity_weight=0.3)

# Evaluate
metrics = evaluate_retrieval_quality(retriever, queries, ground_truth)
```

## Best Practices

### 1. Building Knowledge Base

```python
# 1. Collect and clean documents
documents = load_vietnamese_documents()
processor = VietnameseTextProcessor()
cleaned = [processor.process(doc) for doc in documents]

# 2. Chunk long documents
chunks = chunk_documents(cleaned, chunk_size=256, overlap=32)

# 3. Encode
embeddings = batch_encode_documents(chunks, encoder)

# 4. Deduplicate
unique_chunks, unique_embeddings, _ = deduplicate_documents(
    chunks, embeddings, threshold=0.95
)

# 5. Index
store = FAISSVectorStore(embed_dim=768, index_type='ivf')
for i, (chunk, emb) in enumerate(zip(unique_chunks, unique_embeddings)):
    store.add(id=f"chunk_{i}", embedding=emb, text=chunk)

# 6. Save
store.save("knowledge_index")
```

### 2. Vietnamese-Specific Considerations

- **Word Segmentation**: Essential for Vietnamese. Use underthesea or VnCoreNLP.
- **Encoding**: Use PhoBERT (vinai/phobert-base) for best Vietnamese understanding.
- **Stopwords**: Remove common Vietnamese stopwords for better retrieval.
- **Unicode**: Always normalize to NFC form.

### 3. Retrieval Strategies

For Vietnamese VQA:
1. **Use Hybrid Retrieval**: Combines semantic (dense) with exact matching (sparse)
2. **Tune BM25 Parameters**: k1=1.5, b=0.75 work well for Vietnamese
3. **Context Diversity**: Use MMR to avoid redundant contexts
4. **Reranking**: Consider cross-encoder reranking for precision

### 4. RAG Training

```python
from src.modeling.knowledge_base import RAGLoss

loss_fn = RAGLoss(
    answer_loss_weight=1.0,
    retrieval_loss_weight=0.1,  # Encourage focused attention
    diversity_loss_weight=0.01  # Encourage diverse contexts
)

losses = loss_fn(logits, targets, attention_weights, context_embeddings)
total_loss = losses['total_loss']
```

## Integration with VQA Pipeline

```python
# 1. Setup knowledge base
kb_store = FAISSVectorStore.load("knowledge_index")
retriever = create_hybrid_retriever(encoder, kb_store, documents)

# 2. Create RAG module
rag = RAGModule(retriever, encoder, num_contexts=5)

# 3. Create full model
model = KnowledgeAugmentedVQA(
    visual_encoder=visual_encoder,
    text_encoder=text_encoder,
    rag_module=rag,
    fusion_module=fusion,
    num_answers=num_answers
)

# 4. Training loop
for batch in dataloader:
    outputs = model(batch['images'], batch['questions'])
    losses = loss_fn(
        outputs['logits'],
        batch['answers'],
        outputs['rag_outputs'][0].attention_weights
    )
    losses['total_loss'].backward()
    optimizer.step()
```

## Files Reference

| File | Description |
|------|-------------|
| `kb_config.py` | Configuration dataclasses |
| `document_store.py` | Document storage and management |
| `vector_store.py` | Vector similarity search (FAISS, Chroma) |
| `vietnamese_processor.py` | Vietnamese text processing |
| `knowledge_encoder.py` | Text, visual, multimodal encoders |
| `retriever.py` | Dense, sparse, hybrid retrievers |
| `rag_module.py` | RAG and knowledge-augmented VQA |
| `kb_utils.py` | Utility functions |
