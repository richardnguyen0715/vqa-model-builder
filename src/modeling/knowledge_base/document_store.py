"""
Document Store Module
Provides document and knowledge entry data structures and storage.
"""

import torch
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Union
from datetime import datetime
import json
import hashlib
import uuid


@dataclass
class Document:
    """
    Base document class for knowledge storage.
    
    Attributes:
        content: Text content of the document
        doc_id: Unique document identifier
        metadata: Additional metadata
        embedding: Document embedding vector
        source: Source of the document
        created_at: Creation timestamp
        updated_at: Last update timestamp
    """
    content: str
    doc_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[torch.Tensor] = None
    source: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    
    def __post_init__(self):
        """Generate doc_id and timestamps if not provided."""
        if self.doc_id is None:
            self.doc_id = self._generate_id()
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()
        self.updated_at = datetime.now().isoformat()
    
    def _generate_id(self) -> str:
        """
        Generate unique document ID based on content hash.
        
        Returns:
            Unique document identifier
        """
        content_hash = hashlib.md5(self.content.encode()).hexdigest()[:8]
        return f"doc_{content_hash}_{uuid.uuid4().hex[:8]}"
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert document to dictionary.
        
        Returns:
            Dictionary representation
        """
        return {
            'doc_id': self.doc_id,
            'content': self.content,
            'metadata': self.metadata,
            'source': self.source,
            'created_at': self.created_at,
            'updated_at': self.updated_at
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Document':
        """
        Create document from dictionary.
        
        Args:
            data: Dictionary with document data
            
        Returns:
            Document instance
        """
        return cls(
            content=data['content'],
            doc_id=data.get('doc_id'),
            metadata=data.get('metadata', {}),
            source=data.get('source'),
            created_at=data.get('created_at'),
            updated_at=data.get('updated_at')
        )


@dataclass
class KnowledgeEntry(Document):
    """
    Knowledge entry with additional VQA-specific fields.
    
    Attributes:
        knowledge_type: Type of knowledge (factual, commonsense, visual)
        entities: Extracted entities
        relations: Extracted relations
        confidence: Confidence score
        language: Language code
        category: Knowledge category
    """
    knowledge_type: str = 'factual'
    entities: List[str] = field(default_factory=list)
    relations: List[Dict[str, str]] = field(default_factory=list)
    confidence: float = 1.0
    language: str = 'vi'
    category: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with knowledge fields."""
        base_dict = super().to_dict()
        base_dict.update({
            'knowledge_type': self.knowledge_type,
            'entities': self.entities,
            'relations': self.relations,
            'confidence': self.confidence,
            'language': self.language,
            'category': self.category
        })
        return base_dict


@dataclass
class VisualKnowledgeEntry(KnowledgeEntry):
    """
    Visual knowledge entry with image-specific information.
    
    Attributes:
        image_id: Associated image identifier
        objects: Detected objects with bounding boxes
        attributes: Object attributes
        spatial_relations: Spatial relationships between objects
        scene_type: Scene classification
        caption: Image caption
        ocr_text: OCR extracted text
    """
    image_id: Optional[str] = None
    objects: List[Dict[str, Any]] = field(default_factory=list)
    attributes: List[Dict[str, Any]] = field(default_factory=list)
    spatial_relations: List[Dict[str, Any]] = field(default_factory=list)
    scene_type: Optional[str] = None
    caption: Optional[str] = None
    ocr_text: Optional[str] = None
    
    def __post_init__(self):
        """Set knowledge type to visual."""
        super().__post_init__()
        self.knowledge_type = 'visual'
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with visual fields."""
        base_dict = super().to_dict()
        base_dict.update({
            'image_id': self.image_id,
            'objects': self.objects,
            'attributes': self.attributes,
            'spatial_relations': self.spatial_relations,
            'scene_type': self.scene_type,
            'caption': self.caption,
            'ocr_text': self.ocr_text
        })
        return base_dict


class DocumentStore:
    """
    Document store for managing knowledge documents.
    Provides CRUD operations and indexing.
    """
    
    def __init__(
        self,
        persist_path: Optional[str] = None,
        max_documents: int = 1000000
    ):
        """
        Initialize document store.
        
        Args:
            persist_path: Path to persist documents
            max_documents: Maximum number of documents
        """
        self.persist_path = persist_path
        self.max_documents = max_documents
        
        # Document storage
        self.documents: Dict[str, Document] = {}
        
        # Indices for fast lookup
        self.source_index: Dict[str, List[str]] = {}
        self.type_index: Dict[str, List[str]] = {}
        self.category_index: Dict[str, List[str]] = {}
        
        # Load existing documents if persist path exists
        if persist_path:
            self._load_from_disk()
    
    def add(
        self,
        document: Union[Document, str],
        metadata: Optional[Dict] = None
    ) -> str:
        """
        Add document to store.
        
        Args:
            document: Document or content string
            metadata: Optional metadata
            
        Returns:
            Document ID
        """
        if isinstance(document, str):
            document = Document(content=document, metadata=metadata or {})
        
        if len(self.documents) >= self.max_documents:
            raise ValueError(f"Document store full (max: {self.max_documents})")
        
        self.documents[document.doc_id] = document
        self._update_indices(document)
        
        return document.doc_id
    
    def add_batch(
        self,
        documents: List[Union[Document, str]],
        metadata_list: Optional[List[Dict]] = None
    ) -> List[str]:
        """
        Add multiple documents.
        
        Args:
            documents: List of documents or content strings
            metadata_list: Optional list of metadata dicts
            
        Returns:
            List of document IDs
        """
        doc_ids = []
        for i, doc in enumerate(documents):
            metadata = metadata_list[i] if metadata_list else None
            doc_id = self.add(doc, metadata)
            doc_ids.append(doc_id)
        return doc_ids
    
    def get(self, doc_id: str) -> Optional[Document]:
        """
        Get document by ID.
        
        Args:
            doc_id: Document identifier
            
        Returns:
            Document or None
        """
        return self.documents.get(doc_id)
    
    def get_batch(self, doc_ids: List[str]) -> List[Document]:
        """
        Get multiple documents by IDs.
        
        Args:
            doc_ids: List of document identifiers
            
        Returns:
            List of documents
        """
        return [self.documents[doc_id] for doc_id in doc_ids if doc_id in self.documents]
    
    def update(self, doc_id: str, updates: Dict[str, Any]) -> bool:
        """
        Update document fields.
        
        Args:
            doc_id: Document identifier
            updates: Dictionary of field updates
            
        Returns:
            Success status
        """
        if doc_id not in self.documents:
            return False
        
        doc = self.documents[doc_id]
        for key, value in updates.items():
            if hasattr(doc, key):
                setattr(doc, key, value)
        
        doc.updated_at = datetime.now().isoformat()
        return True
    
    def delete(self, doc_id: str) -> bool:
        """
        Delete document by ID.
        
        Args:
            doc_id: Document identifier
            
        Returns:
            Success status
        """
        if doc_id not in self.documents:
            return False
        
        doc = self.documents.pop(doc_id)
        self._remove_from_indices(doc)
        return True
    
    def search_by_source(self, source: str) -> List[Document]:
        """
        Search documents by source.
        
        Args:
            source: Source identifier
            
        Returns:
            List of matching documents
        """
        doc_ids = self.source_index.get(source, [])
        return self.get_batch(doc_ids)
    
    def search_by_type(self, knowledge_type: str) -> List[Document]:
        """
        Search documents by knowledge type.
        
        Args:
            knowledge_type: Knowledge type
            
        Returns:
            List of matching documents
        """
        doc_ids = self.type_index.get(knowledge_type, [])
        return self.get_batch(doc_ids)
    
    def search_by_category(self, category: str) -> List[Document]:
        """
        Search documents by category.
        
        Args:
            category: Category name
            
        Returns:
            List of matching documents
        """
        doc_ids = self.category_index.get(category, [])
        return self.get_batch(doc_ids)
    
    def get_all(self) -> List[Document]:
        """
        Get all documents.
        
        Returns:
            List of all documents
        """
        return list(self.documents.values())
    
    def count(self) -> int:
        """
        Get document count.
        
        Returns:
            Number of documents
        """
        return len(self.documents)
    
    def clear(self):
        """Clear all documents and indices."""
        self.documents.clear()
        self.source_index.clear()
        self.type_index.clear()
        self.category_index.clear()
    
    def _update_indices(self, doc: Document):
        """
        Update indices with new document.
        
        Args:
            doc: Document to index
        """
        # Source index
        if doc.source:
            if doc.source not in self.source_index:
                self.source_index[doc.source] = []
            self.source_index[doc.source].append(doc.doc_id)
        
        # Type index (for KnowledgeEntry)
        if isinstance(doc, KnowledgeEntry):
            if doc.knowledge_type not in self.type_index:
                self.type_index[doc.knowledge_type] = []
            self.type_index[doc.knowledge_type].append(doc.doc_id)
            
            # Category index
            if doc.category:
                if doc.category not in self.category_index:
                    self.category_index[doc.category] = []
                self.category_index[doc.category].append(doc.doc_id)
    
    def _remove_from_indices(self, doc: Document):
        """
        Remove document from indices.
        
        Args:
            doc: Document to remove
        """
        if doc.source and doc.source in self.source_index:
            self.source_index[doc.source] = [
                d for d in self.source_index[doc.source] if d != doc.doc_id
            ]
        
        if isinstance(doc, KnowledgeEntry):
            if doc.knowledge_type in self.type_index:
                self.type_index[doc.knowledge_type] = [
                    d for d in self.type_index[doc.knowledge_type] if d != doc.doc_id
                ]
            
            if doc.category and doc.category in self.category_index:
                self.category_index[doc.category] = [
                    d for d in self.category_index[doc.category] if d != doc.doc_id
                ]
    
    def save(self, path: Optional[str] = None):
        """
        Save documents to disk.
        
        Args:
            path: Save path (uses persist_path if not provided)
        """
        save_path = path or self.persist_path
        if not save_path:
            raise ValueError("No save path provided")
        
        data = {
            'documents': {
                doc_id: doc.to_dict() 
                for doc_id, doc in self.documents.items()
            },
            'metadata': {
                'count': len(self.documents),
                'saved_at': datetime.now().isoformat()
            }
        }
        
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def _load_from_disk(self):
        """Load documents from disk."""
        import os
        if not os.path.exists(self.persist_path):
            return
        
        try:
            with open(self.persist_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            for doc_id, doc_data in data.get('documents', {}).items():
                # Determine document type
                if 'image_id' in doc_data:
                    doc = VisualKnowledgeEntry(
                        content=doc_data['content'],
                        doc_id=doc_data.get('doc_id'),
                        metadata=doc_data.get('metadata', {}),
                        source=doc_data.get('source'),
                        image_id=doc_data.get('image_id'),
                        objects=doc_data.get('objects', []),
                        attributes=doc_data.get('attributes', []),
                        spatial_relations=doc_data.get('spatial_relations', []),
                        scene_type=doc_data.get('scene_type'),
                        caption=doc_data.get('caption'),
                        ocr_text=doc_data.get('ocr_text')
                    )
                elif 'knowledge_type' in doc_data:
                    doc = KnowledgeEntry(
                        content=doc_data['content'],
                        doc_id=doc_data.get('doc_id'),
                        metadata=doc_data.get('metadata', {}),
                        source=doc_data.get('source'),
                        knowledge_type=doc_data.get('knowledge_type', 'factual'),
                        entities=doc_data.get('entities', []),
                        relations=doc_data.get('relations', []),
                        confidence=doc_data.get('confidence', 1.0),
                        language=doc_data.get('language', 'vi'),
                        category=doc_data.get('category')
                    )
                else:
                    doc = Document.from_dict(doc_data)
                
                self.documents[doc.doc_id] = doc
                self._update_indices(doc)
                
        except Exception as e:
            import warnings
            warnings.warn(f"Failed to load documents from {self.persist_path}: {e}")
