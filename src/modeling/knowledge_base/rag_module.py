"""
RAG Module
Retrieval-Augmented Generation for Vietnamese VQA.
"""

from typing import Optional, List, Dict, Any, Union, Tuple
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from .retriever import BaseRetriever, RetrievalResult


@dataclass
class RAGOutput:
    """
    Container for RAG output.
    
    Attributes:
        output: Generated output embedding or text
        context: Retrieved context
        context_scores: Context relevance scores
        retrieval_results: Raw retrieval results
        attention_weights: Context attention weights
    """
    output: torch.Tensor
    context: Optional[List[str]] = None
    context_scores: Optional[List[float]] = None
    retrieval_results: Optional[RetrievalResult] = None
    attention_weights: Optional[torch.Tensor] = None


class ContextEncoder(nn.Module):
    """
    Encoder for retrieved context documents.
    """
    
    def __init__(
        self,
        encoder: Any,
        max_context_length: int = 512,
        context_pooling: str = 'mean'
    ):
        """
        Initialize context encoder.
        
        Args:
            encoder: Base encoder module
            max_context_length: Maximum context length
            context_pooling: Context pooling method
        """
        super().__init__()
        self.encoder = encoder
        self.max_context_length = max_context_length
        self.context_pooling = context_pooling
    
    def forward(
        self,
        contexts: List[str]
    ) -> torch.Tensor:
        """
        Encode context documents.
        
        Args:
            contexts: List of context strings
            
        Returns:
            Context embeddings [num_contexts, embed_dim]
        """
        if not contexts:
            return None
        
        # Encode all contexts
        result = self.encoder.encode_batch(contexts)
        return result.pooled_output


class ContextAttention(nn.Module):
    """
    Attention mechanism for context selection and weighting.
    """
    
    def __init__(
        self,
        query_dim: int,
        context_dim: int,
        hidden_dim: Optional[int] = None,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        """
        Initialize context attention.
        
        Args:
            query_dim: Query embedding dimension
            context_dim: Context embedding dimension
            hidden_dim: Hidden dimension
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()
        
        hidden_dim = hidden_dim or query_dim
        
        # Project query and context to same dimension
        self.query_proj = nn.Linear(query_dim, hidden_dim)
        self.context_proj = nn.Linear(context_dim, hidden_dim)
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, query_dim)
        
        self.layer_norm = nn.LayerNorm(query_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        query: torch.Tensor,
        context: torch.Tensor,
        context_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply attention over context.
        
        Args:
            query: Query embedding [batch, query_dim]
            context: Context embeddings [batch, num_contexts, context_dim]
            context_mask: Context attention mask
            
        Returns:
            Tuple of (attended_context, attention_weights)
        """
        # Add sequence dimension to query if needed
        if query.dim() == 2:
            query = query.unsqueeze(1)  # [batch, 1, query_dim]
        
        # Project
        query_proj = self.query_proj(query)  # [batch, 1, hidden]
        context_proj = self.context_proj(context)  # [batch, num_contexts, hidden]
        
        # Attention
        attended, attn_weights = self.attention(
            query_proj,
            context_proj,
            context_proj,
            key_padding_mask=context_mask
        )
        
        # Output projection
        attended = self.output_proj(attended)
        attended = attended.squeeze(1)  # [batch, query_dim]
        
        # Residual connection
        query_squeezed = query.squeeze(1)
        output = self.layer_norm(query_squeezed + self.dropout(attended))
        
        return output, attn_weights.squeeze(1)


class RAGModule(nn.Module):
    """
    Retrieval-Augmented Generation module for VQA.
    """
    
    def __init__(
        self,
        retriever: BaseRetriever,
        context_encoder: Any,
        embed_dim: int = 768,
        num_contexts: int = 5,
        use_context_attention: bool = True,
        context_fusion: str = 'attention',
        dropout: float = 0.1
    ):
        """
        Initialize RAG module.
        
        Args:
            retriever: Document retriever
            context_encoder: Context text encoder
            embed_dim: Embedding dimension
            num_contexts: Number of contexts to retrieve
            use_context_attention: Whether to use context attention
            context_fusion: Context fusion method (attention, concat, gated)
            dropout: Dropout rate
        """
        super().__init__()
        
        self.retriever = retriever
        self.context_encoder = ContextEncoder(context_encoder)
        self.embed_dim = embed_dim
        self.num_contexts = num_contexts
        self.use_context_attention = use_context_attention
        self.context_fusion = context_fusion
        
        # Context attention
        if use_context_attention:
            self.context_attention = ContextAttention(
                query_dim=embed_dim,
                context_dim=embed_dim,
                dropout=dropout
            )
        
        # Fusion layers
        if context_fusion == 'concat':
            self.fusion_layer = nn.Linear(embed_dim * 2, embed_dim)
        elif context_fusion == 'gated':
            self.gate_layer = nn.Linear(embed_dim * 2, embed_dim)
            self.fusion_layer = nn.Linear(embed_dim, embed_dim)
        else:
            self.fusion_layer = None
        
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
    
    def retrieve_context(
        self,
        query: str,
        top_k: Optional[int] = None
    ) -> RetrievalResult:
        """
        Retrieve context documents for query.
        
        Args:
            query: Query string
            top_k: Number of documents to retrieve
            
        Returns:
            Retrieval result
        """
        k = top_k or self.num_contexts
        return self.retriever.retrieve(query, top_k=k)
    
    def encode_context(
        self,
        contexts: List[str]
    ) -> torch.Tensor:
        """
        Encode context documents.
        
        Args:
            contexts: List of context strings
            
        Returns:
            Context embeddings
        """
        return self.context_encoder(contexts)
    
    def fuse_with_context(
        self,
        query_embedding: torch.Tensor,
        context_embedding: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Fuse query with context embeddings.
        
        Args:
            query_embedding: Query embedding [batch, embed_dim]
            context_embedding: Context embeddings [batch, num_contexts, embed_dim]
            
        Returns:
            Tuple of (fused_embedding, attention_weights)
        """
        attn_weights = None
        
        if self.use_context_attention:
            attended, attn_weights = self.context_attention(
                query_embedding,
                context_embedding
            )
        else:
            # Simple mean pooling
            attended = context_embedding.mean(dim=1)
        
        # Fusion
        if self.context_fusion == 'attention':
            fused = attended
            
        elif self.context_fusion == 'concat':
            combined = torch.cat([query_embedding, attended], dim=-1)
            fused = self.fusion_layer(combined)
            
        elif self.context_fusion == 'gated':
            combined = torch.cat([query_embedding, attended], dim=-1)
            gate = torch.sigmoid(self.gate_layer(combined))
            fused = self.fusion_layer(gate * attended + (1 - gate) * query_embedding)
            
        else:  # add
            fused = query_embedding + attended
        
        fused = self.layer_norm(fused)
        
        return fused, attn_weights
    
    def forward(
        self,
        query_text: str,
        query_embedding: torch.Tensor
    ) -> RAGOutput:
        """
        Forward pass with retrieval augmentation.
        
        Args:
            query_text: Query text for retrieval
            query_embedding: Query embedding tensor
            
        Returns:
            RAG output
        """
        # Retrieve context
        retrieval_result = self.retrieve_context(query_text)
        
        if not retrieval_result.documents:
            return RAGOutput(
                output=query_embedding,
                context=[],
                context_scores=[],
                retrieval_results=retrieval_result
            )
        
        # Encode context
        context_embedding = self.encode_context(retrieval_result.documents)
        
        # Add batch dimension if needed
        if context_embedding.dim() == 2:
            context_embedding = context_embedding.unsqueeze(0)  # [1, num_contexts, embed_dim]
        
        if query_embedding.dim() == 1:
            query_embedding = query_embedding.unsqueeze(0)  # [1, embed_dim]
        
        # Fuse
        fused, attn_weights = self.fuse_with_context(query_embedding, context_embedding)
        
        return RAGOutput(
            output=fused,
            context=retrieval_result.documents,
            context_scores=retrieval_result.scores,
            retrieval_results=retrieval_result,
            attention_weights=attn_weights
        )


class KnowledgeAugmentedFusion(nn.Module):
    """
    Fusion module that incorporates external knowledge.
    Designed for Vietnamese VQA with knowledge augmentation.
    """
    
    def __init__(
        self,
        visual_dim: int,
        text_dim: int,
        knowledge_dim: int,
        output_dim: int,
        fusion_method: str = 'trilinear',
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        """
        Initialize knowledge augmented fusion.
        
        Args:
            visual_dim: Visual feature dimension
            text_dim: Text feature dimension
            knowledge_dim: Knowledge feature dimension
            output_dim: Output dimension
            fusion_method: Fusion method
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()
        
        self.visual_dim = visual_dim
        self.text_dim = text_dim
        self.knowledge_dim = knowledge_dim
        self.output_dim = output_dim
        self.fusion_method = fusion_method
        
        # Projection layers
        self.visual_proj = nn.Linear(visual_dim, output_dim)
        self.text_proj = nn.Linear(text_dim, output_dim)
        self.knowledge_proj = nn.Linear(knowledge_dim, output_dim)
        
        if fusion_method == 'trilinear':
            # Trilinear interaction
            self.trilinear = nn.Parameter(
                torch.randn(output_dim, output_dim, output_dim) * 0.01
            )
            self.output_layer = nn.Linear(output_dim, output_dim)
            
        elif fusion_method == 'cross_attention':
            # Cross-modal attention
            self.vt_attention = nn.MultiheadAttention(
                output_dim, num_heads, dropout=dropout, batch_first=True
            )
            self.tk_attention = nn.MultiheadAttention(
                output_dim, num_heads, dropout=dropout, batch_first=True
            )
            self.vk_attention = nn.MultiheadAttention(
                output_dim, num_heads, dropout=dropout, batch_first=True
            )
            self.output_layer = nn.Linear(output_dim * 3, output_dim)
            
        elif fusion_method == 'gated':
            # Gated fusion
            self.gate_visual = nn.Linear(output_dim * 3, output_dim)
            self.gate_text = nn.Linear(output_dim * 3, output_dim)
            self.gate_knowledge = nn.Linear(output_dim * 3, output_dim)
            self.output_layer = nn.Linear(output_dim, output_dim)
        
        self.layer_norm = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        visual_features: torch.Tensor,
        text_features: torch.Tensor,
        knowledge_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Fuse visual, text, and knowledge features.
        
        Args:
            visual_features: Visual features [batch, visual_dim]
            text_features: Text features [batch, text_dim]
            knowledge_features: Knowledge features [batch, knowledge_dim]
            
        Returns:
            Fused features [batch, output_dim]
        """
        # Project to common dimension
        v = self.visual_proj(visual_features)
        t = self.text_proj(text_features)
        k = self.knowledge_proj(knowledge_features)
        
        if self.fusion_method == 'trilinear':
            # Trilinear fusion: sum over all (i,j,k) of v_i * t_j * k_k * W_ijk
            # Efficient implementation using einsum
            vt = torch.einsum('bi,bj->bij', v, t)  # [batch, dim, dim]
            vtk = torch.einsum('bij,bk->bijk', vt, k)  # [batch, dim, dim, dim]
            fused = torch.einsum('bijk,ijk->bi', vtk, self.trilinear)
            fused = self.output_layer(fused)
            
        elif self.fusion_method == 'cross_attention':
            # Add sequence dimension
            v_seq = v.unsqueeze(1)
            t_seq = t.unsqueeze(1)
            k_seq = k.unsqueeze(1)
            
            # Cross-modal attention
            vt_attn, _ = self.vt_attention(v_seq, t_seq, t_seq)
            tk_attn, _ = self.tk_attention(t_seq, k_seq, k_seq)
            vk_attn, _ = self.vk_attention(v_seq, k_seq, k_seq)
            
            # Concatenate and project
            combined = torch.cat([
                vt_attn.squeeze(1),
                tk_attn.squeeze(1),
                vk_attn.squeeze(1)
            ], dim=-1)
            fused = self.output_layer(combined)
            
        elif self.fusion_method == 'gated':
            # Concatenate all features
            combined = torch.cat([v, t, k], dim=-1)
            
            # Compute gates
            gate_v = torch.sigmoid(self.gate_visual(combined))
            gate_t = torch.sigmoid(self.gate_text(combined))
            gate_k = torch.sigmoid(self.gate_knowledge(combined))
            
            # Gated combination
            gated = gate_v * v + gate_t * t + gate_k * k
            fused = self.output_layer(gated)
            
        else:  # simple add
            fused = v + t + k
        
        fused = self.layer_norm(fused)
        fused = self.dropout(fused)
        
        return fused


class KnowledgeAugmentedVQA(nn.Module):
    """
    Full Knowledge-Augmented VQA module.
    Combines visual features, question features, and retrieved knowledge.
    """
    
    def __init__(
        self,
        visual_encoder: Any,
        text_encoder: Any,
        rag_module: RAGModule,
        fusion_module: KnowledgeAugmentedFusion,
        answer_head: Optional[nn.Module] = None,
        num_answers: int = 3000
    ):
        """
        Initialize knowledge-augmented VQA.
        
        Args:
            visual_encoder: Visual feature encoder
            text_encoder: Text/question encoder
            rag_module: RAG module for knowledge retrieval
            fusion_module: Fusion module
            answer_head: Answer prediction head
            num_answers: Number of answer classes
        """
        super().__init__()
        
        self.visual_encoder = visual_encoder
        self.text_encoder = text_encoder
        self.rag_module = rag_module
        self.fusion_module = fusion_module
        
        # Answer prediction head
        if answer_head is not None:
            self.answer_head = answer_head
        else:
            self.answer_head = nn.Sequential(
                nn.Linear(fusion_module.output_dim, fusion_module.output_dim),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(fusion_module.output_dim, num_answers)
            )
    
    def forward(
        self,
        images: torch.Tensor,
        questions: List[str],
        question_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:
        """
        Forward pass for knowledge-augmented VQA.
        
        Args:
            images: Image tensors [batch, 3, H, W]
            questions: List of question strings
            question_ids: Tokenized question IDs (optional)
            attention_mask: Attention mask (optional)
            
        Returns:
            Dictionary with predictions and auxiliary outputs
        """
        batch_size = len(questions)
        
        # Encode visual features
        visual_features = self.visual_encoder(images)
        if hasattr(visual_features, 'pooled_output'):
            visual_features = visual_features.pooled_output
        
        # Encode question features
        if question_ids is not None:
            text_features = self.text_encoder(question_ids, attention_mask)
        else:
            text_result = self.text_encoder.encode_batch(questions)
            text_features = text_result.pooled_output
        
        # Retrieve and encode knowledge for each question
        knowledge_features_list = []
        rag_outputs = []
        
        for i, question in enumerate(questions):
            rag_output = self.rag_module(question, text_features[i])
            rag_outputs.append(rag_output)
            knowledge_features_list.append(rag_output.output)
        
        knowledge_features = torch.stack(knowledge_features_list, dim=0)
        
        # Fuse all modalities
        fused = self.fusion_module(
            visual_features,
            text_features,
            knowledge_features
        )
        
        # Predict answers
        logits = self.answer_head(fused)
        
        return {
            'logits': logits,
            'fused_features': fused,
            'visual_features': visual_features,
            'text_features': text_features,
            'knowledge_features': knowledge_features,
            'rag_outputs': rag_outputs
        }


class RAGLoss(nn.Module):
    """
    Loss function for RAG training.
    Combines answer loss with retrieval quality loss.
    """
    
    def __init__(
        self,
        answer_loss_weight: float = 1.0,
        retrieval_loss_weight: float = 0.1,
        diversity_loss_weight: float = 0.01
    ):
        """
        Initialize RAG loss.
        
        Args:
            answer_loss_weight: Weight for answer prediction loss
            retrieval_loss_weight: Weight for retrieval loss
            diversity_loss_weight: Weight for context diversity loss
        """
        super().__init__()
        
        self.answer_loss_weight = answer_loss_weight
        self.retrieval_loss_weight = retrieval_loss_weight
        self.diversity_loss_weight = diversity_loss_weight
        
        self.answer_loss_fn = nn.CrossEntropyLoss()
    
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        attention_weights: Optional[torch.Tensor] = None,
        context_embeddings: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute RAG loss.
        
        Args:
            logits: Answer logits [batch, num_answers]
            targets: Target answer indices [batch]
            attention_weights: Context attention weights
            context_embeddings: Context embeddings for diversity
            
        Returns:
            Dictionary of loss components
        """
        # Answer prediction loss
        answer_loss = self.answer_loss_fn(logits, targets)
        
        total_loss = self.answer_loss_weight * answer_loss
        losses = {'answer_loss': answer_loss}
        
        # Retrieval quality loss (encourage focused attention)
        if attention_weights is not None:
            # Entropy of attention weights (lower is more focused)
            attn_entropy = -torch.sum(
                attention_weights * torch.log(attention_weights + 1e-10),
                dim=-1
            ).mean()
            retrieval_loss = attn_entropy
            total_loss = total_loss + self.retrieval_loss_weight * retrieval_loss
            losses['retrieval_loss'] = retrieval_loss
        
        # Context diversity loss (encourage diverse contexts)
        if context_embeddings is not None and context_embeddings.size(1) > 1:
            # Pairwise cosine similarity
            context_norm = F.normalize(context_embeddings, p=2, dim=-1)
            similarity = torch.bmm(context_norm, context_norm.transpose(1, 2))
            
            # Mask diagonal
            mask = torch.eye(similarity.size(1), device=similarity.device).bool()
            similarity = similarity.masked_fill(mask.unsqueeze(0), 0)
            
            # Encourage diversity (penalize high similarity)
            diversity_loss = similarity.mean()
            total_loss = total_loss + self.diversity_loss_weight * diversity_loss
            losses['diversity_loss'] = diversity_loss
        
        losses['total_loss'] = total_loss
        
        return losses


def create_rag_module(
    retriever: BaseRetriever,
    context_encoder: Any,
    **kwargs
) -> RAGModule:
    """
    Factory function to create RAG module.
    
    Args:
        retriever: Document retriever
        context_encoder: Context encoder
        **kwargs: Additional arguments
        
    Returns:
        RAG module instance
    """
    return RAGModule(
        retriever=retriever,
        context_encoder=context_encoder,
        **kwargs
    )


def create_knowledge_augmented_vqa(
    visual_encoder: Any,
    text_encoder: Any,
    retriever: BaseRetriever,
    context_encoder: Any,
    visual_dim: int = 768,
    text_dim: int = 768,
    knowledge_dim: int = 768,
    output_dim: int = 768,
    num_answers: int = 3000,
    **kwargs
) -> KnowledgeAugmentedVQA:
    """
    Factory function to create knowledge-augmented VQA model.
    
    Args:
        visual_encoder: Visual encoder
        text_encoder: Text encoder
        retriever: Document retriever
        context_encoder: Context encoder
        visual_dim: Visual feature dimension
        text_dim: Text feature dimension
        knowledge_dim: Knowledge feature dimension
        output_dim: Output dimension
        num_answers: Number of answer classes
        **kwargs: Additional arguments
        
    Returns:
        KnowledgeAugmentedVQA instance
    """
    # Create RAG module
    rag_module = RAGModule(
        retriever=retriever,
        context_encoder=context_encoder,
        embed_dim=knowledge_dim
    )
    
    # Create fusion module
    fusion_module = KnowledgeAugmentedFusion(
        visual_dim=visual_dim,
        text_dim=text_dim,
        knowledge_dim=knowledge_dim,
        output_dim=output_dim
    )
    
    return KnowledgeAugmentedVQA(
        visual_encoder=visual_encoder,
        text_encoder=text_encoder,
        rag_module=rag_module,
        fusion_module=fusion_module,
        num_answers=num_answers
    )
