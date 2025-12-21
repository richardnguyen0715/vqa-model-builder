"""
Knowledge Encoder Module
Provides encoding capabilities for text, visual, and multimodal knowledge.
"""

from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any, Union
from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class EncodingResult:
    """
    Container for encoding results.
    
    Attributes:
        embeddings: Encoded embeddings tensor
        attention_mask: Attention mask tensor
        pooled_output: Pooled representation
        hidden_states: All hidden states if requested
        metadata: Additional metadata
    """
    embeddings: torch.Tensor
    attention_mask: Optional[torch.Tensor] = None
    pooled_output: Optional[torch.Tensor] = None
    hidden_states: Optional[List[torch.Tensor]] = None
    metadata: Optional[Dict[str, Any]] = None


class BaseKnowledgeEncoder(nn.Module, ABC):
    """
    Abstract base class for knowledge encoders.
    """
    
    def __init__(self, embed_dim: int):
        """
        Initialize base knowledge encoder.
        
        Args:
            embed_dim: Embedding dimension
        """
        super().__init__()
        self.embed_dim = embed_dim
        
    @abstractmethod
    def encode(
        self,
        inputs: Any,
        return_hidden_states: bool = False
    ) -> EncodingResult:
        """
        Encode inputs to embeddings.
        
        Args:
            inputs: Input data
            return_hidden_states: Whether to return hidden states
            
        Returns:
            Encoding result
        """
        pass
    
    @abstractmethod
    def encode_batch(
        self,
        batch_inputs: List[Any],
        return_hidden_states: bool = False
    ) -> EncodingResult:
        """
        Encode batch of inputs.
        
        Args:
            batch_inputs: List of inputs
            return_hidden_states: Whether to return hidden states
            
        Returns:
            Encoding result
        """
        pass


class TextKnowledgeEncoder(BaseKnowledgeEncoder):
    """
    Text knowledge encoder using transformer models.
    Optimized for Vietnamese with PhoBERT support.
    """
    
    def __init__(
        self,
        model_name: str = 'vinai/phobert-base',
        embed_dim: int = 768,
        max_length: int = 256,
        pooling_strategy: str = 'mean',
        normalize_embeddings: bool = True,
        use_projection: bool = False,
        projection_dim: Optional[int] = None,
        device: Optional[str] = None
    ):
        """
        Initialize text knowledge encoder.
        
        Args:
            model_name: Pretrained model name
            embed_dim: Embedding dimension
            max_length: Maximum sequence length
            pooling_strategy: Pooling strategy (mean, cls, max)
            normalize_embeddings: Whether to normalize embeddings
            use_projection: Whether to use projection layer
            projection_dim: Projection dimension
            device: Device to use
        """
        super().__init__(embed_dim)
        
        self.model_name = model_name
        self.max_length = max_length
        self.pooling_strategy = pooling_strategy
        self.normalize_embeddings = normalize_embeddings
        self.use_projection = use_projection
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize model and tokenizer
        self._init_model()
        
        # Projection layer
        if use_projection and projection_dim:
            self.projection = nn.Linear(embed_dim, projection_dim)
            self.embed_dim = projection_dim
        else:
            self.projection = None
    
    def _init_model(self):
        """Initialize transformer model and tokenizer."""
        try:
            from transformers import AutoModel, AutoTokenizer
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
            
        except ImportError:
            raise ImportError(
                "transformers library required. "
                "Install with: pip install transformers"
            )
    
    def _pool_embeddings(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Pool hidden states to get sentence embedding.
        
        Args:
            hidden_states: Hidden states tensor [batch, seq, hidden]
            attention_mask: Attention mask tensor [batch, seq]
            
        Returns:
            Pooled embeddings [batch, hidden]
        """
        if self.pooling_strategy == 'cls':
            return hidden_states[:, 0, :]
            
        elif self.pooling_strategy == 'max':
            # Mask padding tokens
            attention_mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size())
            hidden_states_masked = hidden_states.masked_fill(attention_mask_expanded == 0, -1e9)
            return hidden_states_masked.max(dim=1)[0]
            
        else:  # mean pooling
            attention_mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
            sum_embeddings = torch.sum(hidden_states * attention_mask_expanded, dim=1)
            sum_mask = torch.clamp(attention_mask_expanded.sum(dim=1), min=1e-9)
            return sum_embeddings / sum_mask
    
    def encode(
        self,
        text: str,
        return_hidden_states: bool = False
    ) -> EncodingResult:
        """
        Encode single text to embedding.
        
        Args:
            text: Input text
            return_hidden_states: Whether to return hidden states
            
        Returns:
            Encoding result
        """
        return self.encode_batch([text], return_hidden_states)
    
    def encode_batch(
        self,
        texts: List[str],
        return_hidden_states: bool = False
    ) -> EncodingResult:
        """
        Encode batch of texts.
        
        Args:
            texts: List of texts
            return_hidden_states: Whether to return hidden states
            
        Returns:
            Encoding result
        """
        # Tokenize
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        input_ids = encoded['input_ids'].to(self.device)
        attention_mask = encoded['attention_mask'].to(self.device)
        
        # Encode
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=return_hidden_states
            )
        
        # Pool
        hidden_states = outputs.last_hidden_state
        pooled = self._pool_embeddings(hidden_states, attention_mask)
        
        # Project if needed
        if self.projection is not None:
            pooled = self.projection(pooled)
        
        # Normalize
        if self.normalize_embeddings:
            pooled = nn.functional.normalize(pooled, p=2, dim=-1)
        
        return EncodingResult(
            embeddings=hidden_states,
            attention_mask=attention_mask,
            pooled_output=pooled,
            hidden_states=outputs.hidden_states if return_hidden_states else None
        )
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass for integration with other modules.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            
        Returns:
            Pooled embeddings
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        pooled = self._pool_embeddings(outputs.last_hidden_state, attention_mask)
        
        if self.projection is not None:
            pooled = self.projection(pooled)
            
        if self.normalize_embeddings:
            pooled = nn.functional.normalize(pooled, p=2, dim=-1)
            
        return pooled


class VisualKnowledgeEncoder(BaseKnowledgeEncoder):
    """
    Visual knowledge encoder using vision transformer models.
    """
    
    def __init__(
        self,
        model_name: str = 'openai/clip-vit-base-patch32',
        embed_dim: int = 768,
        pooling_strategy: str = 'cls',
        normalize_embeddings: bool = True,
        use_projection: bool = False,
        projection_dim: Optional[int] = None,
        device: Optional[str] = None
    ):
        """
        Initialize visual knowledge encoder.
        
        Args:
            model_name: Pretrained model name
            embed_dim: Embedding dimension
            pooling_strategy: Pooling strategy (cls, mean, spatial)
            normalize_embeddings: Whether to normalize embeddings
            use_projection: Whether to use projection layer
            projection_dim: Projection dimension
            device: Device to use
        """
        super().__init__(embed_dim)
        
        self.model_name = model_name
        self.pooling_strategy = pooling_strategy
        self.normalize_embeddings = normalize_embeddings
        self.use_projection = use_projection
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize model
        self._init_model()
        
        # Projection layer
        if use_projection and projection_dim:
            self.projection = nn.Linear(embed_dim, projection_dim)
            self.embed_dim = projection_dim
        else:
            self.projection = None
    
    def _init_model(self):
        """Initialize vision model and processor."""
        try:
            from transformers import AutoModel, AutoImageProcessor
            
            self.processor = AutoImageProcessor.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
            
        except ImportError:
            raise ImportError(
                "transformers library required. "
                "Install with: pip install transformers"
            )
    
    def _pool_embeddings(
        self,
        hidden_states: torch.Tensor
    ) -> torch.Tensor:
        """
        Pool hidden states from vision model.
        
        Args:
            hidden_states: Hidden states tensor [batch, seq, hidden]
            
        Returns:
            Pooled embeddings [batch, hidden]
        """
        if self.pooling_strategy == 'cls':
            return hidden_states[:, 0, :]
            
        elif self.pooling_strategy == 'mean':
            return hidden_states.mean(dim=1)
            
        else:  # spatial - return all spatial features
            return hidden_states[:, 1:, :]  # Exclude CLS token
    
    def encode(
        self,
        image: Any,
        return_hidden_states: bool = False
    ) -> EncodingResult:
        """
        Encode single image.
        
        Args:
            image: Input image (PIL Image or tensor)
            return_hidden_states: Whether to return hidden states
            
        Returns:
            Encoding result
        """
        return self.encode_batch([image], return_hidden_states)
    
    def encode_batch(
        self,
        images: List[Any],
        return_hidden_states: bool = False
    ) -> EncodingResult:
        """
        Encode batch of images.
        
        Args:
            images: List of images
            return_hidden_states: Whether to return hidden states
            
        Returns:
            Encoding result
        """
        # Process images
        inputs = self.processor(images=images, return_tensors='pt')
        pixel_values = inputs['pixel_values'].to(self.device)
        
        # Encode
        with torch.no_grad():
            outputs = self.model(
                pixel_values=pixel_values,
                output_hidden_states=return_hidden_states
            )
        
        # Get hidden states
        if hasattr(outputs, 'last_hidden_state'):
            hidden_states = outputs.last_hidden_state
        else:
            hidden_states = outputs.image_embeds if hasattr(outputs, 'image_embeds') else outputs[0]
        
        # Pool
        if hidden_states.dim() == 3:
            pooled = self._pool_embeddings(hidden_states)
        else:
            pooled = hidden_states
        
        # Project if needed
        if self.projection is not None:
            pooled = self.projection(pooled)
        
        # Normalize
        if self.normalize_embeddings and pooled.dim() == 2:
            pooled = nn.functional.normalize(pooled, p=2, dim=-1)
        
        return EncodingResult(
            embeddings=hidden_states,
            pooled_output=pooled,
            hidden_states=outputs.hidden_states if return_hidden_states and hasattr(outputs, 'hidden_states') else None
        )
    
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for integration with other modules.
        
        Args:
            pixel_values: Input pixel values [batch, 3, H, W]
            
        Returns:
            Pooled embeddings
        """
        outputs = self.model(pixel_values=pixel_values)
        
        if hasattr(outputs, 'last_hidden_state'):
            hidden_states = outputs.last_hidden_state
        else:
            hidden_states = outputs.image_embeds if hasattr(outputs, 'image_embeds') else outputs[0]
        
        if hidden_states.dim() == 3:
            pooled = self._pool_embeddings(hidden_states)
        else:
            pooled = hidden_states
        
        if self.projection is not None:
            pooled = self.projection(pooled)
            
        if self.normalize_embeddings and pooled.dim() == 2:
            pooled = nn.functional.normalize(pooled, p=2, dim=-1)
            
        return pooled


class MultimodalKnowledgeEncoder(BaseKnowledgeEncoder):
    """
    Multimodal knowledge encoder combining text and visual encoders.
    """
    
    def __init__(
        self,
        text_encoder: Optional[TextKnowledgeEncoder] = None,
        visual_encoder: Optional[VisualKnowledgeEncoder] = None,
        embed_dim: int = 768,
        fusion_method: str = 'concat',
        normalize_embeddings: bool = True,
        device: Optional[str] = None
    ):
        """
        Initialize multimodal knowledge encoder.
        
        Args:
            text_encoder: Text encoder module
            visual_encoder: Visual encoder module
            embed_dim: Output embedding dimension
            fusion_method: Fusion method (concat, add, multiply, attention)
            normalize_embeddings: Whether to normalize embeddings
            device: Device to use
        """
        super().__init__(embed_dim)
        
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.fusion_method = fusion_method
        self.normalize_embeddings = normalize_embeddings
        
        # Initialize encoders
        self.text_encoder = text_encoder or TextKnowledgeEncoder(device=self.device)
        self.visual_encoder = visual_encoder or VisualKnowledgeEncoder(device=self.device)
        
        # Get encoder dimensions
        text_dim = self.text_encoder.embed_dim
        visual_dim = self.visual_encoder.embed_dim
        
        # Fusion layers
        if fusion_method == 'concat':
            self.projection = nn.Linear(text_dim + visual_dim, embed_dim)
        elif fusion_method == 'attention':
            self.cross_attention = nn.MultiheadAttention(
                embed_dim=max(text_dim, visual_dim),
                num_heads=8,
                batch_first=True
            )
            self.projection = nn.Linear(max(text_dim, visual_dim), embed_dim)
            
            # Align dimensions if needed
            if text_dim != visual_dim:
                self.text_proj = nn.Linear(text_dim, max(text_dim, visual_dim))
                self.visual_proj = nn.Linear(visual_dim, max(text_dim, visual_dim))
            else:
                self.text_proj = None
                self.visual_proj = None
        else:
            # For add/multiply, align dimensions
            if text_dim != visual_dim:
                self.text_proj = nn.Linear(text_dim, embed_dim)
                self.visual_proj = nn.Linear(visual_dim, embed_dim)
            else:
                self.text_proj = None
                self.visual_proj = None
            self.projection = None
    
    def encode(
        self,
        text: Optional[str] = None,
        image: Optional[Any] = None,
        return_hidden_states: bool = False
    ) -> EncodingResult:
        """
        Encode text and/or image.
        
        Args:
            text: Input text
            image: Input image
            return_hidden_states: Whether to return hidden states
            
        Returns:
            Encoding result
        """
        text_list = [text] if text else None
        image_list = [image] if image else None
        return self.encode_batch(text_list, image_list, return_hidden_states)
    
    def encode_batch(
        self,
        texts: Optional[List[str]] = None,
        images: Optional[List[Any]] = None,
        return_hidden_states: bool = False
    ) -> EncodingResult:
        """
        Encode batch of texts and images.
        
        Args:
            texts: List of texts
            images: List of images
            return_hidden_states: Whether to return hidden states
            
        Returns:
            Encoding result
        """
        text_pooled = None
        visual_pooled = None
        
        # Encode text
        if texts:
            text_result = self.text_encoder.encode_batch(texts, return_hidden_states)
            text_pooled = text_result.pooled_output
        
        # Encode images
        if images:
            visual_result = self.visual_encoder.encode_batch(images, return_hidden_states)
            visual_pooled = visual_result.pooled_output
        
        # Handle single modality
        if text_pooled is None:
            pooled = visual_pooled
        elif visual_pooled is None:
            pooled = text_pooled
        else:
            # Fuse modalities
            pooled = self._fuse(text_pooled, visual_pooled)
        
        # Normalize
        if self.normalize_embeddings:
            pooled = nn.functional.normalize(pooled, p=2, dim=-1)
        
        return EncodingResult(
            embeddings=pooled,
            pooled_output=pooled
        )
    
    def _fuse(
        self,
        text_emb: torch.Tensor,
        visual_emb: torch.Tensor
    ) -> torch.Tensor:
        """
        Fuse text and visual embeddings.
        
        Args:
            text_emb: Text embeddings [batch, dim]
            visual_emb: Visual embeddings [batch, dim]
            
        Returns:
            Fused embeddings
        """
        if self.fusion_method == 'concat':
            fused = torch.cat([text_emb, visual_emb], dim=-1)
            return self.projection(fused)
            
        elif self.fusion_method == 'add':
            if self.text_proj:
                text_emb = self.text_proj(text_emb)
                visual_emb = self.visual_proj(visual_emb)
            return text_emb + visual_emb
            
        elif self.fusion_method == 'multiply':
            if self.text_proj:
                text_emb = self.text_proj(text_emb)
                visual_emb = self.visual_proj(visual_emb)
            return text_emb * visual_emb
            
        elif self.fusion_method == 'attention':
            if self.text_proj:
                text_emb = self.text_proj(text_emb)
                visual_emb = self.visual_proj(visual_emb)
            
            # Add sequence dimension
            text_emb = text_emb.unsqueeze(1)
            visual_emb = visual_emb.unsqueeze(1)
            
            # Cross attention
            attended, _ = self.cross_attention(text_emb, visual_emb, visual_emb)
            attended = attended.squeeze(1)
            
            return self.projection(attended)
        
        else:
            raise ValueError(f"Unknown fusion method: {self.fusion_method}")
    
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass for integration with other modules.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            pixel_values: Input pixel values
            
        Returns:
            Fused embeddings
        """
        text_pooled = None
        visual_pooled = None
        
        if input_ids is not None:
            text_pooled = self.text_encoder(input_ids, attention_mask)
        
        if pixel_values is not None:
            visual_pooled = self.visual_encoder(pixel_values)
        
        if text_pooled is None:
            return visual_pooled
        elif visual_pooled is None:
            return text_pooled
        else:
            return self._fuse(text_pooled, visual_pooled)


def create_text_encoder(
    model_name: str = 'vinai/phobert-base',
    **kwargs
) -> TextKnowledgeEncoder:
    """
    Factory function to create text encoder.
    
    Args:
        model_name: Model name
        **kwargs: Additional arguments
        
    Returns:
        Text encoder instance
    """
    return TextKnowledgeEncoder(model_name=model_name, **kwargs)


def create_visual_encoder(
    model_name: str = 'openai/clip-vit-base-patch32',
    **kwargs
) -> VisualKnowledgeEncoder:
    """
    Factory function to create visual encoder.
    
    Args:
        model_name: Model name
        **kwargs: Additional arguments
        
    Returns:
        Visual encoder instance
    """
    return VisualKnowledgeEncoder(model_name=model_name, **kwargs)


def create_multimodal_encoder(
    text_model: str = 'vinai/phobert-base',
    visual_model: str = 'openai/clip-vit-base-patch32',
    **kwargs
) -> MultimodalKnowledgeEncoder:
    """
    Factory function to create multimodal encoder.
    
    Args:
        text_model: Text model name
        visual_model: Visual model name
        **kwargs: Additional arguments
        
    Returns:
        Multimodal encoder instance
    """
    text_encoder = create_text_encoder(text_model)
    visual_encoder = create_visual_encoder(visual_model)
    
    return MultimodalKnowledgeEncoder(
        text_encoder=text_encoder,
        visual_encoder=visual_encoder,
        **kwargs
    )
