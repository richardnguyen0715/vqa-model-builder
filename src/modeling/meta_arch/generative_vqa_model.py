"""
Generative VQA Model
Encoder-Decoder architecture for Vietnamese Visual Question Answering.
Generates answer tokens instead of classifying into fixed vocabulary.
"""

import math
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoModel, 
    AutoTokenizer,
    AutoModelForCausalLM,
    GPT2LMHeadModel,
    GPT2Config
)

# Import MOE components
from src.modeling.moe.moe_layer import MOELayer, VQAMOELayer, SparseMOELayer
from src.modeling.moe.moe_config import MOEConfig


@dataclass
class GenerativeVQAConfig:
    """Configuration for Generative VQA Model."""
    
    # Visual Encoder
    visual_backbone: str = 'openai/clip-vit-base-patch32'
    visual_output_dim: int = 768
    freeze_visual_encoder: bool = False
    freeze_visual: bool = False  # Alias
    
    # Text Encoder (for question)
    text_encoder: str = 'vinai/phobert-base'
    text_output_dim: int = 768
    freeze_question_encoder: bool = False
    freeze_text_encoder: bool = False  # Alias
    max_question_length: int = 64
    
    # Decoder (for answer generation)
    decoder_type: str = 'transformer'  # 'transformer', 'gpt2', 'phobert'
    hidden_size: int = 768  # Main dimension
    decoder_hidden_dim: int = 768  # Alias
    num_decoder_layers: int = 6
    decoder_num_layers: int = 6  # Alias
    num_attention_heads: int = 8
    decoder_num_heads: int = 8  # Alias
    decoder_ff_dim: int = 2048
    decoder_dropout: float = 0.1
    max_answer_length: int = 64
    
    # Fusion
    fusion_dim: int = 768
    fusion_num_heads: int = 8
    fusion_num_layers: int = 2
    fusion_dropout: float = 0.1
    
    # MOE (Mixture of Experts)
    use_moe: bool = False
    moe_type: str = 'standard'  # 'standard', 'vqa', 'sparse'
    num_experts: int = 4
    num_experts_per_token: int = 2
    expert_capacity_factor: float = 1.25
    moe_loss_weight: float = 0.01
    moe_position: str = 'fusion'  # 'fusion', 'decoder', 'both'
    
    # VQA MOE specific (when moe_type='vqa')
    num_vision_experts: int = 1
    num_text_experts: int = 1
    num_multimodal_experts: int = 1
    num_specialized_experts: int = 1  # SegmentationExpert, ObjectDetectionExpert, OCRExpert, etc.
    vietnamese_optimized: bool = True
    
    # Generation
    vocab_size: int = 64000  # PhoBERT vocab size
    pad_token_id: int = 1
    bos_token_id: int = 0
    eos_token_id: int = 2
    
    # Training
    label_smoothing: float = 0.1
    tie_word_embeddings: bool = True
    
    def __post_init__(self):
        """Sync aliased fields."""
        # Sync freeze fields - check both directions
        if self.freeze_visual_encoder:
            self.freeze_visual = True
        if self.freeze_visual:
            self.freeze_visual_encoder = True
            
        if self.freeze_question_encoder:
            self.freeze_text_encoder = True
        if self.freeze_text_encoder:
            self.freeze_question_encoder = True
        
        # Sync decoder dimension fields
        self.decoder_hidden_dim = self.hidden_size
        self.decoder_num_layers = self.num_decoder_layers
        self.decoder_num_heads = self.num_attention_heads


@dataclass
class GenerativeVQAOutput:
    """Output from Generative VQA Model."""
    
    logits: torch.Tensor  # [batch, seq_len, vocab_size]
    loss: Optional[torch.Tensor] = None
    generated_ids: Optional[torch.Tensor] = None  # [batch, generated_len]
    encoder_hidden_states: Optional[torch.Tensor] = None
    decoder_hidden_states: Optional[torch.Tensor] = None
    cross_attention_weights: Optional[torch.Tensor] = None


class VisualEncoder(nn.Module):
    """Visual encoder using CLIP ViT or other vision transformers."""
    
    def __init__(self, config: GenerativeVQAConfig):
        super().__init__()
        self.config = config
        
        # Load CLIP vision model
        from transformers import CLIPVisionModel
        self.vision_model = CLIPVisionModel.from_pretrained(config.visual_backbone)
        
        if config.freeze_visual:
            for param in self.vision_model.parameters():
                param.requires_grad = False
        
        # Project to fusion dimension if needed
        vision_hidden_size = self.vision_model.config.hidden_size
        if vision_hidden_size != config.fusion_dim:
            self.projection = nn.Linear(vision_hidden_size, config.fusion_dim)
        else:
            self.projection = nn.Identity()
    
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pixel_values: [batch, 3, H, W]
        Returns:
            visual_features: [batch, num_patches+1, fusion_dim]
        """
        outputs = self.vision_model(pixel_values)
        # Get all hidden states including CLS token
        hidden_states = outputs.last_hidden_state  # [batch, num_patches+1, hidden]
        return self.projection(hidden_states)


class QuestionEncoder(nn.Module):
    """Question encoder using PhoBERT."""
    
    def __init__(self, config: GenerativeVQAConfig):
        super().__init__()
        self.config = config
        
        # Load PhoBERT
        self.encoder = AutoModel.from_pretrained(config.text_encoder)
        
        if config.freeze_text_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
        
        # Project to fusion dimension if needed
        encoder_hidden_size = self.encoder.config.hidden_size
        if encoder_hidden_size != config.fusion_dim:
            self.projection = nn.Linear(encoder_hidden_size, config.fusion_dim)
        else:
            self.projection = nn.Identity()
    
    def forward(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            input_ids: [batch, seq_len]
            attention_mask: [batch, seq_len]
        Returns:
            question_features: [batch, seq_len, fusion_dim]
            attention_mask: [batch, seq_len]
        """
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state
        return self.projection(hidden_states), attention_mask


class CrossModalFusion(nn.Module):
    """Cross-modal fusion between visual and text features with optional MOE."""
    
    def __init__(self, config: GenerativeVQAConfig):
        super().__init__()
        self.config = config
        self.use_moe = config.use_moe and config.moe_position in ['fusion', 'both']
        self.moe_type = config.moe_type if hasattr(config, 'moe_type') else 'standard'
        
        # Cross-attention layers
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=config.fusion_dim,
                nhead=config.fusion_num_heads,
                dim_feedforward=config.decoder_ff_dim,
                dropout=config.fusion_dropout,
                activation='gelu',
                batch_first=True,
                norm_first=True
            )
            for _ in range(config.fusion_num_layers)
        ])
        
        # MOE layer (optional) - replaces FFN in fusion
        self.moe_layer = None
        self.moe_aux_loss = 0.0
        if self.use_moe:
            self._create_moe_layer(config)
        
        self.layer_norm = nn.LayerNorm(config.fusion_dim)
    
    def _create_moe_layer(self, config: GenerativeVQAConfig):
        """Create MOE layer based on configuration."""
        from src.modeling.moe.moe_config import RouterConfig
        
        if self.moe_type == 'vqa':
            # Use VQAMOELayer with specialized experts
            self.moe_layer = VQAMOELayer(
                input_dim=config.fusion_dim,
                hidden_dim=config.decoder_ff_dim,
                output_dim=config.fusion_dim,
                num_vision_experts=getattr(config, 'num_vision_experts', 1),
                num_text_experts=getattr(config, 'num_text_experts', 1),
                num_multimodal_experts=getattr(config, 'num_multimodal_experts', 1),
                num_specialized_experts=getattr(config, 'num_specialized_experts', 1),
                top_k=config.num_experts_per_token,
                dropout=config.fusion_dropout,
                vietnamese_optimized=getattr(config, 'vietnamese_optimized', True)
            )
        elif self.moe_type == 'sparse':
            # Use SparseMOELayer for memory efficiency
            router_config = RouterConfig(
                router_type='topk',
                num_experts=config.num_experts,
                top_k=config.num_experts_per_token,
                capacity_factor=config.expert_capacity_factor,
                load_balance_weight=config.moe_loss_weight,
                use_aux_loss=True
            )
            moe_config = MOEConfig(
                input_dim=config.fusion_dim,
                hidden_dim=config.decoder_ff_dim,
                output_dim=config.fusion_dim,
                num_experts=config.num_experts,
                num_experts_per_token=config.num_experts_per_token,
                router_config=router_config,
                expert_dropout=config.fusion_dropout,
                use_sparse_moe=True
            )
            self.moe_layer = SparseMOELayer(config=moe_config)
        else:
            # Standard MOELayer with FeedForwardExperts
            router_config = RouterConfig(
                router_type='topk',
                num_experts=config.num_experts,
                top_k=config.num_experts_per_token,
                capacity_factor=config.expert_capacity_factor,
                load_balance_weight=config.moe_loss_weight,
                use_aux_loss=True
            )
            moe_config = MOEConfig(
                input_dim=config.fusion_dim,
                hidden_dim=config.decoder_ff_dim,
                output_dim=config.fusion_dim,
                num_experts=config.num_experts,
                num_experts_per_token=config.num_experts_per_token,
                router_config=router_config,
                expert_dropout=config.fusion_dropout
            )
            self.moe_layer = MOELayer(config=moe_config)
        
        self.layer_norm = nn.LayerNorm(config.fusion_dim)
    
    def forward(
        self,
        visual_features: torch.Tensor,
        question_features: torch.Tensor,
        question_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, float]:
        """
        Fuse visual and question features.
        
        Args:
            visual_features: [batch, num_patches, dim]
            question_features: [batch, seq_len, dim]
            question_mask: [batch, seq_len]
            
        Returns:
            fused_features: [batch, num_patches + seq_len, dim]
            moe_aux_loss: Auxiliary loss from MOE layer (0.0 if MOE not used)
        """
        # Concatenate visual and question features
        # Visual features come first, then question
        fused = torch.cat([visual_features, question_features], dim=1)
        
        # Create attention mask if provided
        if question_mask is not None:
            batch_size = visual_features.size(0)
            num_visual = visual_features.size(1)
            
            # Visual tokens are always attended to (mask = False means attend)
            visual_mask = torch.zeros(batch_size, num_visual, dtype=torch.bool, device=fused.device)
            # Invert question mask (1 = attend -> False, 0 = pad -> True)
            question_pad_mask = ~question_mask.bool()
            
            # Combine masks
            src_key_padding_mask = torch.cat([visual_mask, question_pad_mask], dim=1)
        else:
            src_key_padding_mask = None
        
        # Apply transformer layers
        for layer in self.layers:
            fused = layer(fused, src_key_padding_mask=src_key_padding_mask)
        
        # Apply MOE layer if enabled
        moe_aux_loss = 0.0
        if self.moe_layer is not None:
            # MOELayer.forward() returns tensor directly
            fused = self.moe_layer(fused)
            # Get auxiliary loss for load balancing
            aux_loss = self.moe_layer.get_aux_loss()
            if isinstance(aux_loss, torch.Tensor):
                moe_aux_loss = aux_loss.item() if aux_loss.numel() == 1 else aux_loss.mean().item()
            else:
                moe_aux_loss = float(aux_loss) if aux_loss else 0.0
        
        return self.layer_norm(fused), moe_aux_loss


class TransformerDecoder(nn.Module):
    """Transformer decoder for answer generation."""
    
    def __init__(self, config: GenerativeVQAConfig, embedding: Optional[nn.Embedding] = None):
        super().__init__()
        self.config = config
        
        # Token embedding
        if embedding is not None:
            self.embedding = embedding
        else:
            self.embedding = nn.Embedding(config.vocab_size, config.decoder_hidden_dim)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(
            config.decoder_hidden_dim, 
            config.decoder_dropout,
            max_len=config.max_answer_length
        )
        
        # Decoder layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.decoder_hidden_dim,
            nhead=config.decoder_num_heads,
            dim_feedforward=config.decoder_ff_dim,
            dropout=config.decoder_dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=config.decoder_num_layers)
        
        self.layer_norm = nn.LayerNorm(config.decoder_hidden_dim)
        
        # Output projection
        self.output_projection = nn.Linear(config.decoder_hidden_dim, config.vocab_size, bias=False)
        
        # Tie weights
        if config.tie_word_embeddings:
            self.output_projection.weight = self.embedding.weight
    
    def forward(
        self,
        encoder_hidden_states: torch.Tensor,
        decoder_input_ids: torch.Tensor,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        decoder_attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            encoder_hidden_states: [batch, enc_seq, dim]
            decoder_input_ids: [batch, dec_seq]
            encoder_attention_mask: [batch, enc_seq] - 1 = attend, 0 = mask
            decoder_attention_mask: [batch, dec_seq]
            
        Returns:
            logits: [batch, dec_seq, vocab_size]
        """
        # Embed decoder inputs
        decoder_embeds = self.embedding(decoder_input_ids)
        decoder_embeds = self.pos_encoding(decoder_embeds)
        
        # Create causal mask for decoder self-attention
        seq_len = decoder_input_ids.size(1)
        causal_mask = self._generate_causal_mask(seq_len, decoder_input_ids.device)
        
        # Create padding mask for encoder
        if encoder_attention_mask is not None:
            # Invert: 1 = attend -> False, 0 = pad -> True
            memory_key_padding_mask = ~encoder_attention_mask.bool()
        else:
            memory_key_padding_mask = None
        
        # Create padding mask for decoder
        if decoder_attention_mask is not None:
            tgt_key_padding_mask = ~decoder_attention_mask.bool()
        else:
            tgt_key_padding_mask = None
        
        # Decode
        decoder_output = self.decoder(
            tgt=decoder_embeds,
            memory=encoder_hidden_states,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask
        )
        
        decoder_output = self.layer_norm(decoder_output)
        
        # Project to vocabulary
        logits = self.output_projection(decoder_output)
        
        return logits
    
    def _generate_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Generate causal attention mask."""
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 512):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, d_model]
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class GenerativeVQAModel(nn.Module):
    """
    Generative Visual Question Answering Model.
    
    Architecture:
        1. Visual Encoder (CLIP ViT) → visual features
        2. Question Encoder (PhoBERT) → question features  
        3. Cross-Modal Fusion → fused multimodal context
        4. Transformer Decoder → generate answer tokens
    """
    
    def __init__(self, config: GenerativeVQAConfig):
        super().__init__()
        self.config = config
        
        # Encoders
        self.visual_encoder = VisualEncoder(config)
        self.question_encoder = QuestionEncoder(config)
        
        # Fusion
        self.fusion = CrossModalFusion(config)
        
        # Answer embedding (shared with decoder output)
        self.answer_embedding = nn.Embedding(config.vocab_size, config.decoder_hidden_dim)
        
        # Decoder
        self.decoder = TransformerDecoder(config, embedding=self.answer_embedding)
        
        # Loss with ignore_index=-100 (standard for language modeling)
        self.loss_fn = nn.CrossEntropyLoss(
            ignore_index=-100,  # Standard ignore index for padding in labels
            label_smoothing=config.label_smoothing
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for non-pretrained components."""
        for module in [self.fusion, self.decoder]:
            for p in module.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
    
    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        decoder_input_ids: Optional[torch.Tensor] = None,
        decoder_attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_dict: bool = True
    ) -> GenerativeVQAOutput:
        """
        Forward pass for training or generation.
        
        Args:
            pixel_values: Images [batch, 3, H, W]
            input_ids: Question token IDs [batch, q_seq]
            attention_mask: Question attention mask [batch, q_seq]
            decoder_input_ids: Answer tokens for teacher forcing [batch, a_seq]
            decoder_attention_mask: Answer attention mask [batch, a_seq]
            labels: Target answer tokens [batch, a_seq] (shifted right from decoder_input_ids)
            
        Returns:
            GenerativeVQAOutput
        """
        # Encode visual features
        visual_features = self.visual_encoder(pixel_values)
        
        # Encode question
        question_features, question_mask = self.question_encoder(input_ids, attention_mask)
        
        # Fuse multimodal features (returns tuple with MOE aux loss)
        encoder_hidden_states, moe_aux_loss = self.fusion(visual_features, question_features, question_mask)
        
        # Create encoder attention mask (all ones for visual + question mask)
        batch_size = pixel_values.size(0)
        num_visual = visual_features.size(1)
        visual_mask = torch.ones(batch_size, num_visual, device=pixel_values.device)
        encoder_attention_mask = torch.cat([visual_mask, attention_mask.float()], dim=1)
        
        # Decode
        if decoder_input_ids is None:
            # For generation: start with BOS token
            decoder_input_ids = torch.full(
                (batch_size, 1),
                self.config.bos_token_id,
                dtype=torch.long,
                device=pixel_values.device
            )
        
        logits = self.decoder(
            encoder_hidden_states=encoder_hidden_states,
            decoder_input_ids=decoder_input_ids,
            encoder_attention_mask=encoder_attention_mask,
            decoder_attention_mask=decoder_attention_mask
        )
        
        # Compute loss if labels provided
        loss = None
        if labels is not None:
            # Shift logits and labels for loss computation
            # logits: [batch, seq, vocab] -> [batch * seq, vocab]
            # labels: [batch, seq] -> [batch * seq]
            shift_logits = logits.contiguous().view(-1, self.config.vocab_size)
            shift_labels = labels.contiguous().view(-1)
            loss = self.loss_fn(shift_logits, shift_labels)
            
            # Add MOE auxiliary loss (load balancing)
            if self.config.use_moe and moe_aux_loss > 0:
                loss = loss + self.config.moe_loss_weight * moe_aux_loss
        
        return GenerativeVQAOutput(
            logits=logits,
            loss=loss,
            encoder_hidden_states=encoder_hidden_states
        )
    
    @torch.no_grad()
    def generate(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        max_length: int = 64,
        min_length: int = 1,
        num_beams: int = 1,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.95,
        do_sample: bool = False,
        early_stopping: bool = True
    ) -> torch.Tensor:
        """
        Generate answer tokens autoregressively.
        
        Args:
            pixel_values: Images [batch, 3, H, W]
            input_ids: Question tokens [batch, seq]
            attention_mask: Question mask [batch, seq]
            max_length: Maximum generation length
            num_beams: Number of beams for beam search
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Nucleus sampling threshold
            do_sample: Whether to sample or greedy decode
            
        Returns:
            generated_ids: [batch, generated_length]
        """
        batch_size = pixel_values.size(0)
        device = pixel_values.device
        
        # Encode
        visual_features = self.visual_encoder(pixel_values)
        question_features, question_mask = self.question_encoder(input_ids, attention_mask)
        # fusion returns (output, moe_aux_loss) tuple
        encoder_hidden_states, _ = self.fusion(visual_features, question_features, question_mask)
        
        # Encoder attention mask
        num_visual = visual_features.size(1)
        visual_mask = torch.ones(batch_size, num_visual, device=device)
        encoder_attention_mask = torch.cat([visual_mask, attention_mask.float()], dim=1)
        
        # Initialize with BOS token
        generated = torch.full(
            (batch_size, 1),
            self.config.bos_token_id,
            dtype=torch.long,
            device=device
        )
        
        # Track which sequences are finished
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        
        for _ in range(max_length - 1):
            # Get logits for next token
            logits = self.decoder(
                encoder_hidden_states=encoder_hidden_states,
                decoder_input_ids=generated,
                encoder_attention_mask=encoder_attention_mask
            )
            
            # Get logits for last position
            next_token_logits = logits[:, -1, :] / temperature
            
            # Apply top-k filtering
            if top_k > 0:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = float('-inf')
            
            # Apply top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                next_token_logits[indices_to_remove] = float('-inf')
            
            # Sample or greedy
            if do_sample:
                probs = F.softmax(next_token_logits, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)
            else:
                next_tokens = next_token_logits.argmax(dim=-1)
            
            # Replace finished sequences with pad
            next_tokens = next_tokens.masked_fill(finished, self.config.pad_token_id)
            
            # Append to generated
            generated = torch.cat([generated, next_tokens.unsqueeze(-1)], dim=1)
            
            # Check for EOS
            finished = finished | (next_tokens == self.config.eos_token_id)
            
            if early_stopping and finished.all():
                break
        
        return generated


def create_generative_vqa_model(
    config: Optional[GenerativeVQAConfig] = None,
    **kwargs
) -> GenerativeVQAModel:
    """
    Factory function to create Generative VQA model.
    
    Args:
        config: Model configuration
        **kwargs: Override config parameters
        
    Returns:
        GenerativeVQAModel instance
    """
    if config is None:
        config = GenerativeVQAConfig()
    
    # Override with kwargs
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    return GenerativeVQAModel(config)


def get_default_generative_vqa_config(
    visual_backbone: str = 'openai/clip-vit-base-patch32',
    text_encoder: str = 'vinai/phobert-base',
    vocab_size: int = 64000,
    bos_token_id: int = 0,
    eos_token_id: int = 2,
    pad_token_id: int = 1,
    **kwargs
) -> GenerativeVQAConfig:
    """Get default configuration optimized for Vietnamese VQA.
    
    Args:
        visual_backbone: Visual encoder model name
        text_encoder: Text encoder model name
        vocab_size: Vocabulary size
        bos_token_id: Beginning of sequence token ID
        eos_token_id: End of sequence token ID
        pad_token_id: Padding token ID
        **kwargs: Additional config overrides including:
            - use_moe: Enable MOE (default: False)
            - moe_type: Type of MOE ('standard', 'vqa', 'sparse')
            - num_experts: Number of experts for standard MOE (default: 4)
            - num_experts_per_token: Experts per token (default: 2)
            - expert_capacity_factor: Capacity factor (default: 1.25)
            - moe_loss_weight: Load balancing loss weight (default: 0.01)
            - moe_position: Where to apply MOE ('fusion', 'decoder', 'both')
            - num_vision_experts: VQA MOE vision experts (default: 2)
            - num_text_experts: VQA MOE text experts (default: 2)
            - num_multimodal_experts: VQA MOE multimodal experts (default: 2)
            - num_specialized_experts: VQA MOE specialized experts (default: 2)
            - vietnamese_optimized: Enable Vietnamese optimization (default: True)
    """
    config = GenerativeVQAConfig(
        visual_backbone=visual_backbone,
        visual_output_dim=768,
        freeze_visual_encoder=kwargs.get('freeze_visual_encoder', False),
        
        text_encoder=text_encoder,
        text_output_dim=768,
        freeze_question_encoder=kwargs.get('freeze_question_encoder', False),
        max_question_length=64,
        
        decoder_type='transformer',
        hidden_size=768,
        num_decoder_layers=6,
        num_attention_heads=8,
        decoder_ff_dim=2048,
        decoder_dropout=0.1,
        max_answer_length=kwargs.get('max_answer_length', 64),
        
        fusion_dim=768,
        fusion_num_heads=8,
        fusion_num_layers=2,
        fusion_dropout=0.1,
        
        # MOE configuration
        use_moe=kwargs.get('use_moe', False),
        moe_type=kwargs.get('moe_type', 'standard'),  # 'standard', 'vqa', 'sparse'
        num_experts=kwargs.get('num_experts', 4),
        num_experts_per_token=kwargs.get('num_experts_per_token', 2),
        expert_capacity_factor=kwargs.get('expert_capacity_factor', 1.25),
        moe_loss_weight=kwargs.get('moe_loss_weight', 0.01),
        moe_position=kwargs.get('moe_position', 'fusion'),
        
        # VQA MOE specific - specialized experts
        num_vision_experts=kwargs.get('num_vision_experts', 2),
        num_text_experts=kwargs.get('num_text_experts', 2),
        num_multimodal_experts=kwargs.get('num_multimodal_experts', 2),
        num_specialized_experts=kwargs.get('num_specialized_experts', 2),
        vietnamese_optimized=kwargs.get('vietnamese_optimized', True),
        
        vocab_size=vocab_size,
        pad_token_id=pad_token_id,
        bos_token_id=bos_token_id,
        eos_token_id=eos_token_id,
        
        label_smoothing=0.1,
        tie_word_embeddings=True
    )
    
    # Apply any additional overrides that weren't handled above
    handled_keys = ['freeze_visual_encoder', 'freeze_question_encoder', 'max_answer_length',
                    'use_moe', 'moe_type', 'num_experts', 'num_experts_per_token', 
                    'expert_capacity_factor', 'moe_loss_weight', 'moe_position',
                    'num_vision_experts', 'num_text_experts', 'num_multimodal_experts',
                    'num_specialized_experts', 'vietnamese_optimized']
    for key, value in kwargs.items():
        if hasattr(config, key) and key not in handled_keys:
            setattr(config, key, value)
    
    # Re-sync aliased fields after kwargs
    config.__post_init__()
    
    return config