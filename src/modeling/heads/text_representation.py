"""
Text Representation Module for VQA
Supports multiple approaches for extracting textual features from questions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Union, Tuple
from abc import ABC, abstractmethod
import warnings


# Utility function to load pretrained language models
def load_pretrained_language_model(model_name: str, source: str = 'huggingface'):
    """
    Load pretrained language model from HuggingFace.
    
    Args:
        model_name: Name of the model (e.g., 'bert-base-uncased', 'roberta-base')
        source: Source to load from (currently only 'huggingface' supported)
        
    Returns:
        Loaded model and tokenizer
    """
    try:
        from transformers import AutoModel, AutoTokenizer, AutoConfig
        
        print(f"Loading pretrained model: {model_name} from HuggingFace...")
        
        # Load model and tokenizer
        config = AutoConfig.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        print(f"Successfully loaded {model_name}")
        print(f"  Hidden size: {config.hidden_size}")
        print(f"  Vocab size: {config.vocab_size}")
        
        return model, tokenizer, config
        
    except ImportError:
        raise ImportError(
            "transformers library required for pretrained models. "
            "Install: pip install transformers"
        )
    except Exception as e:
        raise ValueError(f"Error loading model {model_name}: {e}")


class BaseTextRepresentation(ABC, nn.Module):
    """
    Abstract base class for text representation models.
    All text encoders should inherit from this class.
    """
    
    def __init__(self, output_dim: int = 768):
        """
        Args:
            output_dim: Dimension of output feature vectors
        """
        super().__init__()
        self.output_dim = output_dim
    
    @abstractmethod
    def forward(
        self, 
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Extract textual features from input tokens.
        
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            token_type_ids: Token type IDs [batch_size, seq_len] (optional)
            
        Returns:
            features: Text features [batch_size, seq_len, output_dim] or [batch_size, output_dim]
        """
        pass
    
    @abstractmethod
    def get_feature_dim(self) -> int:
        """Return the output feature dimension."""
        pass
    
    def get_tokenizer(self):
        """Return the tokenizer if available."""
        if hasattr(self, 'tokenizer'):
            return self.tokenizer
        return None


class BERTTextEmbedding(BaseTextRepresentation):
    """
    BERT-based text representation.
    Uses Bidirectional Encoder Representations from Transformers.
    
    Key Features:
    - Bidirectional context understanding
    - Masked language modeling pretraining
    - Next sentence prediction (NSP) task
    
    Reference: Devlin et al. "BERT: Pre-training of Deep Bidirectional Transformers" (2019)
    """
    
    def __init__(
        self,
        model_name: str = 'bert-base-uncased',
        output_dim: int = 768,
        use_pretrained: bool = True,
        pooling_strategy: str = 'cls',
        finetune: bool = True,
        add_projection: bool = False
    ):
        """
        Args:
            model_name: HuggingFace model ID (e.g., 'bert-base-uncased', 'bert-large-uncased')
            output_dim: Output feature dimension (if add_projection=True)
            use_pretrained: Whether to use pretrained weights
            pooling_strategy: How to pool token embeddings ('cls', 'mean', 'max', 'all')
            finetune: Whether to allow finetuning the BERT model
            add_projection: Whether to add projection layer to match output_dim
        """
        super().__init__(output_dim)
        
        self.model_name = model_name
        self.pooling_strategy = pooling_strategy
        self.finetune = finetune
        self.add_projection = add_projection
        
        if use_pretrained:
            # Load pretrained BERT
            self.bert_model, self.tokenizer, self.config = load_pretrained_language_model(model_name)
            self.hidden_size = self.config.hidden_size
        else:
            raise NotImplementedError("Random initialization not yet supported. Use pretrained models.")
        
        # Freeze or unfreeze parameters
        if not finetune:
            for param in self.bert_model.parameters():
                param.requires_grad = False
            print(f"BERT parameters frozen (finetune=False)")
        else:
            print(f"BERT parameters trainable (finetune=True)")
        
        # Optional projection layer
        if add_projection and self.hidden_size != output_dim:
            self.projection = nn.Sequential(
                nn.Linear(self.hidden_size, output_dim),
                nn.LayerNorm(output_dim),
                nn.Dropout(0.1)
            )
        else:
            self.projection = None
            self.output_dim = self.hidden_size
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
            token_type_ids: [batch_size, seq_len]
            
        Returns:
            features: Depending on pooling_strategy:
                - 'cls': [batch_size, hidden_size]
                - 'mean'/'max': [batch_size, hidden_size]
                - 'all': [batch_size, seq_len, hidden_size]
        """
        # Forward through BERT
        outputs = self.bert_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True
        )
        
        # Get hidden states
        sequence_output = outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]
        
        # Apply pooling strategy
        if self.pooling_strategy == 'cls':
            # Use CLS token representation
            pooled_output = sequence_output[:, 0, :]  # [batch_size, hidden_size]
        elif self.pooling_strategy == 'mean':
            # Mean pooling over sequence
            if attention_mask is not None:
                # Mask out padding tokens
                mask_expanded = attention_mask.unsqueeze(-1).expand(sequence_output.size()).float()
                sum_embeddings = torch.sum(sequence_output * mask_expanded, dim=1)
                sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
                pooled_output = sum_embeddings / sum_mask
            else:
                pooled_output = torch.mean(sequence_output, dim=1)
        elif self.pooling_strategy == 'max':
            # Max pooling over sequence
            if attention_mask is not None:
                mask_expanded = attention_mask.unsqueeze(-1).expand(sequence_output.size()).float()
                sequence_output = sequence_output.clone()
                sequence_output[mask_expanded == 0] = -1e9
            pooled_output = torch.max(sequence_output, dim=1)[0]
        elif self.pooling_strategy == 'all':
            # Return all token embeddings
            pooled_output = sequence_output
        else:
            raise ValueError(f"Unknown pooling strategy: {self.pooling_strategy}")
        
        # Apply projection if available
        if self.projection is not None:
            pooled_output = self.projection(pooled_output)
        
        return pooled_output
    
    def get_feature_dim(self) -> int:
        return self.output_dim


class RoBERTaTextEmbedding(BaseTextRepresentation):
    """
    RoBERTa-based text representation.
    Robustly Optimized BERT Approach - an improved version of BERT.
    
    Key Features:
    - Trained on more data than BERT
    - Removed Next Sentence Prediction (NSP) task
    - Dynamic masking during training
    - Larger batch sizes and learning rates
    
    Reference: Liu et al. "RoBERTa: A Robustly Optimized BERT Pretraining Approach" (2019)
    """
    
    def __init__(
        self,
        model_name: str = 'roberta-base',
        output_dim: int = 768,
        use_pretrained: bool = True,
        pooling_strategy: str = 'cls',
        finetune: bool = True,
        add_projection: bool = False
    ):
        """
        Args:
            model_name: HuggingFace model ID (e.g., 'roberta-base', 'roberta-large')
            output_dim: Output feature dimension (if add_projection=True)
            use_pretrained: Whether to use pretrained weights
            pooling_strategy: How to pool token embeddings ('cls', 'mean', 'max', 'all')
            finetune: Whether to allow finetuning the RoBERTa model
            add_projection: Whether to add projection layer to match output_dim
        """
        super().__init__(output_dim)
        
        self.model_name = model_name
        self.pooling_strategy = pooling_strategy
        self.finetune = finetune
        self.add_projection = add_projection
        
        if use_pretrained:
            # Load pretrained RoBERTa
            self.roberta_model, self.tokenizer, self.config = load_pretrained_language_model(model_name)
            self.hidden_size = self.config.hidden_size
        else:
            raise NotImplementedError("Random initialization not yet supported. Use pretrained models.")
        
        # Freeze or unfreeze parameters
        if not finetune:
            for param in self.roberta_model.parameters():
                param.requires_grad = False
            print(f"RoBERTa parameters frozen (finetune=False)")
        else:
            print(f"RoBERTa parameters trainable (finetune=True)")
        
        # Optional projection layer
        if add_projection and self.hidden_size != output_dim:
            self.projection = nn.Sequential(
                nn.Linear(self.hidden_size, output_dim),
                nn.LayerNorm(output_dim),
                nn.Dropout(0.1)
            )
        else:
            self.projection = None
            self.output_dim = self.hidden_size
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
            token_type_ids: [batch_size, seq_len] (not used by RoBERTa)
            
        Returns:
            features: Depending on pooling_strategy
        """
        # RoBERTa doesn't use token_type_ids
        outputs = self.roberta_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        # Get hidden states
        sequence_output = outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]
        
        # Apply pooling strategy (same as BERT)
        if self.pooling_strategy == 'cls':
            pooled_output = sequence_output[:, 0, :]
        elif self.pooling_strategy == 'mean':
            if attention_mask is not None:
                mask_expanded = attention_mask.unsqueeze(-1).expand(sequence_output.size()).float()
                sum_embeddings = torch.sum(sequence_output * mask_expanded, dim=1)
                sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
                pooled_output = sum_embeddings / sum_mask
            else:
                pooled_output = torch.mean(sequence_output, dim=1)
        elif self.pooling_strategy == 'max':
            if attention_mask is not None:
                mask_expanded = attention_mask.unsqueeze(-1).expand(sequence_output.size()).float()
                sequence_output = sequence_output.clone()
                sequence_output[mask_expanded == 0] = -1e9
            pooled_output = torch.max(sequence_output, dim=1)[0]
        elif self.pooling_strategy == 'all':
            pooled_output = sequence_output
        else:
            raise ValueError(f"Unknown pooling strategy: {self.pooling_strategy}")
        
        # Apply projection if available
        if self.projection is not None:
            pooled_output = self.projection(pooled_output)
        
        return pooled_output
    
    def get_feature_dim(self) -> int:
        return self.output_dim


class DeBERTaV3TextEmbedding(BaseTextRepresentation):
    """
    DeBERTa V3 text representation.
    Decoding-enhanced BERT with Disentangled Attention.
    
    Key Features:
    - Disentangled attention mechanism (separate content and position)
    - Enhanced mask decoder for better MLM
    - Improved training efficiency
    - Better performance on downstream tasks
    
    Reference: He et al. "DeBERTaV3: Improving DeBERTa using ELECTRA-Style Pre-Training" (2021)
    """
    
    def __init__(
        self,
        model_name: str = 'microsoft/deberta-v3-base',
        output_dim: int = 768,
        use_pretrained: bool = True,
        pooling_strategy: str = 'cls',
        finetune: bool = True,
        add_projection: bool = False
    ):
        """
        Args:
            model_name: HuggingFace model ID (e.g., 'microsoft/deberta-v3-base', 'microsoft/deberta-v3-large')
            output_dim: Output feature dimension (if add_projection=True)
            use_pretrained: Whether to use pretrained weights
            pooling_strategy: How to pool token embeddings ('cls', 'mean', 'max', 'all')
            finetune: Whether to allow finetuning the DeBERTa model
            add_projection: Whether to add projection layer to match output_dim
        """
        super().__init__(output_dim)
        
        self.model_name = model_name
        self.pooling_strategy = pooling_strategy
        self.finetune = finetune
        self.add_projection = add_projection
        
        if use_pretrained:
            # Load pretrained DeBERTa V3
            self.deberta_model, self.tokenizer, self.config = load_pretrained_language_model(model_name)
            self.hidden_size = self.config.hidden_size
        else:
            raise NotImplementedError("Random initialization not yet supported. Use pretrained models.")
        
        # Freeze or unfreeze parameters
        if not finetune:
            for param in self.deberta_model.parameters():
                param.requires_grad = False
            print(f"DeBERTa V3 parameters frozen (finetune=False)")
        else:
            print(f"DeBERTa V3 parameters trainable (finetune=True)")
        
        # Optional projection layer
        if add_projection and self.hidden_size != output_dim:
            self.projection = nn.Sequential(
                nn.Linear(self.hidden_size, output_dim),
                nn.LayerNorm(output_dim),
                nn.Dropout(0.1)
            )
        else:
            self.projection = None
            self.output_dim = self.hidden_size
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
            token_type_ids: [batch_size, seq_len]
            
        Returns:
            features: Depending on pooling_strategy
        """
        # Forward through DeBERTa V3
        outputs = self.deberta_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True
        )
        
        # Get hidden states
        sequence_output = outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]
        
        # Apply pooling strategy
        if self.pooling_strategy == 'cls':
            pooled_output = sequence_output[:, 0, :]
        elif self.pooling_strategy == 'mean':
            if attention_mask is not None:
                mask_expanded = attention_mask.unsqueeze(-1).expand(sequence_output.size()).float()
                sum_embeddings = torch.sum(sequence_output * mask_expanded, dim=1)
                sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
                pooled_output = sum_embeddings / sum_mask
            else:
                pooled_output = torch.mean(sequence_output, dim=1)
        elif self.pooling_strategy == 'max':
            if attention_mask is not None:
                mask_expanded = attention_mask.unsqueeze(-1).expand(sequence_output.size()).float()
                sequence_output = sequence_output.clone()
                sequence_output[mask_expanded == 0] = -1e9
            pooled_output = torch.max(sequence_output, dim=1)[0]
        elif self.pooling_strategy == 'all':
            pooled_output = sequence_output
        else:
            raise ValueError(f"Unknown pooling strategy: {self.pooling_strategy}")
        
        # Apply projection if available
        if self.projection is not None:
            pooled_output = self.projection(pooled_output)
        
        return pooled_output
    
    def get_feature_dim(self) -> int:
        return self.output_dim


class GenericTransformerTextEmbedding(BaseTextRepresentation):
    """
    Generic text embedding using any HuggingFace transformer model.
    Flexible wrapper for loading any pretrained language model.
    
    Supports models like:
    - ALBERT, DistilBERT, ELECTRA
    - T5, GPT-2, XLNet
    - And many more from HuggingFace Model Hub
    """
    
    def __init__(
        self,
        model_name: str,
        output_dim: int = 768,
        use_pretrained: bool = True,
        pooling_strategy: str = 'cls',
        finetune: bool = True,
        add_projection: bool = False,
        trust_remote_code: bool = False
    ):
        """
        Args:
            model_name: Any HuggingFace model ID
            output_dim: Output feature dimension (if add_projection=True)
            use_pretrained: Whether to use pretrained weights
            pooling_strategy: How to pool token embeddings ('cls', 'mean', 'max', 'all')
            finetune: Whether to allow finetuning the model
            add_projection: Whether to add projection layer to match output_dim
            trust_remote_code: Whether to trust remote code (for some models)
        """
        super().__init__(output_dim)
        
        self.model_name = model_name
        self.pooling_strategy = pooling_strategy
        self.finetune = finetune
        self.add_projection = add_projection
        
        if use_pretrained:
            try:
                from transformers import AutoModel, AutoTokenizer, AutoConfig
                
                print(f"Loading generic transformer model: {model_name}")
                
                # Load with optional trust_remote_code
                self.config = AutoConfig.from_pretrained(
                    model_name,
                    trust_remote_code=trust_remote_code
                )
                self.transformer_model = AutoModel.from_pretrained(
                    model_name,
                    trust_remote_code=trust_remote_code
                )
                self.tokenizer = AutoTokenizer.from_pretrained(
                    model_name,
                    trust_remote_code=trust_remote_code
                )
                
                self.hidden_size = self.config.hidden_size
                print(f"Successfully loaded {model_name} (hidden_size: {self.hidden_size})")
                
            except Exception as e:
                raise ValueError(f"Error loading model {model_name}: {e}")
        else:
            raise NotImplementedError("Random initialization not yet supported. Use pretrained models.")
        
        # Freeze or unfreeze parameters
        if not finetune:
            for param in self.transformer_model.parameters():
                param.requires_grad = False
            print(f"Model parameters frozen (finetune=False)")
        else:
            print(f"Model parameters trainable (finetune=True)")
        
        # Optional projection layer
        if add_projection and self.hidden_size != output_dim:
            self.projection = nn.Sequential(
                nn.Linear(self.hidden_size, output_dim),
                nn.LayerNorm(output_dim),
                nn.Dropout(0.1)
            )
        else:
            self.projection = None
            self.output_dim = self.hidden_size
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
            token_type_ids: [batch_size, seq_len]
            
        Returns:
            features: Depending on pooling_strategy
        """
        # Try to forward with all arguments, fallback if not supported
        try:
            outputs = self.transformer_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                return_dict=True
            )
        except TypeError:
            # Some models don't support token_type_ids
            outputs = self.transformer_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True
            )
        
        # Get hidden states
        sequence_output = outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]
        
        # Apply pooling strategy
        if self.pooling_strategy == 'cls':
            pooled_output = sequence_output[:, 0, :]
        elif self.pooling_strategy == 'mean':
            if attention_mask is not None:
                mask_expanded = attention_mask.unsqueeze(-1).expand(sequence_output.size()).float()
                sum_embeddings = torch.sum(sequence_output * mask_expanded, dim=1)
                sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
                pooled_output = sum_embeddings / sum_mask
            else:
                pooled_output = torch.mean(sequence_output, dim=1)
        elif self.pooling_strategy == 'max':
            if attention_mask is not None:
                mask_expanded = attention_mask.unsqueeze(-1).expand(sequence_output.size()).float()
                sequence_output = sequence_output.clone()
                sequence_output[mask_expanded == 0] = -1e9
            pooled_output = torch.max(sequence_output, dim=1)[0]
        elif self.pooling_strategy == 'all':
            pooled_output = sequence_output
        else:
            raise ValueError(f"Unknown pooling strategy: {self.pooling_strategy}")
        
        # Apply projection if available
        if self.projection is not None:
            pooled_output = self.projection(pooled_output)
        
        return pooled_output
    
    def get_feature_dim(self) -> int:
        return self.output_dim


# Factory function for easy model creation
def create_text_representation(
    model_type: str,
    **kwargs
) -> BaseTextRepresentation:
    """
    Factory function to create text representation models.
    
    Args:
        model_type: Type of model ('bert', 'roberta', 'deberta-v3', 'generic')
        **kwargs: Additional arguments for the specific model
        
    Returns:
        Text representation model
        
    Examples:
        >>> # BERT
        >>> model = create_text_representation('bert', model_name='bert-base-uncased')
        
        >>> # RoBERTa
        >>> model = create_text_representation('roberta', model_name='roberta-large')
        
        >>> # DeBERTa V3
        >>> model = create_text_representation('deberta-v3', model_name='microsoft/deberta-v3-base')
        
        >>> # Any other model
        >>> model = create_text_representation('generic', model_name='distilbert-base-uncased')
    """
    model_registry = {
        'bert': BERTTextEmbedding,
        'roberta': RoBERTaTextEmbedding,
        'deberta-v3': DeBERTaV3TextEmbedding,
        'deberta': DeBERTaV3TextEmbedding,  # Alias
        'generic': GenericTransformerTextEmbedding
    }
    
    if model_type not in model_registry:
        # If model_type is not recognized, treat it as a model name and use generic
        print(f"Model type '{model_type}' not found in registry. Using GenericTransformerTextEmbedding.")
        return GenericTransformerTextEmbedding(model_name=model_type, **kwargs)
    
    return model_registry[model_type](**kwargs)
