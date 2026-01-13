"""
Model Pipeline Module
Handles model creation, configuration, and initialization with comprehensive logging.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field

import torch
import torch.nn as nn

from src.core.pipeline_logger import get_pipeline_logger, PipelineLogger


@dataclass
class ModelPipelineConfig:
    """Configuration for model pipeline."""
    
    # Visual encoder
    visual_backbone: str = "vit"  # vit, resnet, clip, swin
    visual_model_name: str = "openai/clip-vit-base-patch32"
    visual_output_dim: int = 768
    freeze_visual: bool = False
    
    # Text encoder
    text_encoder_type: str = "phobert"  # phobert, bert, roberta
    text_model_name: str = "vinai/phobert-base"
    text_output_dim: int = 768
    text_max_length: int = 64
    freeze_text: bool = False
    
    # Fusion
    fusion_type: str = "cross_attention"  # cross_attention, concat, bilinear
    fusion_hidden_dim: int = 768
    fusion_num_heads: int = 8
    fusion_num_layers: int = 2
    fusion_dropout: float = 0.1
    
    # MOE (Mixture of Experts)
    use_moe: bool = False
    moe_num_experts: int = 8
    moe_top_k: int = 2
    moe_hidden_dim: int = 2048
    moe_load_balance_weight: float = 0.01
    
    # Knowledge/RAG
    use_knowledge: bool = False
    knowledge_num_contexts: int = 5
    knowledge_retriever_type: str = "dense"
    
    # Answer head
    num_answers: int = 3000
    answer_hidden_dims: List[int] = field(default_factory=lambda: [768, 512])
    answer_dropout: float = 0.3
    
    # General
    embed_dim: int = 768
    dropout: float = 0.1
    
    # Device
    device: str = "auto"  # auto, cuda, cpu
    

@dataclass
class ModelPipelineOutput:
    """Output from model pipeline."""
    
    model: nn.Module
    config: Any  # VQAModelConfig
    device: torch.device
    
    total_params: int
    trainable_params: int
    frozen_params: int
    model_size_mb: float


class ModelPipeline:
    """
    Complete model pipeline for VQA.
    
    Handles:
    - Model configuration
    - Model building (visual encoder, text encoder, fusion, MOE, answer head)
    - Weight initialization
    - Model validation
    - Device placement
    
    All steps are logged with comprehensive validation output.
    """
    
    def __init__(
        self,
        config: Optional[ModelPipelineConfig] = None,
        logger: Optional[PipelineLogger] = None
    ):
        """
        Initialize model pipeline.
        
        Args:
            config: Model pipeline configuration
            logger: Pipeline logger instance
        """
        self.config = config or ModelPipelineConfig()
        self.logger = logger or get_pipeline_logger()
        
        self.model = None
        self.model_config = None
        self.device = None
        
    def run(self, num_answers: Optional[int] = None) -> ModelPipelineOutput:
        """
        Execute the complete model pipeline.
        
        Args:
            num_answers: Number of answer classes (overrides config if provided)
            
        Returns:
            ModelPipelineOutput with model and info
        """
        self.logger.start_stage("MODEL PIPELINE")
        
        try:
            # Update num_answers if provided
            if num_answers is not None:
                self.config.num_answers = num_answers
                
            # Step 1: Setup device
            self._setup_device()
            
            # Step 2: Build model configuration
            self._build_model_config()
            
            # Step 3: Create model
            self._create_model()
            
            # Step 4: Initialize weights
            self._initialize_weights()
            
            # Step 5: Move to device
            self._move_to_device()
            
            # Step 6: Log model architecture
            self._log_model_architecture()
            
            # Step 7: Validate model
            output = self._validate_model()
            
            self.logger.end_stage("MODEL PIPELINE", success=True)
            
            return output
            
        except Exception as e:
            self.logger.error(f"Model pipeline failed: {e}")
            self.logger.end_stage("MODEL PIPELINE", success=False)
            raise
            
    def _setup_device(self):
        """Setup computation device."""
        self.logger.subsection("Device Setup")
        
        if self.config.device == "auto":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                self.logger.success(f"Using CUDA: {gpu_name}")
                self.logger.key_value("GPU Memory", f"{gpu_memory:.1f} GB")
            else:
                self.device = torch.device("cpu")
                self.logger.info("CUDA not available, using CPU")
        else:
            self.device = torch.device(self.config.device)
            
        self.logger.key_value("Device", str(self.device))
        
        # Log CUDA details if available
        if self.device.type == "cuda":
            self.logger.key_value("CUDA Version", torch.version.cuda)
            self.logger.key_value("cuDNN Version", torch.backends.cudnn.version())
            self.logger.key_value("cuDNN Enabled", torch.backends.cudnn.enabled)
            
    def _build_model_config(self):
        """Build model configuration."""
        self.logger.subsection("Building Model Configuration")
        
        from src.modeling.meta_arch import (
            VQAModelConfig,
            VisualEncoderConfig,
            TextEncoderConfig,
            FusionConfig,
            MOEConfig,
            KnowledgeConfig,
            AnswerHeadConfig
        )
        
        # Visual encoder config
        visual_config = VisualEncoderConfig(
            backbone_type=self.config.visual_backbone,
            model_name=self.config.visual_model_name,
            pretrained=True,
            freeze_backbone=self.config.freeze_visual,
            output_dim=self.config.visual_output_dim
        )
        
        self.logger.info("Visual Encoder Configuration:")
        self.logger.key_value("Backbone type", self.config.visual_backbone)
        self.logger.key_value("Model name", self.config.visual_model_name)
        self.logger.key_value("Output dim", self.config.visual_output_dim)
        self.logger.key_value("Frozen", self.config.freeze_visual)
        
        # Text encoder config
        text_config = TextEncoderConfig(
            encoder_type=self.config.text_encoder_type,
            model_name=self.config.text_model_name,
            pretrained=True,
            freeze_encoder=self.config.freeze_text,
            output_dim=self.config.text_output_dim,
            max_length=self.config.text_max_length
        )
        
        self.logger.info("Text Encoder Configuration:")
        self.logger.key_value("Encoder type", self.config.text_encoder_type)
        self.logger.key_value("Model name", self.config.text_model_name)
        self.logger.key_value("Output dim", self.config.text_output_dim)
        self.logger.key_value("Max length", self.config.text_max_length)
        self.logger.key_value("Frozen", self.config.freeze_text)
        
        # Fusion config
        fusion_config = FusionConfig(
            fusion_type=self.config.fusion_type,
            hidden_dim=self.config.fusion_hidden_dim,
            output_dim=self.config.fusion_hidden_dim,
            num_heads=self.config.fusion_num_heads,
            num_layers=self.config.fusion_num_layers,
            dropout=self.config.fusion_dropout
        )
        
        self.logger.info("Fusion Configuration:")
        self.logger.key_value("Fusion type", self.config.fusion_type)
        self.logger.key_value("Hidden dim", self.config.fusion_hidden_dim)
        self.logger.key_value("Num heads", self.config.fusion_num_heads)
        self.logger.key_value("Num layers", self.config.fusion_num_layers)
        self.logger.key_value("Dropout", self.config.fusion_dropout)
        
        # MOE config
        moe_config = MOEConfig(
            use_moe=self.config.use_moe,
            num_experts=self.config.moe_num_experts,
            top_k=self.config.moe_top_k,
            hidden_dim=self.config.moe_hidden_dim,
            load_balance_weight=self.config.moe_load_balance_weight
        )
        
        self.logger.info("MOE Configuration:")
        self.logger.key_value("Use MOE", self.config.use_moe)
        if self.config.use_moe:
            self.logger.key_value("Num experts", self.config.moe_num_experts)
            self.logger.key_value("Top-K", self.config.moe_top_k)
            self.logger.key_value("Hidden dim", self.config.moe_hidden_dim)
            
        # Knowledge config
        knowledge_config = KnowledgeConfig(
            use_knowledge=self.config.use_knowledge,
            num_contexts=self.config.knowledge_num_contexts,
            retriever_type=self.config.knowledge_retriever_type
        )
        
        self.logger.info("Knowledge Configuration:")
        self.logger.key_value("Use Knowledge/RAG", self.config.use_knowledge)
        if self.config.use_knowledge:
            self.logger.key_value("Num contexts", self.config.knowledge_num_contexts)
            self.logger.key_value("Retriever type", self.config.knowledge_retriever_type)
            
        # Answer head config
        answer_config = AnswerHeadConfig(
            num_answers=self.config.num_answers,
            hidden_dims=self.config.answer_hidden_dims,
            dropout=self.config.answer_dropout
        )
        
        self.logger.info("Answer Head Configuration:")
        self.logger.key_value("Num answers (classes)", self.config.num_answers)
        self.logger.key_value("Hidden dims", self.config.answer_hidden_dims)
        self.logger.key_value("Dropout", self.config.answer_dropout)
        
        # Complete model config
        self.model_config = VQAModelConfig(
            visual_encoder=visual_config,
            text_encoder=text_config,
            fusion=fusion_config,
            moe=moe_config,
            knowledge=knowledge_config,
            answer_head=answer_config,
            embed_dim=self.config.embed_dim,
            dropout=self.config.dropout
        )
        
        self.logger.success("Model configuration built")
        
    def _create_model(self):
        """Create the VQA model."""
        self.logger.subsection("Creating Model")
        
        from src.modeling.meta_arch import VietnameseVQAModel
        
        self.logger.info("Initializing VQA model...")
        self.logger.info("(This may take a while to download pretrained weights)")
        
        self.model = VietnameseVQAModel(self.model_config)
        
        self.logger.success("Model created successfully")
        
    def _initialize_weights(self):
        """Initialize model weights."""
        self.logger.subsection("Weight Initialization")
        
        # Count pre-initialized weights (from pretrained models)
        pretrained_modules = []
        random_modules = []
        
        for name, module in self.model.named_modules():
            if hasattr(module, 'weight') and module.weight is not None:
                # Check if weights look pretrained (not all zeros/ones)
                weight = module.weight.data
                if weight.abs().mean() > 0.001 and weight.std() > 0.001:
                    pretrained_modules.append(name)
                else:
                    random_modules.append(name)
                    
        self.logger.key_value("Pretrained modules", len(pretrained_modules))
        self.logger.key_value("Random initialized modules", len(random_modules))
        
        # Apply custom initialization to non-pretrained modules if needed
        def init_weights(m):
            if isinstance(m, nn.Linear):
                if m.weight.abs().mean() < 0.001:  # Likely uninitialized
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
                
        # Only apply to answer head and fusion layers
        if hasattr(self.model, 'answer_head'):
            self.model.answer_head.apply(init_weights)
            self.logger.info("Applied Xavier initialization to answer head")
            
        self.logger.success("Weight initialization completed")
        
    def _move_to_device(self):
        """Move model to device."""
        self.logger.subsection("Moving Model to Device")
        
        self.model = self.model.to(self.device)
        
        # Verify all parameters are on correct device
        all_on_device = all(p.device == self.device for p in self.model.parameters())
        
        if all_on_device:
            self.logger.success(f"Model moved to {self.device}")
        else:
            self.logger.warning("Some parameters may not be on the correct device")
            
    def _log_model_architecture(self):
        """Log detailed model architecture."""
        self.logger.subsection("Model Architecture")
        
        # Parameter counts
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params
        model_size_mb = total_params * 4 / 1024 / 1024
        
        self.logger.info("Parameter Statistics:")
        self.logger.key_value("Total parameters", f"{total_params:,}")
        self.logger.key_value("Trainable parameters", f"{trainable_params:,}")
        self.logger.key_value("Frozen parameters", f"{frozen_params:,}")
        self.logger.key_value("Model size (MB)", f"{model_size_mb:.2f}")
        self.logger.key_value("Trainable ratio", f"{trainable_params/total_params*100:.1f}%")
        
        # Per-module breakdown
        self.logger.info("Module Parameter Breakdown:")
        
        module_params = {}
        for name, child in self.model.named_children():
            child_params = sum(p.numel() for p in child.parameters())
            child_trainable = sum(p.numel() for p in child.parameters() if p.requires_grad)
            module_params[name] = {
                'total': child_params,
                'trainable': child_trainable,
                'frozen': child_params - child_trainable
            }
            
        # Create table
        headers = ["Module", "Total", "Trainable", "Frozen", "%"]
        rows = []
        for name, counts in module_params.items():
            pct = counts['total'] / total_params * 100 if total_params > 0 else 0
            rows.append([
                name,
                f"{counts['total']:,}",
                f"{counts['trainable']:,}",
                f"{counts['frozen']:,}",
                f"{pct:.1f}%"
            ])
            
        self.logger.table(headers, rows, col_widths=[20, 15, 15, 15, 10])
        
        # Log model structure (abbreviated)
        self.logger.subsection("Model Structure")
        self.logger.info("Top-level modules:")
        for name, child in self.model.named_children():
            child_class = child.__class__.__name__
            self.logger.bullet(f"{name}: {child_class}", indent=4)
            
        # Store for output
        self._param_stats = {
            'total': total_params,
            'trainable': trainable_params,
            'frozen': frozen_params,
            'size_mb': model_size_mb
        }
        
    def _validate_model(self) -> ModelPipelineOutput:
        """Validate model with dummy forward pass."""
        self.logger.subsection("Model Validation")
        
        self.logger.info("Running validation forward pass...")
        
        # Create dummy inputs
        batch_size = 2
        dummy_images = torch.randn(batch_size, 3, 224, 224).to(self.device)
        dummy_input_ids = torch.randint(0, 30000, (batch_size, 64)).to(self.device)
        dummy_attention_mask = torch.ones(batch_size, 64).to(self.device)
        dummy_labels = torch.randint(0, self.config.num_answers, (batch_size,)).to(self.device)
        
        self.logger.info("Dummy input shapes:")
        self.logger.key_value("Images", list(dummy_images.shape))
        self.logger.key_value("Input IDs", list(dummy_input_ids.shape))
        self.logger.key_value("Attention mask", list(dummy_attention_mask.shape))
        self.logger.key_value("Labels", list(dummy_labels.shape))
        
        # Forward pass
        self.model.eval()
        with torch.no_grad():
            try:
                output = self.model(
                    pixel_values=dummy_images,
                    input_ids=dummy_input_ids,
                    attention_mask=dummy_attention_mask,
                    labels=dummy_labels
                )
                
                self.logger.success("Forward pass successful")
                
                # Log output
                self.logger.info("Model output:")
                if hasattr(output, 'logits'):
                    self.logger.key_value("Logits shape", list(output.logits.shape))
                if hasattr(output, 'loss') and output.loss is not None:
                    self.logger.key_value("Loss", f"{output.loss.item():.4f}")
                if hasattr(output, 'predictions') and output.predictions is not None:
                    self.logger.key_value("Predictions shape", list(output.predictions.shape))
                    
                # Verify output dimensions
                if hasattr(output, 'logits'):
                    expected_shape = (batch_size, self.config.num_answers)
                    actual_shape = tuple(output.logits.shape)
                    if actual_shape == expected_shape:
                        self.logger.success(f"Output shape verified: {actual_shape}")
                    else:
                        self.logger.warning(f"Output shape mismatch: {actual_shape} != {expected_shape}")
                        
            except Exception as e:
                self.logger.error(f"Forward pass failed: {e}")
                raise
                
        # Memory usage
        if self.device.type == "cuda":
            memory_allocated = torch.cuda.memory_allocated(self.device) / 1024**3
            memory_cached = torch.cuda.memory_reserved(self.device) / 1024**3
            self.logger.info("GPU Memory Usage:")
            self.logger.key_value("Allocated", f"{memory_allocated:.2f} GB")
            self.logger.key_value("Cached", f"{memory_cached:.2f} GB")
            
        self.logger.success("Model validation completed")
        
        return ModelPipelineOutput(
            model=self.model,
            config=self.model_config,
            device=self.device,
            total_params=self._param_stats['total'],
            trainable_params=self._param_stats['trainable'],
            frozen_params=self._param_stats['frozen'],
            model_size_mb=self._param_stats['size_mb']
        )
        
    def load_checkpoint(self, checkpoint_path: str) -> ModelPipelineOutput:
        """
        Load model from checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
            
        Returns:
            ModelPipelineOutput with loaded model
        """
        self.logger.subsection("Loading Checkpoint")
        
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
            
        self.logger.key_value("Checkpoint path", checkpoint_path)
        
        # Load checkpoint (weights_only=False for PyTorch 2.6+ compatibility with custom classes)
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        # Log checkpoint info
        self.logger.info("Checkpoint contents:")
        for key in checkpoint.keys():
            if key == 'model_state_dict':
                self.logger.key_value(key, f"{len(checkpoint[key])} parameters")
            elif key == 'optimizer_state_dict':
                self.logger.key_value(key, "present")
            elif key == 'vocabulary':
                self.logger.key_value(key, f"{len(checkpoint[key])} classes" if checkpoint[key] else "None")
            else:
                self.logger.key_value(key, checkpoint[key])
        
        # Check if num_answers from checkpoint differs from current config
        # Try to get num_answers from checkpoint or infer from model state dict
        checkpoint_num_answers = checkpoint.get('num_answers')
        if checkpoint_num_answers is None:
            # Infer from model state dict (answer_head final output layer)
            # The final layer bias shape gives us the number of classes
            model_state = checkpoint.get('model_state_dict', {})
            # Find the last layer in answer_head (highest numbered layer with bias)
            answer_head_biases = {}
            for key, value in model_state.items():
                if 'answer_head' in key and '.bias' in key:
                    # Extract layer number from key like "answer_head.classifier.6.bias"
                    try:
                        parts = key.split('.')
                        for i, part in enumerate(parts):
                            if part.isdigit():
                                layer_num = int(part)
                                answer_head_biases[layer_num] = value.shape[0]
                    except:
                        pass
            
            if answer_head_biases:
                # Get the highest layer number (final output layer)
                final_layer_num = max(answer_head_biases.keys())
                checkpoint_num_answers = answer_head_biases[final_layer_num]
                self.logger.info(f"Inferred num_answers={checkpoint_num_answers} from checkpoint model state (layer {final_layer_num})")
        
        if checkpoint_num_answers is not None and checkpoint_num_answers != self.config.num_answers:
            self.logger.warning(f"Checkpoint has {checkpoint_num_answers} classes, current config has {self.config.num_answers}")
            self.logger.info(f"Rebuilding model with {checkpoint_num_answers} classes from checkpoint")
            self.config.num_answers = checkpoint_num_answers
            self.model = None  # Force rebuild
                
        # Load model state
        if self.model is None:
            self._build_model_config()
            self._create_model()
            
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        
        self.logger.success("Checkpoint loaded successfully")
        
        # Return output
        self._log_model_architecture()
        
        return ModelPipelineOutput(
            model=self.model,
            config=self.model_config,
            device=self.device,
            total_params=self._param_stats['total'],
            trainable_params=self._param_stats['trainable'],
            frozen_params=self._param_stats['frozen'],
            model_size_mb=self._param_stats['size_mb']
        )
