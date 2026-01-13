"""
Image Representation Module for VQA
Supports multiple approaches for extracting visual features from images.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List, Union
from abc import ABC, abstractmethod
import warnings
from src.middleware.logger import model_builder_logger as logger


# Utility function to load pretrained backbones
def load_pretrained_backbone(backbone_name: str, source: str = 'auto'):
    """
    Load pretrained backbone from various sources.
    
    Args:
        backbone_name: Name of the backbone (e.g., 'resnet50', 'vit-base-patch16-224')
        source: Source to load from ('auto', 'torchvision', 'huggingface', 'timm')
        
    Returns:
        Loaded backbone model and feature dimension
    """
    from torchvision import models
    
    # ResNet models from torchvision
    if backbone_name in ['resnet50', 'resnet101', 'resnet152']:
        try:
            if backbone_name == 'resnet50':
                backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
            elif backbone_name == 'resnet101':
                backbone = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V2)
            elif backbone_name == 'resnet152':
                backbone = models.resnet152(weights=models.ResNet152_Weights.IMAGENET1K_V2)
            backbone_dim = 2048
            logger.info(f"Loaded {backbone_name} from torchvision with ImageNet weights")
        except AttributeError:
            # Fallback for older torchvision
            backbone = getattr(models, backbone_name)(pretrained=True)
            backbone_dim = 2048
            logger.info(f"Loaded {backbone_name} from torchvision (legacy API)")
        return backbone, backbone_dim
    
    # ViT models from HuggingFace
    elif backbone_name.startswith('vit') or source == 'huggingface':
        try:
            from transformers import ViTModel, ViTConfig
            
            # Map common names to HuggingFace model IDs
            model_map = {
                'vit-base': 'google/vit-base-patch16-224',
                'vit-base-patch16-224': 'google/vit-base-patch16-224',
                'vit-large': 'google/vit-large-patch16-224',
                'vit-large-patch16-224': 'google/vit-large-patch16-224',
            }
            
            model_id = model_map.get(backbone_name, backbone_name)
            backbone = ViTModel.from_pretrained(model_id)
            backbone_dim = backbone.config.hidden_size
            logger.info(f"Loaded {model_id} from HuggingFace")
            return backbone, backbone_dim
        except ImportError:
            raise ImportError("transformers library required for ViT models. Install: pip install transformers")
    
    # Models from timm
    elif source == 'timm':
        try:
            import timm
            backbone = timm.create_model(backbone_name, pretrained=True, features_only=True)
            backbone_dim = backbone.feature_info.channels()[-1]
            logger.info(f"Loaded {backbone_name} from timm")
            return backbone, backbone_dim
        except ImportError:
            raise ImportError("timm library required. Install: pip install timm")
    
    else:
        raise ValueError(f"Unsupported backbone: {backbone_name}")


class BaseImageRepresentation(ABC, nn.Module):
    """
    Abstract base class for image representation models.
    All image encoders should inherit from this class.
    """
    
    def __init__(self, output_dim: int = 768):
        """
        Args:
            output_dim: Dimension of output feature vectors
        """
        super().__init__()
        self.output_dim = output_dim
    
    @abstractmethod
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extract visual features from images.
        
        Args:
            images: Input images tensor [batch_size, 3, H, W]
            
        Returns:
            features: Visual features [batch_size, num_features, output_dim]
        """
        pass
    
    @abstractmethod
    def get_feature_dim(self) -> int:
        """Return the output feature dimension."""
        pass


class RegionBasedVisionEmbedding(BaseImageRepresentation):
    """
    Region-based visual embeddings using Faster R-CNN or similar object detection models.
    Extracts features from detected object regions (Bottom-Up Attention).
    
    Reference: Anderson et al. "Bottom-Up and Top-Down Attention for VQA" (2018)
    """
    
    def __init__(
        self,
        backbone_name: str = 'resnet101',
        num_regions: int = 36,
        region_feat_dim: int = 2048,
        output_dim: int = 768,
        use_spatial_features: bool = True,
        pretrained_source: str = 'auto'
    ):
        """
        Args:
            backbone_name: Name of backbone CNN (resnet50, resnet101, etc.)
            num_regions: Maximum number of detected regions
            region_feat_dim: Feature dimension from region proposals
            output_dim: Output feature dimension
            use_spatial_features: Whether to include spatial location features
            pretrained_source: Source for pretrained weights ('auto', 'torchvision', 'huggingface', 'timm')
        """
        super().__init__(output_dim)
        
        self.num_regions = num_regions
        self.region_feat_dim = region_feat_dim
        self.use_spatial_features = use_spatial_features
        
        # Load pretrained backbone using utility function
        backbone, backbone_dim = load_pretrained_backbone(backbone_name, pretrained_source)
        
        # Remove final FC layer and avgpool
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])
        
        # Region proposal network (simplified - in practice use Faster R-CNN)
        self.roi_pool = nn.AdaptiveAvgPool2d((7, 7))
        
        # Spatial feature encoder (bbox coordinates)
        spatial_dim = 6 if use_spatial_features else 0  # [x1, y1, x2, y2, w, h]
        
        # Feature projection
        self.feature_projection = nn.Sequential(
            nn.Linear(region_feat_dim + spatial_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.LayerNorm(output_dim)
        )
    
    def extract_regions(self, feature_map: torch.Tensor) -> torch.Tensor:
        """
        Extract region features from feature map.
        Simplified version - in practice would use RPN + RoI pooling.
        
        Args:
            feature_map: Feature map [batch_size, C, H, W]
            
        Returns:
            region_features: [batch_size, num_regions, region_feat_dim]
        """
        batch_size = feature_map.size(0)
        
        # Simplified: Use spatial pyramid pooling to get fixed number of regions
        # In practice: Use Faster R-CNN RPN to generate proposals
        h, w = feature_map.shape[2], feature_map.shape[3]
        grid_size = int(self.num_regions ** 0.5)
        
        region_features = []
        for i in range(grid_size):
            for j in range(grid_size):
                # Extract region
                h_start = i * h // grid_size
                h_end = (i + 1) * h // grid_size
                w_start = j * w // grid_size
                w_end = (j + 1) * w // grid_size
                
                region = feature_map[:, :, h_start:h_end, w_start:w_end]
                pooled = F.adaptive_avg_pool2d(region, (1, 1))
                region_features.append(pooled.squeeze(-1).squeeze(-1))
        
        return torch.stack(region_features, dim=1)
    
    def get_spatial_features(self, batch_size: int) -> torch.Tensor:
        """
        Generate spatial features (bbox coordinates) for regions.
        
        Returns:
            spatial_features: [batch_size, num_regions, 6]
        """
        grid_size = int(self.num_regions ** 0.5)
        spatial_feats = []
        
        for i in range(grid_size):
            for j in range(grid_size):
                x1 = j / grid_size
                y1 = i / grid_size
                x2 = (j + 1) / grid_size
                y2 = (i + 1) / grid_size
                w = x2 - x1
                h = y2 - y1
                spatial_feats.append([x1, y1, x2, y2, w, h])
        
        spatial_feats = torch.tensor(spatial_feats, dtype=torch.float32)
        return spatial_feats.unsqueeze(0).repeat(batch_size, 1, 1)
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: [batch_size, 3, H, W]
            
        Returns:
            region_features: [batch_size, num_regions, output_dim]
        """
        batch_size = images.size(0)
        
        # Extract feature map from backbone
        feature_map = self.backbone(images)  # [B, C, H, W]
        
        # Extract region features
        region_feats = self.extract_regions(feature_map)  # [B, num_regions, region_feat_dim]
        
        # Add spatial features if enabled
        if self.use_spatial_features:
            spatial_feats = self.get_spatial_features(batch_size).to(images.device)
            region_feats = torch.cat([region_feats, spatial_feats], dim=-1)
        
        # Project to output dimension
        output = self.feature_projection(region_feats)
        
        return output
    
    def get_feature_dim(self) -> int:
        return self.output_dim


class VisionTransformerEmbedding(BaseImageRepresentation):
    """
    Vision Transformer (ViT) based image representation.
    Uses patch embeddings and transformer encoder.
    
    Reference: Dosovitskiy et al. "An Image is Worth 16x16 Words" (2021)
    """
    
    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 768,
        num_heads: int = 12,
        num_layers: int = 12,
        mlp_ratio: int = 4,
        dropout: float = 0.1,
        use_pretrained: bool = True,
        pretrained_model: str = 'google/vit-base-patch16-224-in21k'
    ):
        """
        Args:
            image_size: Input image size (assumes square images)
            patch_size: Size of each patch
            in_channels: Number of input channels (3 for RGB)
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            mlp_ratio: MLP hidden dimension ratio
            dropout: Dropout rate
            use_pretrained: Whether to use pretrained weights
            pretrained_model: HuggingFace model ID or 'timm' model name
        """
        super().__init__(embed_dim)
        
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.pretrained_model = pretrained_model
        
        # Patch embedding
        self.patch_embed = nn.Conv2d(
            in_channels, embed_dim, 
            kernel_size=patch_size, 
            stride=patch_size
        )
        
        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Position embedding
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, embed_dim)
        )
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * mlp_ratio,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Normalization
        self.norm = nn.LayerNorm(embed_dim)
        
        # Initialize weights
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
        if use_pretrained:
            self._load_pretrained()
    
    def _load_pretrained(self):
        """Load pretrained ViT weights from HuggingFace or timm."""
        try:
            from transformers import ViTModel
            logger.info(f"Loading pretrained ViT from HuggingFace...")
            
            # Load pretrained ViT model
            pretrained_vit = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
            
            # Copy weights
            self.patch_embed.weight.data = pretrained_vit.embeddings.patch_embeddings.projection.weight.data
            self.patch_embed.bias.data = pretrained_vit.embeddings.patch_embeddings.projection.bias.data
            
            self.cls_token.data = pretrained_vit.embeddings.cls_token.data
            self.pos_embed.data = pretrained_vit.embeddings.position_embeddings.data
            
            # Copy transformer weights
            for i, layer in enumerate(self.transformer.layers):
                pretrained_layer = pretrained_vit.encoder.layer[i]
                
                # Self-attention
                layer.self_attn.in_proj_weight.data = torch.cat([
                    pretrained_layer.attention.attention.query.weight.data,
                    pretrained_layer.attention.attention.key.weight.data,
                    pretrained_layer.attention.attention.value.weight.data
                ], dim=0)
                layer.self_attn.in_proj_bias.data = torch.cat([
                    pretrained_layer.attention.attention.query.bias.data,
                    pretrained_layer.attention.attention.key.bias.data,
                    pretrained_layer.attention.attention.value.bias.data
                ], dim=0)
                layer.self_attn.out_proj.weight.data = pretrained_layer.attention.output.dense.weight.data
                layer.self_attn.out_proj.bias.data = pretrained_layer.attention.output.dense.bias.data
                
                # FFN
                layer.linear1.weight.data = pretrained_layer.intermediate.dense.weight.data
                layer.linear1.bias.data = pretrained_layer.intermediate.dense.bias.data
                layer.linear2.weight.data = pretrained_layer.output.dense.weight.data
                layer.linear2.bias.data = pretrained_layer.output.dense.bias.data
                
                # Layer norms
                layer.norm1.weight.data = pretrained_layer.layernorm_before.weight.data
                layer.norm1.bias.data = pretrained_layer.layernorm_before.bias.data
                layer.norm2.weight.data = pretrained_layer.layernorm_after.weight.data
                layer.norm2.bias.data = pretrained_layer.layernorm_after.bias.data
            
            # Final norm
            self.norm.weight.data = pretrained_vit.layernorm.weight.data
            self.norm.bias.data = pretrained_vit.layernorm.bias.data
            
            logger.info("Successfully loaded pretrained ViT weights from HuggingFace")
            
        except ImportError:
            logger.info("transformers library not found. Trying timm...")
            try:
                import timm
                pretrained_vit = timm.create_model('vit_base_patch16_224', pretrained=True)
                # Load weights from timm model
                self.load_state_dict(pretrained_vit.state_dict(), strict=False)
                logger.info("Successfully loaded pretrained ViT weights from timm")
            except ImportError:
                logger.info("Neither transformers nor timm found. Using random initialization.")
        except Exception as e:
            logger.info(f"Error loading pretrained weights: {e}. Using random initialization.")
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: [batch_size, 3, H, W]
            
        Returns:
            features: [batch_size, num_patches + 1, embed_dim]
        """
        batch_size = images.size(0)
        
        # Patch embedding
        x = self.patch_embed(images)  # [B, embed_dim, H/P, W/P]
        x = x.flatten(2).transpose(1, 2)  # [B, num_patches, embed_dim]
        
        # Add CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # [B, num_patches + 1, embed_dim]
        
        # Add position embedding
        x = x + self.pos_embed
        
        # Transformer encoding
        x = self.transformer(x)
        
        # Normalize
        x = self.norm(x)
        
        return x
    
    def get_feature_dim(self) -> int:
        return self.output_dim


class MultiResolutionFeatures(BaseImageRepresentation):
    """
    Multi-resolution feature extraction using Feature Pyramid Network (FPN).
    Combines features from multiple scales for better representation.
    
    Reference: Lin et al. "Feature Pyramid Networks for Object Detection" (2017)
    """
    
    def __init__(
        self,
        backbone_name: str = 'resnet50',
        output_dim: int = 768,
        feature_levels: List[int] = None,
        fpn_dim: int = 256,
        pretrained_source: str = 'auto'
    ):
        """
        Args:
            backbone_name: Name of backbone CNN
            output_dim: Output feature dimension
            feature_levels: Which stages to extract features from (default: [2, 3, 4])
            fpn_dim: FPN feature dimension
            pretrained_source: Source for pretrained weights ('auto', 'torchvision', 'huggingface', 'timm')
        """
        super().__init__(output_dim)
        
        if feature_levels is None:
            feature_levels = [2, 3, 4]
        
        self.feature_levels = feature_levels
        self.fpn_dim = fpn_dim
        
        # Load pretrained backbone using utility function
        backbone, _ = load_pretrained_backbone(backbone_name, pretrained_source)
        
        # Extract layers for each level
        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        
        self.layer1 = backbone.layer1  # Stage 1
        self.layer2 = backbone.layer2  # Stage 2
        self.layer3 = backbone.layer3  # Stage 3
        self.layer4 = backbone.layer4  # Stage 4
        
        # FPN lateral connections
        self.lateral_convs = nn.ModuleDict()
        self.fpn_convs = nn.ModuleDict()
        
        # Channel dimensions for ResNet stages
        stage_channels = {1: 256, 2: 512, 3: 1024, 4: 2048}
        
        for level in feature_levels:
            # Lateral 1x1 conv to reduce channels
            self.lateral_convs[f'level{level}'] = nn.Conv2d(
                stage_channels[level], fpn_dim, kernel_size=1
            )
            # 3x3 conv after upsampling
            self.fpn_convs[f'level{level}'] = nn.Conv2d(
                fpn_dim, fpn_dim, kernel_size=3, padding=1
            )
        
        # Global pooling and projection
        total_fpn_features = fpn_dim * len(feature_levels)
        self.feature_projection = nn.Sequential(
            nn.Linear(total_fpn_features, output_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.LayerNorm(output_dim)
        )
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: [batch_size, 3, H, W]
            
        Returns:
            features: [batch_size, num_features, output_dim]
        """
        # Bottom-up pathway
        x = self.conv1(images)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        c1 = self.layer1(x)   # Stage 1
        c2 = self.layer2(c1)  # Stage 2
        c3 = self.layer3(c2)  # Stage 3
        c4 = self.layer4(c3)  # Stage 4
        
        stage_features = {1: c1, 2: c2, 3: c3, 4: c4}
        
        # Top-down pathway with lateral connections (FPN)
        fpn_features = []
        prev_features = None
        
        for level in sorted(self.feature_levels, reverse=True):
            # Lateral connection
            lateral = self.lateral_convs[f'level{level}'](stage_features[level])
            
            # Add upsampled previous level
            if prev_features is not None:
                lateral = lateral + F.interpolate(
                    prev_features, size=lateral.shape[-2:], 
                    mode='nearest'
                )
            
            # 3x3 conv
            fpn_feat = self.fpn_convs[f'level{level}'](lateral)
            fpn_features.append(fpn_feat)
            prev_features = fpn_feat
        
        # Reverse to get low-to-high resolution order
        fpn_features = fpn_features[::-1]
        
        # Pool each level and concatenate
        pooled_features = []
        for feat in fpn_features:
            pooled = F.adaptive_avg_pool2d(feat, (1, 1))
            pooled_features.append(pooled.squeeze(-1).squeeze(-1))
        
        # Concatenate multi-scale features
        multi_scale_feat = torch.cat(pooled_features, dim=-1)
        
        # Project to output dimension and add sequence dimension
        output = self.feature_projection(multi_scale_feat)
        output = output.unsqueeze(1)  # [B, 1, output_dim]
        
        return output
    
    def get_feature_dim(self) -> int:
        return self.output_dim


class VisionTokenEmbedding(BaseImageRepresentation):
    """
    Vision Token based representation inspired by recent VQA papers.
    Uses learnable visual tokens that attend to image features.
    
    This approach uses a set of learnable "vision tokens" that query the image
    through cross-attention, similar to Perceiver or BLIP-2 approaches.
    
    Reference: Inspired by approaches in modern VLMs
    """
    
    def __init__(
        self,
        backbone_name: str = 'resnet50',
        num_vision_tokens: int = 32,
        token_dim: int = 768,
        num_attention_heads: int = 8,
        num_layers: int = 4,
        dropout: float = 0.1,
        use_token_type: bool = True,
        pretrained_source: str = 'auto'
    ):
        """
        Args:
            backbone_name: Name of CNN backbone for image encoding or 'vit-*' for ViT
            num_vision_tokens: Number of learnable vision query tokens
            token_dim: Dimension of vision tokens
            num_attention_heads: Number of attention heads
            num_layers: Number of cross-attention layers
            dropout: Dropout rate
            use_token_type: Whether to use token type embeddings
            pretrained_source: Source for pretrained weights ('auto', 'torchvision', 'huggingface', 'timm')
        """
        super().__init__(token_dim)
        
        self.num_vision_tokens = num_vision_tokens
        self.token_dim = token_dim
        self.use_token_type = use_token_type
        
        # Load pretrained backbone using utility function
        backbone, backbone_dim = load_pretrained_backbone(backbone_name, pretrained_source)
        
        # Handle different backbone types
        if backbone_name.startswith('vit'):
            # For ViT backbones, use the whole model
            self.backbone = backbone
            self.is_vit_backbone = True
        else:
            # For CNN backbones, remove final layers
            self.backbone = nn.Sequential(*list(backbone.children())[:-2])
            self.is_vit_backbone = False
        
        # Project image features to token dimension
        self.image_projection = nn.Linear(backbone_dim, token_dim)
        
        # Learnable vision query tokens
        self.vision_tokens = nn.Parameter(
            torch.randn(1, num_vision_tokens, token_dim)
        )
        
        # Token type embeddings (optional)
        if use_token_type:
            self.token_type_embeddings = nn.Embedding(2, token_dim)
        
        # Position embeddings for image patches
        self.pos_embed = nn.Parameter(torch.randn(1, 49, token_dim))  # 7x7 for ResNet
        
        # Cross-attention layers
        self.cross_attention_layers = nn.ModuleList([
            CrossAttentionLayer(
                dim=token_dim,
                num_heads=num_attention_heads,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])
        
        # Final normalization
        self.norm = nn.LayerNorm(token_dim)
        
        # Initialize
        nn.init.trunc_normal_(self.vision_tokens, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: [batch_size, 3, H, W]
            
        Returns:
            vision_tokens: [batch_size, num_vision_tokens, token_dim]
        """
        batch_size = images.size(0)
        
        # Extract image features based on backbone type
        if self.is_vit_backbone:
            # For ViT backbone: extract patch embeddings
            outputs = self.backbone(images, output_hidden_states=True)
            img_features = outputs.last_hidden_state  # [B, num_patches+1, hidden_dim]
            # Remove CLS token
            img_features = img_features[:, 1:, :]  # [B, num_patches, hidden_dim]
        else:
            # For CNN backbone: extract feature map
            img_features = self.backbone(images)  # [B, C, H, W]
            
            # Flatten spatial dimensions
            B, C, H, W = img_features.shape
            img_features = img_features.flatten(2).transpose(1, 2)  # [B, H*W, C]
        
        # Project to token dimension
        img_features = self.image_projection(img_features)  # [B, seq_len, token_dim]
        
        # Add position embeddings (adjust size if needed)
        seq_len = img_features.size(1)
        if seq_len <= self.pos_embed.size(1):
            img_features = img_features + self.pos_embed[:, :seq_len, :]
        else:
            # Interpolate position embeddings if sequence length is different
            pos_embed_interp = F.interpolate(
                self.pos_embed.transpose(1, 2),
                size=seq_len,
                mode='linear',
                align_corners=False
            ).transpose(1, 2)
            img_features = img_features + pos_embed_interp
        
        # Initialize vision tokens
        vision_tokens = self.vision_tokens.expand(batch_size, -1, -1)
        
        # Add token type embeddings if enabled
        if self.use_token_type:
            vision_tokens = vision_tokens + self.token_type_embeddings(
                torch.zeros(batch_size, self.num_vision_tokens, dtype=torch.long, device=images.device)
            )
            img_features = img_features + self.token_type_embeddings(
                torch.ones(batch_size, seq_len, dtype=torch.long, device=images.device)
            )
        
        # Apply cross-attention layers
        for layer in self.cross_attention_layers:
            vision_tokens = layer(vision_tokens, img_features)
        
        # Final normalization
        vision_tokens = self.norm(vision_tokens)
        
        return vision_tokens
    
    def get_feature_dim(self) -> int:
        return self.token_dim


class CrossAttentionLayer(nn.Module):
    """
    Cross-attention layer for vision token querying.
    Query: Vision tokens, Key/Value: Image features
    """
    
    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout)
        )
        
        # Layer norms
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
    
    def forward(self, query: torch.Tensor, key_value: torch.Tensor) -> torch.Tensor:
        """
        Args:
            query: [batch_size, num_queries, dim]
            key_value: [batch_size, num_kv, dim]
            
        Returns:
            output: [batch_size, num_queries, dim]
        """
        # Cross-attention
        attn_output, _ = self.multihead_attn(
            query, key_value, key_value
        )
        query = self.norm1(query + attn_output)
        
        # Feed-forward
        ffn_output = self.ffn(query)
        output = self.norm2(query + ffn_output)
        
        return output


# Factory function for easy model creation
def create_image_representation(
    model_type: str,
    **kwargs
) -> BaseImageRepresentation:
    """
    Factory function to create image representation models.
    
    Args:
        model_type: Type of model ('region_based', 'vit', 'multi_resolution', 'vision_token')
        **kwargs: Additional arguments for the specific model
        
    Returns:
        Image representation model
    """
    model_registry = {
        'region_based': RegionBasedVisionEmbedding,
        'vit': VisionTransformerEmbedding,
        'multi_resolution': MultiResolutionFeatures,
        'vision_token': VisionTokenEmbedding
    }
    
    if model_type not in model_registry:
        raise ValueError(
            f"Unknown model type: {model_type}. "
            f"Available types: {list(model_registry.keys())}"
        )
    
    return model_registry[model_type](**kwargs)
