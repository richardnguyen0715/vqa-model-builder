"""
Examples of using Image Representation models with pretrained weights.
"""

import torch
from src.modeling.heads import create_image_representation


def example_region_based():
    """Example: Region-based approach with pretrained ResNet."""
    print("\n=== Region-Based Vision Embedding ===")
    
    # Using torchvision ResNet101
    model = create_image_representation(
        model_type='region_based',
        backbone_name='resnet101',
        num_regions=36,
        output_dim=768,
        use_spatial_features=True,
        pretrained_source='torchvision'
    )
    
    # Test forward pass
    images = torch.randn(2, 3, 224, 224)
    features = model(images)
    print(f"Input shape: {images.shape}")
    print(f"Output shape: {features.shape}")  # [2, 36, 768]
    print(f"Feature dimension: {model.get_feature_dim()}")


def example_vision_transformer():
    """Example: Vision Transformer with HuggingFace pretrained weights."""
    print("\n=== Vision Transformer Embedding ===")
    
    # Using HuggingFace ViT
    model = create_image_representation(
        model_type='vit',
        image_size=224,
        patch_size=16,
        embed_dim=768,
        num_heads=12,
        num_layers=12,
        use_pretrained=True,
        pretrained_model='google/vit-base-patch16-224-in21k'
    )
    
    # Test forward pass
    images = torch.randn(2, 3, 224, 224)
    features = model(images)
    print(f"Input shape: {images.shape}")
    print(f"Output shape: {features.shape}")  # [2, 197, 768] (196 patches + 1 CLS)
    print(f"Feature dimension: {model.get_feature_dim()}")


def example_multi_resolution():
    """Example: Multi-resolution FPN with pretrained ResNet."""
    print("\n=== Multi-Resolution Features ===")
    
    # Using torchvision ResNet50
    model = create_image_representation(
        model_type='multi_resolution',
        backbone_name='resnet50',
        output_dim=768,
        feature_levels=[2, 3, 4],
        fpn_dim=256,
        pretrained_source='torchvision'
    )
    
    # Test forward pass
    images = torch.randn(2, 3, 224, 224)
    features = model(images)
    print(f"Input shape: {images.shape}")
    print(f"Output shape: {features.shape}")  # [2, 1, 768]
    print(f"Feature dimension: {model.get_feature_dim()}")


def example_vision_token_cnn():
    """Example: Vision Token with CNN backbone."""
    print("\n=== Vision Token Embedding (CNN backbone) ===")
    
    # Using ResNet50 as backbone
    model = create_image_representation(
        model_type='vision_token',
        backbone_name='resnet50',
        num_vision_tokens=32,
        token_dim=768,
        num_attention_heads=8,
        num_layers=4,
        use_token_type=True,
        pretrained_source='torchvision'
    )
    
    # Test forward pass
    images = torch.randn(2, 3, 224, 224)
    features = model(images)
    print(f"Input shape: {images.shape}")
    print(f"Output shape: {features.shape}")  # [2, 32, 768]
    print(f"Feature dimension: {model.get_feature_dim()}")


def example_vision_token_vit():
    """Example: Vision Token with ViT backbone from HuggingFace."""
    print("\n=== Vision Token Embedding (ViT backbone) ===")
    
    # Using ViT as backbone
    model = create_image_representation(
        model_type='vision_token',
        backbone_name='vit-base-patch16-224',
        num_vision_tokens=32,
        token_dim=768,
        num_attention_heads=8,
        num_layers=4,
        use_token_type=True,
        pretrained_source='huggingface'
    )
    
    # Test forward pass
    images = torch.randn(2, 3, 224, 224)
    features = model(images)
    print(f"Input shape: {images.shape}")
    print(f"Output shape: {features.shape}")  # [2, 32, 768]
    print(f"Feature dimension: {model.get_feature_dim()}")


def example_comparison():
    """Compare different approaches on same image."""
    print("\n=== Comparison of All Approaches ===")
    
    # Create models
    models = {
        'Region-Based': create_image_representation(
            model_type='region_based',
            backbone_name='resnet50',
            num_regions=36,
            output_dim=768
        ),
        'ViT': create_image_representation(
            model_type='vit',
            image_size=224,
            embed_dim=768,
            use_pretrained=False  # Use random init for speed
        ),
        'Multi-Resolution': create_image_representation(
            model_type='multi_resolution',
            backbone_name='resnet50',
            output_dim=768
        ),
        'Vision Token': create_image_representation(
            model_type='vision_token',
            backbone_name='resnet50',
            num_vision_tokens=32,
            token_dim=768
        )
    }
    
    # Test on same image
    images = torch.randn(1, 3, 224, 224)
    
    print(f"\nInput image shape: {images.shape}")
    print("\nOutput shapes:")
    for name, model in models.items():
        model.eval()
        with torch.no_grad():
            features = model(images)
        print(f"  {name:20s}: {tuple(features.shape)}")


def example_with_different_sources():
    """Example showing different pretrained sources."""
    print("\n=== Different Pretrained Sources ===")
    
    sources = ['torchvision', 'huggingface']
    
    for source in sources:
        print(f"\n--- Using {source} ---")
        try:
            if source == 'torchvision':
                model = create_image_representation(
                    model_type='region_based',
                    backbone_name='resnet50',
                    pretrained_source=source
                )
            else:  # huggingface
                model = create_image_representation(
                    model_type='vision_token',
                    backbone_name='vit-base-patch16-224',
                    pretrained_source=source
                )
            print(f"✓ Successfully loaded model from {source}")
        except Exception as e:
            print(f"✗ Error loading from {source}: {e}")


if __name__ == "__main__":
    print("=" * 60)
    print("Image Representation Models - Pretrained Examples")
    print("=" * 60)
    
    # Run examples
    try:
        example_region_based()
    except Exception as e:
        print(f"Error in region_based example: {e}")
    
    try:
        example_multi_resolution()
    except Exception as e:
        print(f"Error in multi_resolution example: {e}")
    
    try:
        example_vision_token_cnn()
    except Exception as e:
        print(f"Error in vision_token_cnn example: {e}")
    
    # Note: These require HuggingFace transformers
    try:
        example_vision_transformer()
    except Exception as e:
        print(f"Error in ViT example (may need transformers): {e}")
    
    try:
        example_vision_token_vit()
    except Exception as e:
        print(f"Error in vision_token_vit example (may need transformers): {e}")
    
    # Comparison
    try:
        example_comparison()
    except Exception as e:
        print(f"Error in comparison: {e}")
    
    try:
        example_with_different_sources()
    except Exception as e:
        print(f"Error in sources comparison: {e}")
    
    print("\n" + "=" * 60)
    print("Examples completed!")
    print("=" * 60)
