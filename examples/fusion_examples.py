"""
Examples of using Multimodal Fusion models.
"""

import torch
from src.modeling.fusion import create_fusion_model


def example_cross_attention():
    """Example: Cross-Attention Fusion."""
    print("\n=== Cross-Attention Fusion ===")
    
    fusion = create_fusion_model(
        'cross_attention',
        vision_dim=768,
        text_dim=768,
        output_dim=768,
        num_attention_heads=8,
        num_layers=4,
        fusion_method='concat'
    )
    
    # Simulate vision and text features
    batch_size = 2
    num_vision_tokens = 36
    num_text_tokens = 20
    
    vision_features = torch.randn(batch_size, num_vision_tokens, 768)
    text_features = torch.randn(batch_size, num_text_tokens, 768)
    
    # Forward pass
    fusion.eval()
    with torch.no_grad():
        output = fusion(vision_features, text_features)
    
    print(f"Vision features: {vision_features.shape}")
    print(f"Text features: {text_features.shape}")
    print(f"Fused output: {output.shape}")
    print(f"Output dimension: {fusion.get_output_dim()}")


def example_qformer():
    """Example: Q-Former Fusion."""
    print("\n=== Q-Former Fusion ===")
    
    fusion = create_fusion_model(
        'qformer',
        vision_dim=768,
        text_dim=768,
        output_dim=768,
        num_query_tokens=32,
        num_attention_heads=8,
        num_layers=6
    )
    
    # Simulate features
    batch_size = 2
    vision_features = torch.randn(batch_size, 36, 768)
    text_features = torch.randn(batch_size, 20, 768)
    
    # Forward pass
    fusion.eval()
    with torch.no_grad():
        output = fusion(vision_features, text_features)
    
    print(f"Vision features: {vision_features.shape}")
    print(f"Text features: {text_features.shape}")
    print(f"Number of query tokens: 32")
    print(f"Fused output: {output.shape}")
    print(f"Output dimension: {fusion.get_output_dim()}")


def example_single_stream():
    """Example: Single-Stream Fusion (ViLT)."""
    print("\n=== Single-Stream Fusion (ViLT) ===")
    
    fusion = create_fusion_model(
        'single_stream',
        vision_dim=768,
        text_dim=768,
        output_dim=768,
        num_attention_heads=12,
        num_layers=6
    )
    
    # Simulate features
    batch_size = 2
    vision_features = torch.randn(batch_size, 36, 768)
    text_features = torch.randn(batch_size, 20, 768)
    
    # Forward pass
    fusion.eval()
    with torch.no_grad():
        output = fusion(vision_features, text_features)
    
    print(f"Vision features: {vision_features.shape}")
    print(f"Text features: {text_features.shape}")
    print(f"Combined sequence length: {1 + 36 + 20} ([CLS] + vision + text)")
    print(f"Fused output: {output.shape}")
    print(f"Output dimension: {fusion.get_output_dim()}")


def example_with_masks():
    """Example: Using attention masks."""
    print("\n=== Fusion with Attention Masks ===")
    
    fusion = create_fusion_model(
        'cross_attention',
        vision_dim=768,
        text_dim=768
    )
    
    batch_size = 2
    vision_features = torch.randn(batch_size, 36, 768)
    text_features = torch.randn(batch_size, 20, 768)
    
    # Create masks (True = valid token, False = padding)
    vision_mask = torch.ones(batch_size, 36, dtype=torch.bool)
    text_mask = torch.ones(batch_size, 20, dtype=torch.bool)
    
    # Mask last 5 text tokens in batch
    text_mask[:, 15:] = False
    
    # Forward with masks
    fusion.eval()
    with torch.no_grad():
        output = fusion(vision_features, text_features, vision_mask, text_mask)
    
    print(f"Vision mask: {vision_mask.sum(dim=1)} valid tokens")
    print(f"Text mask: {text_mask.sum(dim=1)} valid tokens")
    print(f"Fused output: {output.shape}")


def example_different_dimensions():
    """Example: Different input dimensions."""
    print("\n=== Different Input Dimensions ===")
    
    fusion = create_fusion_model(
        'qformer',
        vision_dim=2048,  # ResNet features
        text_dim=768,     # BERT features
        output_dim=512,   # Custom output
        num_query_tokens=32
    )
    
    batch_size = 2
    vision_features = torch.randn(batch_size, 36, 2048)
    text_features = torch.randn(batch_size, 20, 768)
    
    # Forward pass
    fusion.eval()
    with torch.no_grad():
        output = fusion(vision_features, text_features)
    
    print(f"Vision features: {vision_features.shape}")
    print(f"Text features: {text_features.shape}")
    print(f"Fused output: {output.shape}")
    print("Fusion handles dimension projection internally!")


def example_fusion_methods():
    """Compare different fusion methods in Cross-Attention."""
    print("\n=== Fusion Methods Comparison (Cross-Attention) ===")
    
    fusion_methods = ['concat', 'add', 'multiply']
    
    vision_features = torch.randn(1, 36, 768)
    text_features = torch.randn(1, 20, 768)
    
    for method in fusion_methods:
        print(f"\n--- Fusion Method: {method} ---")
        
        fusion = create_fusion_model(
            'cross_attention',
            vision_dim=768,
            text_dim=768,
            output_dim=768,
            fusion_method=method
        )
        
        fusion.eval()
        with torch.no_grad():
            output = fusion(vision_features, text_features)
        
        print(f"  Output shape: {output.shape}")


def example_comparison():
    """Compare all three fusion approaches."""
    print("\n=== Comparison of All Fusion Approaches ===")
    
    batch_size = 2
    vision_features = torch.randn(batch_size, 36, 768)
    text_features = torch.randn(batch_size, 20, 768)
    
    fusion_configs = [
        ('cross_attention', {'num_layers': 4}),
        ('qformer', {'num_query_tokens': 32}),
        ('single_stream', {'num_layers': 6}),
    ]
    
    print(f"\nInput:")
    print(f"  Vision: {vision_features.shape}")
    print(f"  Text: {text_features.shape}")
    print("\nOutputs:")
    
    for fusion_type, config in fusion_configs:
        try:
            fusion = create_fusion_model(
                fusion_type,
                vision_dim=768,
                text_dim=768,
                output_dim=768,
                **config
            )
            
            fusion.eval()
            with torch.no_grad():
                output = fusion(vision_features, text_features)
            
            # Count parameters
            num_params = sum(p.numel() for p in fusion.parameters())
            
            print(f"  {fusion_type:20s}: {tuple(output.shape)}, "
                  f"{num_params/1e6:.2f}M params")
            
        except Exception as e:
            print(f"  {fusion_type:20s}: Error - {e}")


def example_vqa_integration():
    """Example: Integration in a VQA model."""
    print("\n=== VQA Model Integration ===")
    
    import torch.nn as nn
    
    class SimpleVQAModel(nn.Module):
        def __init__(self, num_answers=3129):
            super().__init__()
            
            # Vision encoder (placeholder)
            self.vision_encoder = nn.Linear(2048, 768)
            
            # Text encoder (placeholder)
            self.text_encoder = nn.Linear(300, 768)
            
            # Fusion module
            self.fusion = create_fusion_model(
                'qformer',
                vision_dim=768,
                text_dim=768,
                output_dim=768,
                num_query_tokens=32
            )
            
            # Answer prediction head
            self.answer_head = nn.Sequential(
                nn.Linear(768, 512),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(512, num_answers)
            )
        
        def forward(self, vision_feat, text_feat):
            # Encode
            vision_feat = self.vision_encoder(vision_feat)
            text_feat = self.text_encoder(text_feat)
            
            # Fuse
            fused = self.fusion(vision_feat, text_feat)
            
            # Predict
            logits = self.answer_head(fused)
            return logits
    
    # Create model
    model = SimpleVQAModel(num_answers=3129)
    
    # Test forward pass
    vision_input = torch.randn(2, 36, 2048)
    text_input = torch.randn(2, 20, 300)
    
    model.eval()
    with torch.no_grad():
        logits = model(vision_input, text_input)
    
    print(f"Vision input: {vision_input.shape}")
    print(f"Text input: {text_input.shape}")
    print(f"Answer logits: {logits.shape}")
    
    # Count total parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params/1e6:.2f}M")


if __name__ == "__main__":
    print("=" * 60)
    print("Multimodal Fusion Examples")
    print("=" * 60)
    
    # Run examples
    try:
        example_cross_attention()
    except Exception as e:
        print(f"Error in Cross-Attention example: {e}")
    
    try:
        example_qformer()
    except Exception as e:
        print(f"Error in Q-Former example: {e}")
    
    try:
        example_single_stream()
    except Exception as e:
        print(f"Error in Single-Stream example: {e}")
    
    try:
        example_with_masks()
    except Exception as e:
        print(f"Error in masks example: {e}")
    
    try:
        example_different_dimensions()
    except Exception as e:
        print(f"Error in different dimensions example: {e}")
    
    try:
        example_fusion_methods()
    except Exception as e:
        print(f"Error in fusion methods example: {e}")
    
    try:
        example_comparison()
    except Exception as e:
        print(f"Error in comparison: {e}")
    
    try:
        example_vqa_integration()
    except Exception as e:
        print(f"Error in VQA integration: {e}")
    
    print("\n" + "=" * 60)
    print("Examples completed!")
    print("=" * 60)
