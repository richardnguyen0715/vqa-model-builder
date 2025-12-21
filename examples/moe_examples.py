"""
MOE (Mixture of Experts) Examples for Vietnamese VQA
Demonstrates usage of various MOE components.
"""

import torch
import torch.nn as nn

# Example 1: Basic MOE Layer
def example_basic_moe():
    """
    Demonstrates basic MOE layer usage.
    """
    from src.modeling.moe import MOELayer, MOEConfig
    
    # Configuration
    config = MOEConfig(
        input_dim=768,
        hidden_dim=3072,
        output_dim=768,
        num_experts=8,
        num_experts_per_token=2
    )
    
    # Create MOE layer
    moe_layer = MOELayer(config=config)
    
    # Sample input: [batch_size, seq_len, input_dim]
    x = torch.randn(4, 128, 768)
    
    # Forward pass
    output = moe_layer(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    # Get auxiliary loss for training
    aux_loss = moe_layer.get_aux_loss()
    print(f"Auxiliary loss: {aux_loss.item():.4f}")
    
    return moe_layer


# Example 2: VQA-specific MOE with specialized experts
def example_vqa_moe():
    """
    Demonstrates VQA-specific MOE layer with specialized experts.
    Includes vision, text, multimodal, and specialized experts.
    """
    from src.modeling.moe import VQAMOELayer
    
    # Create VQA MOE layer
    vqa_moe = VQAMOELayer(
        input_dim=768,
        hidden_dim=3072,
        output_dim=768,
        num_vision_experts=2,      # Vision-specialized experts
        num_text_experts=2,        # Text-specialized experts (Vietnamese)
        num_multimodal_experts=2,  # Cross-modal experts
        num_specialized_experts=2, # SAM, Detection, OCR experts
        top_k=2,
        vietnamese_optimized=True
    )
    
    # Sample input
    x = torch.randn(4, 128, 768)
    
    # Forward pass
    output = vqa_moe(x)
    print(f"VQA MOE output shape: {output.shape}")
    
    # Get expert usage statistics
    expert_usage = vqa_moe.get_expert_usage()
    print("Expert usage:")
    for expert_id, usage in expert_usage.items():
        print(f"  Expert {expert_id}: {usage:.4f}")
    
    return vqa_moe


# Example 3: Hierarchical MOE
def example_hierarchical_moe():
    """
    Demonstrates hierarchical MOE with two-level routing.
    First routes to expert groups, then to individual experts.
    """
    from src.modeling.moe import HierarchicalMOE
    
    # Create hierarchical MOE
    hierarchical_moe = HierarchicalMOE(
        input_dim=768,
        hidden_dim=3072,
        output_dim=768,
        num_expert_groups=4,      # 4 groups: vision, text, multimodal, specialized
        experts_per_group=4,      # 4 experts in each group
        top_k_groups=2,           # Select 2 groups
        top_k_experts=1,          # Select 1 expert per group
        expert_types=['vision', 'text', 'multimodal', 'feedforward']
    )
    
    # Sample input
    x = torch.randn(4, 128, 768)
    
    # Forward pass
    output = hierarchical_moe(x)
    print(f"Hierarchical MOE output shape: {output.shape}")
    
    # Get auxiliary loss
    aux_loss = hierarchical_moe.get_aux_loss()
    print(f"Hierarchical MOE aux loss: {aux_loss.item():.4f}")
    
    return hierarchical_moe


# Example 4: Sparse MOE for memory efficiency
def example_sparse_moe():
    """
    Demonstrates sparse MOE layer for memory-efficient processing.
    Uses token dispatch with capacity constraints.
    """
    from src.modeling.moe import SparseMOELayer
    
    # Create sparse MOE layer
    sparse_moe = SparseMOELayer(
        input_dim=768,
        hidden_dim=3072,
        output_dim=768,
        num_experts=8,
        top_k=2,
        capacity_factor=1.25,  # Capacity per expert
        expert_type='feedforward'
    )
    
    # Sample input
    x = torch.randn(4, 128, 768)
    
    # Forward pass
    output = sparse_moe(x)
    print(f"Sparse MOE output shape: {output.shape}")
    
    return sparse_moe


# Example 5: Different router types
def example_router_types():
    """
    Demonstrates different router implementations.
    """
    from src.modeling.moe import (
        TopKRouter,
        SoftRouter,
        NoisyTopKRouter,
        ExpertChoiceRouter
    )
    
    input_dim = 768
    num_experts = 8
    
    # Sample input
    x = torch.randn(4, 128, input_dim)
    
    # Top-K Router
    topk_router = TopKRouter(input_dim, num_experts, top_k=2)
    weights, indices, aux = topk_router(x)
    print(f"Top-K Router - weights: {weights.shape}, indices: {indices.shape}")
    
    # Soft Router (all experts)
    soft_router = SoftRouter(input_dim, num_experts, temperature=1.0)
    weights, indices, aux = soft_router(x)
    print(f"Soft Router - weights: {weights.shape}, entropy: {aux['entropy'].item():.4f}")
    
    # Noisy Top-K Router
    noisy_router = NoisyTopKRouter(input_dim, num_experts, top_k=2, noise_std=1.0)
    weights, indices, aux = noisy_router(x)
    print(f"Noisy Router - weights: {weights.shape}")
    
    # Expert Choice Router
    expert_choice = ExpertChoiceRouter(input_dim, num_experts, capacity_factor=1.25)
    weights, indices, aux = expert_choice(x)
    print(f"Expert Choice Router - capacity: {aux['capacity']}")


# Example 6: Specialized experts
def example_specialized_experts():
    """
    Demonstrates specialized expert usage.
    """
    from src.modeling.moe.specialized_experts import (
        SegmentationExpert,
        ObjectDetectionExpert,
        OCRExpert,
        SceneUnderstandingExpert,
        SpatialReasoningExpert
    )
    
    input_dim = 768
    hidden_dim = 3072
    output_dim = 768
    
    # Sample input
    x = torch.randn(4, 128, input_dim)
    
    # Segmentation Expert (SAM-inspired)
    seg_expert = SegmentationExpert(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        num_mask_tokens=4
    )
    seg_out = seg_expert(x)
    print(f"Segmentation Expert output: {seg_out.shape}")
    
    # Object Detection Expert (DETR-inspired)
    det_expert = ObjectDetectionExpert(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        num_queries=100
    )
    det_out = det_expert(x)
    print(f"Detection Expert output: {det_out.shape}")
    
    # OCR Expert (Vietnamese optimized)
    ocr_expert = OCRExpert(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        vietnamese_optimized=True
    )
    ocr_out = ocr_expert(x)
    print(f"OCR Expert output: {ocr_out.shape}")
    
    # Scene Understanding Expert
    scene_expert = SceneUnderstandingExpert(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim
    )
    scene_out = scene_expert(x)
    print(f"Scene Expert output: {scene_out.shape}")
    
    # Spatial Reasoning Expert
    spatial_expert = SpatialReasoningExpert(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim
    )
    spatial_out = spatial_expert(x)
    print(f"Spatial Expert output: {spatial_out.shape}")


# Example 7: MOE with multimodal inputs
def example_moe_multimodal():
    """
    Demonstrates MOE processing of multimodal (vision + text) features.
    """
    from src.modeling.moe import VQAMOELayer
    from src.modeling.moe.expert_types import MultimodalExpert
    
    # Create multimodal expert
    mm_expert = MultimodalExpert(
        input_dim=768,
        hidden_dim=3072,
        output_dim=768,
        use_cross_attention=True,
        use_modality_gate=True
    )
    
    # Vision and text features
    vision_features = torch.randn(4, 196, 768)  # 14x14 patches
    text_features = torch.randn(4, 32, 768)     # 32 text tokens
    
    # Process vision with text context
    vision_enhanced = mm_expert(
        vision_features,
        context=text_features
    )
    print(f"Enhanced vision features: {vision_enhanced.shape}")
    
    # Process text with vision context
    text_enhanced = mm_expert(
        text_features,
        context=vision_features
    )
    print(f"Enhanced text features: {text_enhanced.shape}")


# Example 8: Analyzing routing patterns
def example_routing_analysis():
    """
    Demonstrates routing pattern analysis utilities.
    """
    from src.modeling.moe import MOELayer, analyze_routing_patterns
    
    # Create MOE layer
    moe_layer = MOELayer(
        input_dim=768,
        hidden_dim=3072,
        output_dim=768,
        num_experts=8,
        top_k=2
    )
    
    # Sample input
    x = torch.randn(4, 128, 768)
    
    # Forward pass
    output = moe_layer(x)
    
    # Get routing information
    router_probs = moe_layer.aux_outputs.get('router_probs')
    if router_probs is not None:
        # Simulate expert indices for analysis
        _, expert_indices = torch.topk(router_probs, 2, dim=-1)
        
        # Analyze patterns
        analysis = analyze_routing_patterns(
            router_probs,
            expert_indices,
            num_experts=8
        )
        
        print("Routing Analysis:")
        print(f"  Entropy: {analysis['routing_entropy']:.4f}")
        print(f"  Max prob mean: {analysis['max_prob_mean']:.4f}")
        print("  Expert utilization:")
        for expert_id, util in analysis['expert_utilization'].items():
            print(f"    Expert {expert_id}: {util:.4f}")


if __name__ == '__main__':
    print("=" * 60)
    print("MOE Examples for Vietnamese VQA")
    print("=" * 60)
    
    print("\n1. Basic MOE Layer")
    print("-" * 40)
    example_basic_moe()
    
    print("\n2. VQA-specific MOE")
    print("-" * 40)
    example_vqa_moe()
    
    print("\n3. Hierarchical MOE")
    print("-" * 40)
    example_hierarchical_moe()
    
    print("\n4. Sparse MOE")
    print("-" * 40)
    example_sparse_moe()
    
    print("\n5. Router Types")
    print("-" * 40)
    example_router_types()
    
    print("\n6. Specialized Experts")
    print("-" * 40)
    example_specialized_experts()
    
    print("\n7. Multimodal MOE")
    print("-" * 40)
    example_moe_multimodal()
    
    print("\n8. Routing Analysis")
    print("-" * 40)
    example_routing_analysis()
    
    print("\n" + "=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)
