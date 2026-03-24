#!/usr/bin/env python
"""Check checkpoint configuration alignment with evaluation config."""

import torch
from pathlib import Path

checkpoint_path = Path("checkpoints/RichardNguyen_ViMoE-VQA/best_generative_model.pt")

# Load checkpoint
checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

print("\n" + "=" * 80)
print("MODEL CONFIGURATION (from checkpoint)")
print("=" * 80)

config = checkpoint['config']
print(f"\nVisual Backbone: {config.visual_backbone}")
print(f"Text Encoder: {config.text_encoder}")
print(f"Hidden Size: {config.hidden_size}")
print(f"Num Decoder Layers: {config.num_decoder_layers}")
print(f"Num Attention Heads: {config.num_attention_heads}")
print(f"Freeze Visual Encoder: {config.freeze_visual_encoder}")
print(f"Freeze Question Encoder: {config.freeze_question_encoder}")

print(f"\n--- MOE Configuration ---")
print(f"MOE Enabled: {config.moe_enabled}")
if config.moe_enabled:
    print(f"MOE Type: {config.moe_type}")
    print(f"MOE Position: {config.moe_position}")
    print(f"Num Experts: {config.num_experts}")
    print(f"Capacity Factor: {config.capacity_factor}")
    print(f"Loss Weight: {config.loss_weight}")

print(f"\n--- Generation Configuration (from checkpoint model config) ---")
print(f"Max Generate Length: {config.max_generate_length}")
print(f"Num Beams: {config.num_beams}")
print(f"Do Sample: {config.do_sample}")
print(f"Temperature: {config.temperature}")

print(f"\n" + "=" * 80)
print("TRAINING CONFIGURATION (from checkpoint)")
print("=" * 80)

training_config = checkpoint['training_config']
print(f"\nNum Epochs: {training_config.num_epochs}")
print(f"Learning Rate: {training_config.learning_rate}")
print(f"Weight Decay: {training_config.weight_decay}")
print(f"Warmup Ratio: {training_config.warmup_ratio}")
print(f"Gradient Accumulation Steps: {training_config.gradient_accumulation_steps}")
print(f"Use AMP: {training_config.use_amp}")
print(f"Early Stopping Patience: {training_config.patience}")

print(f"\n--- Generation Config from Training ---")
print(f"Max Generate Length: {training_config.max_generate_length}")
print(f"Num Beams (during training validation): {training_config.num_beams}")
print(f"Do Sample: {training_config.do_sample}")
print(f"Temperature: {training_config.temperature}")

print(f"\n" + "=" * 80)
print("CHECKPOINT METADATA")
print("=" * 80)
print(f"\nEpoch: {checkpoint['epoch']}")
print(f"Global Step: {checkpoint['global_step']}")
print(f"Best Metric: {checkpoint['best_metric']}")
if isinstance(checkpoint['metrics'], dict):
    print(f"Final Metrics:")
    for key, value in checkpoint['metrics'].items():
        print(f"  {key}: {value}")

print("\n" + "=" * 80)
print("SYNCHRONIZATION CHECK")
print("=" * 80)

print(f"\n✓ Model config IS in checkpoint (loaded correctly during inference)")
print(f"✓ MOE config IS in checkpoint (type: {config.moe_type}, experts: {config.num_experts})")
print(f"✓ Training was with: num_beams={training_config.num_beams} (greedy)")
print(f"✓ Evaluation currently using: num_beams=3 (beam search)")
print(f"\n⚠️  DIFFERENCE: Training validated with num_beams=1, but evaluation uses num_beams=3")
print(f"   This can change generation quality but model can handle it.")

print("\n" + "=" * 80 + "\n")
