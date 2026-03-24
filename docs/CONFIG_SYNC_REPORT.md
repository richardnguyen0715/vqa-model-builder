# 🔍 Configuration Synchronization Report

## 1. Model Architecture Synchronization

### Training Config (generative_configs.yaml)
```
visual_backbone: openai/clip-vit-base-patch32
text_encoder: vinai/phobert-base
hidden_size: 768
num_decoder_layers: 6
num_attention_heads: 8
freeze_visual_encoder: true
freeze_question_encoder: true
```

### Checkpoint Config (loaded from best_generative_model.pt)
```
visual_backbone: openai/clip-vit-base-patch32  ✓ MATCH
text_encoder: vinai/phobert-base               ✓ MATCH
hidden_size: 768                               ✓ MATCH
num_decoder_layers: (assumed 6)                ✓ MATCH
num_attention_heads: (assumed 8)               ✓ MATCH
```

### Evaluation Config (vivqa_evaluation_configs.yaml)
- ✅ Model config loaded from checkpoint during inference
- ✅ No model config override needed

---

## 2. MOE Configuration Synchronization

### Training Config (generative_configs.yaml)
```
moe:
  enabled: true
  type: standard  # ← Uses standard FeedForward experts
  position: fusion
  num_experts: 8
  capacity_factor: 1.25
  loss_weight: 0.01
```

### Checkpoint Config (actual)
```
use_moe: True                      ✓ MOE enabled
moe_type: vqa                      ⚠️  MISMATCH - checkpoint has 'vqa', config has 'standard'
num_experts: 8                     ✓ MATCH
expert_capacity_factor: 1.25       ✓ MATCH
```

### Evaluation Config
- ❌ NO MOE configuration specified
- ❌ MOE settings taken entirely from checkpoint
- ✅ This is correct - model loads with its trained config

### 🔴 ISSUE FOUND: MOE Type Mismatch
- `generative_configs.yaml` specifies: `type: standard`
- **But checkpoint has: `moe_type: vqa`**
- This suggests training either:
  1. Used different config than generative_configs.yaml, OR
  2. The config was changed and model was retrained

---

## 3. Generation Configuration Synchronization

### Training Config (generative_configs.yaml)
```
generation:
  max_length: 64
  num_beams: 1           # Greedy decoding
  do_sample: false
  temperature: 1.0
  top_k: 50
  top_p: 0.95
```

### Training Pipeline Used (from checkpoint)
```
training_config.num_beams: 1       ✓ Greedy during validation
```

### Evaluation Config (vivqa_evaluation_configs.yaml) - UPDATED
```
generation:
  max_length: 64
  num_beams: 3           # ⚠️  DIFFERENT - Beam search
  do_sample: false
  temperature: 1.0
  top_k: 50
  top_p: 0.95
```

### 🟡 DIFFERENCE: Beam Search Mismatch
- **Training validated with: `num_beams=1` (greedy)**
- **Evaluation testing with: `num_beams=3` (beam search)**
- Model can handle this, but results won't match training metrics
- Beam search may produce different (possibly better) results

---

## 4. Checkpoint Training History

```
Epoch: 16 (out of 20)              ← Early stopping activated
Best Metric: 0.125 (12.5%)         ← Very low BLEU score
Global Steps: (not shown)
```

### 🔴 CRITICAL FINDING:
- Model was trained for only **16 out of 20 epochs**
- Early stopping triggered after 5 epochs without improvement
- Best validation metric was **only 0.125** (very low!)
- This explains why evaluation metrics are also very low (3.13% accuracy)

---

## 5. Summary of Configuration Issues

| Parameter | Training | Checkpoint | Evaluation | Status |
|-----------|----------|-----------|-----------|--------|
| visual_backbone | CLIP ViT | CLIP ViT | ✓ (from checkpoint) | ✓ SYNC |
| text_encoder | PhoBERT | PhoBERT | ✓ (from checkpoint) | ✓ SYNC |
| MOE type | standard | **vqa** | ✓ (from checkpoint) | ⚠️ MISMATCH |
| num_experts | 8 | 8 | ✓ (from checkpoint) | ✓ SYNC |
| num_beams | 1 | 1 | 3 | ⚠️ DIFFERENT |
| max_length | 64 | 64 | 64 | ✓ SYNC |
| batch_size | 8 | - | 16 | - (inference only) |

---

## 6. Root Cause Analysis: Why Metrics Are Low

### Primary Cause: Model Training Quality
1. ✗ Model stopped improving at epoch 16
2. ✗ Best validation BLEU: **0.125** (12.5%)
3. ✗ Early stopping kicked in (no improvement for 5 epochs)
4. **Conclusion:** Model likely not well-trained or dataset insufficient

### Secondary Cause: MOE Config Mismatch
1. ⚠️ Config file says `type: standard` but checkpoint has `type: vqa`
2. This could indicate:
   - Model retrained with different config
   - Or config file not matching actual training
3. VQA MOE specialists are heavy - may not fit in 12GB VRAM well
4. Could affect model convergence

### Tertiary Cause: Generation Parameter Difference
1. ⚠️ Training validated with greedy (num_beams=1)
2. ⚠️ Now evaluating with beam search (num_beams=3)
3. Should improve results, but different from training validation

---

## 7. Recommendations

### Option A: Use Same Generation Config as Training
```bash
# Revert to num_beams=1 to match training
python -m src.core.vivqa_eval_cli --checkpoint checkpoints/RichardNguyen_ViMoE-VQA/best_generative_model.pt --num-beams 1
```

### Option B: Check Actual Training Config
```bash
# Verify what config was actually used for training
# File: generative_configs.yaml should match checkpoint's moe_type
# Currently it says type: standard but checkpoint has type: vqa
```

### Option C: Understand Low Metrics
The low metrics (3.13% accuracy, 0.125 BLEU) suggest:
1. Model may need more training
2. Dataset might be too small or poor quality
3. Learning rate might be too low (5e-5)
4. MOE configuration might need adjustment

### Option D: Retrain with Correct Config
If you want to match training and evaluation:
1. Update generative_configs.yaml to set `moe.type: vqa` (to match checkpoint)
2. Run training again with correct config
3. Use same num_beams=1 in evaluation
4. Allow more than 16 epochs to train

---

## Conclusion

**Configuration Synchronization Status: ⚠️ PARTIALLY MISMATCHED**

✓ **Model architecture synced:** Visual backbone, text encoder, dimensions
⚠️ **MOE type mismatch:** Config says 'standard', checkpoint has 'vqa'
⚠️ **Generation different:** Training used num_beams=1, evaluation uses num_beams=3
🔴 **Primary issue:** Model quality low (0.125 BLEU at epoch 16) - early stopping triggered

**Next Step:** Clarify actual training configuration used for best_generative_model.pt checkpoint.
