# Ablation Study Report: moe_ablation_study

**Generated:** 2026-02-19 05:11:10
**Total experiments:** 22
**Primary metric:** `vqa_accuracy`

## 1. Key Findings

1. Baseline (all experts, default router): vqa_accuracy = 33.35%
2. Disabling MOE entirely: vqa_accuracy = 33.35% (+0.00% degradation from baseline)
3. Expert importance ranking (LOO drops): vision (+0.00%), text (+0.00%), multimodal (+0.00%), specialized (+0.00%)
4. Most important expert: vision (removing it drops vqa_accuracy by 0.00%)
5. Least important expert: specialized (removing it changes vqa_accuracy by only 0.00%)
6. Router sensitivity: low. Best: Router=noisy_topk | K=4 | noise=on | lb=0.0
7. Best overall experiment: subset_multimodal_specialized_vision__noisy_topk_k2 (vqa_accuracy = 33.35%)

## 2. Experiment Ranking

| Rank | Experiment ID | Score |
|------|---------------|-------|
| 1 | `subset_multimodal_specialized_vision__noisy_topk_k2` | 33.35% |
| 2 | `subset_multimodal_specialized_text__noisy_topk_k2` | 33.35% |
| 3 | `subset_multimodal_specialized__noisy_topk_k2` | 33.35% |
| 4 | `single_expert_vision__noisy_topk_k2` | 33.35% |
| 5 | `single_expert_text__noisy_topk_k2` | 33.35% |
| 6 | `single_expert_specialized__noisy_topk_k2` | 33.35% |
| 7 | `single_expert_multimodal__noisy_topk_k2` | 33.35% |
| 8 | `no_moe__noisy_topk_k2` | 33.35% |
| 9 | `leave_one_out_no_vision__noisy_topk_k2` | 33.35% |
| 10 | `leave_one_out_no_text__noisy_topk_k2` | 33.35% |
| 11 | `leave_one_out_no_specialized__noisy_topk_k2` | 33.35% |
| 12 | `leave_one_out_no_multimodal__noisy_topk_k2` | 33.35% |
| 13 | `full__noisy_topk_k4` | 33.35% |
| 14 | `full__noisy_topk_k2` | 33.35% |
| 15 | `full__noisy_topk_k1` | 33.35% |
| 16 | `subset_multimodal_text__noisy_topk_k2` | 33.35% |
| 17 | `subset_multimodal_text_vision__noisy_topk_k2` | 33.35% |
| 18 | `subset_multimodal_vision__noisy_topk_k2` | 33.35% |
| 19 | `subset_specialized_text__noisy_topk_k2` | 33.35% |
| 20 | `subset_specialized_text_vision__noisy_topk_k2` | 33.35% |
| 21 | `subset_text_vision__noisy_topk_k2` | 33.35% |
| 22 | `subset_specialized_vision__noisy_topk_k2` | 32.78% |

## 3. Expert Contribution Analysis

### 3.1 Importance Ranking (Leave-One-Out)

| Rank | Expert Type | Importance (Δ%) | Solo Score (%) | Essential? | Recommendation |
|------|-------------|-----------------|----------------|------------|----------------|
| 1 | vision | +0.00% | 33.35% | × | REDUNDANT: vision has minimal impact (0.00%). Consider remov... |
| 2 | text | +0.00% | 33.35% | × | REDUNDANT: text has minimal impact (0.00%). Consider removin... |
| 3 | multimodal | +0.00% | 33.35% | × | REDUNDANT: multimodal has minimal impact (0.00%). Consider r... |
| 4 | specialized | +0.00% | 33.35% | × | REDUNDANT: specialized has minimal impact (0.00%). Consider ... |

### 3.2 Pairwise Expert Synergy

| Expert A | Expert B | Pair Score (%) | Synergy (Δ%) | Synergistic? |
|----------|----------|----------------|--------------|--------------|
| vision | text | 33.35% | +0.00% | – |
| vision | multimodal | 33.35% | +0.00% | – |
| text | multimodal | 33.35% | +0.00% | – |
| text | specialized | 33.35% | +0.00% | – |
| multimodal | specialized | 33.35% | +0.00% | – |
| vision | specialized | 32.78% | -0.57% | – |

## 4. Router Analysis

- **Best router:** Router=noisy_topk | K=4 | noise=on | lb=0.0 (33.35%)
- **Worst router:** Router=noisy_topk | K=4 | noise=on | lb=0.0 (33.35%)
- **Sensitivity:** low
- **Best top_k:** 4

### Router Type Comparison

| Router Type | Avg Score (%) |
|-------------|---------------|
| noisy_topk | 33.35% |

## 5. Full Metric Table

| Experiment | Vqa Accuracy | Accuracy | Exact Match | Bleu | Meteor | Rouge L | Cider | Precision | Recall | F1 | Loss |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `subset_multimodal_specialized_vision__noisy_topk_k2` | 33.35 | 100.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 33.80 |
| `subset_multimodal_specialized_text__noisy_topk_k2` | 33.35 | 100.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 30.55 |
| `subset_multimodal_specialized__noisy_topk_k2` | 33.35 | 100.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 27.15 |
| `single_expert_vision__noisy_topk_k2` | 33.35 | 100.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 23.03 |
| `single_expert_text__noisy_topk_k2` | 33.35 | 100.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 34.73 |
| `single_expert_specialized__noisy_topk_k2` | 33.35 | 100.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 33.15 |
| `single_expert_multimodal__noisy_topk_k2` | 33.35 | 100.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 23.24 |
| `no_moe__noisy_topk_k2` | 33.35 | 100.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 35.73 |
| `leave_one_out_no_vision__noisy_topk_k2` | 33.35 | 100.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 26.84 |
| `leave_one_out_no_text__noisy_topk_k2` | 33.35 | 100.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 33.57 |
| `leave_one_out_no_specialized__noisy_topk_k2` | 33.35 | 100.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 28.66 |
| `leave_one_out_no_multimodal__noisy_topk_k2` | 33.35 | 100.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 35.90 |
| `full__noisy_topk_k4` | 33.35 | 100.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 31.90 |
| `full__noisy_topk_k2` | 33.35 | 100.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 24.10 |
| `full__noisy_topk_k1` | 33.35 | 100.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 33.05 |
| `subset_multimodal_text__noisy_topk_k2` | 33.35 | 100.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 30.95 |
| `subset_multimodal_text_vision__noisy_topk_k2` | 33.35 | 100.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 34.48 |
| `subset_multimodal_vision__noisy_topk_k2` | 33.35 | 100.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 36.00 |
| `subset_specialized_text__noisy_topk_k2` | 33.35 | 100.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 31.82 |
| `subset_specialized_text_vision__noisy_topk_k2` | 33.35 | 100.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 33.57 |
| `subset_specialized_vision__noisy_topk_k2` | 32.78 | 97.22 | 1.05 | 0.01 | 1.44 | 1.91 | 3.75 | 2.01 | 1.86 | 1.92 | 38.46 |
| `subset_text_vision__noisy_topk_k2` | 33.35 | 100.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 26.19 |

## 6. Performance Delta from Baseline

| Experiment | Δ (%) |
|------------|-------|
| `subset_multimodal_specialized_vision__noisy_topk_k2` | +0.00% = |
| `subset_multimodal_specialized_text__noisy_topk_k2` | +0.00% = |
| `subset_multimodal_specialized__noisy_topk_k2` | +0.00% = |
| `single_expert_vision__noisy_topk_k2` | +0.00% = |
| `single_expert_text__noisy_topk_k2` | +0.00% = |
| `single_expert_specialized__noisy_topk_k2` | +0.00% = |
| `single_expert_multimodal__noisy_topk_k2` | +0.00% = |
| `no_moe__noisy_topk_k2` | +0.00% = |
| `leave_one_out_no_vision__noisy_topk_k2` | +0.00% = |
| `leave_one_out_no_text__noisy_topk_k2` | +0.00% = |
| `leave_one_out_no_specialized__noisy_topk_k2` | +0.00% = |
| `leave_one_out_no_multimodal__noisy_topk_k2` | +0.00% = |
| `full__noisy_topk_k4` | +0.00% = |
| `full__noisy_topk_k2` | +0.00% = |
| `full__noisy_topk_k1` | +0.00% = |
| `subset_multimodal_text__noisy_topk_k2` | +0.00% = |
| `subset_multimodal_text_vision__noisy_topk_k2` | +0.00% = |
| `subset_multimodal_vision__noisy_topk_k2` | +0.00% = |
| `subset_specialized_text__noisy_topk_k2` | +0.00% = |
| `subset_specialized_text_vision__noisy_topk_k2` | +0.00% = |
| `subset_text_vision__noisy_topk_k2` | +0.00% = |
| `subset_specialized_vision__noisy_topk_k2` | -0.57% ↓ |

## 7. Optimal Configuration Recommendation

- **Recommended experts:** 
- **Recommended router:** noisy_topk
- **Recommended top_k:** 4
- **Expected score:** 33.35%
- **Reasoning:** Redundant experts (can remove): vision, text, multimodal, specialized. Best router: noisy_topk with top_k=4.

## 8. MOE-Specific Metrics

| Experiment | Load Balance Loss | Routing Entropy | Load Imbalance | Active Experts |
|---|---|---|---|---|
| `subset_multimodal_specialized_vision__noisy_topk_k2` | 0.0489 | 0.3260 | 0.0000 | 6 |
| `subset_multimodal_specialized_text__noisy_topk_k2` | 0.0643 | 1.0337 | 0.0000 | 6 |
| `subset_multimodal_specialized__noisy_topk_k2` | 0.0510 | 0.3101 | 0.0000 | 4 |
| `single_expert_vision__noisy_topk_k2` | 0.0574 | 0.8092 | 0.0000 | 2 |
| `single_expert_text__noisy_topk_k2` | 0.0710 | 0.6036 | 0.0000 | 2 |
| `single_expert_specialized__noisy_topk_k2` | 0.0736 | 0.4497 | 0.0000 | 2 |
| `single_expert_multimodal__noisy_topk_k2` | 0.0424 | 0.6907 | 0.0000 | 2 |
| `no_moe__noisy_topk_k2` | N/A | N/A | 0.0000 | N/A |
| `leave_one_out_no_vision__noisy_topk_k2` | 0.0487 | 0.3073 | 0.0000 | 6 |
| `leave_one_out_no_text__noisy_topk_k2` | 0.0434 | 0.7951 | 0.0000 | 6 |
| `leave_one_out_no_specialized__noisy_topk_k2` | 0.0482 | 0.2254 | 0.0000 | 6 |
| `leave_one_out_no_multimodal__noisy_topk_k2` | 0.0650 | 0.8607 | 0.0000 | 6 |
| `full__noisy_topk_k4` | 0.0575 | 0.2815 | 0.0000 | 8 |
| `full__noisy_topk_k2` | 0.0504 | 0.2368 | 0.0000 | 8 |
| `full__noisy_topk_k1` | 0.0141 | 1.9581 | 0.0000 | 8 |
| `subset_multimodal_text__noisy_topk_k2` | 0.0489 | 0.3108 | 0.0000 | 4 |
| `subset_multimodal_text_vision__noisy_topk_k2` | 0.0646 | 0.4618 | 0.0000 | 6 |
| `subset_multimodal_vision__noisy_topk_k2` | 0.0454 | 0.7772 | 0.0000 | 4 |
| `subset_specialized_text__noisy_topk_k2` | 0.0509 | 0.3323 | 0.0000 | 4 |
| `subset_specialized_text_vision__noisy_topk_k2` | 0.0493 | 0.2308 | 0.0000 | 6 |
| `subset_specialized_vision__noisy_topk_k2` | 0.0490 | 0.4390 | 0.0000 | 4 |
| `subset_text_vision__noisy_topk_k2` | 0.0488 | 0.1667 | 0.0000 | 4 |

---
*Report generated by AutoViVQA Ablation Pipeline — 2026-02-19 05:11*