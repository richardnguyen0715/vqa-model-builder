# Ablation Study Report: moe_ablation_study

**Generated:** 2026-02-25 17:40:36
**Total experiments:** 9
**Primary metric:** `bleu`

## 1. Key Findings

1. Baseline (all experts, default router): bleu = 6.39%
2. Expert importance ranking (LOO drops): vision (+0.00%), text (+0.00%), multimodal (+0.00%), specialized (+0.00%)
3. Most important expert: vision (removing it drops bleu by 0.00%)
4. Least important expert: specialized (removing it changes bleu by only 0.00%)
5. Strongest synergy: multimodal + specialized (synergy = +6.12% over best solo)
6. Router sensitivity: low. Best: Router=topk | K=4 | lb=0.0
7. Best overall experiment: full__topk_k4 (bleu = 6.45%)

## 2. Experiment Ranking

| Rank | Experiment ID | Score |
|------|---------------|-------|
| 1 | `full__topk_k4` | 6.45% |
| 2 | `full__topk_k2` | 6.39% |
| 3 | `subset_multimodal_specialized__noisy_topk_k2` | 6.12% |
| 4 | `subset_multimodal_specialized_text__noisy_topk_k2` | 6.12% |
| 5 | `subset_multimodal_text__noisy_topk_k2` | 4.99% |
| 6 | `subset_multimodal_text_vision__noisy_topk_k2` | 4.72% |
| 7 | `subset_multimodal_vision__noisy_topk_k2` | 4.72% |
| 8 | `subset_multimodal_specialized_vision__noisy_topk_k2` | 4.67% |
| 9 | `subset_specialized_vision__noisy_topk_k2` | 4.65% |

## 3. Expert Contribution Analysis

### 3.1 Importance Ranking (Leave-One-Out)

| Rank | Expert Type | Importance (Δ%) | Solo Score (%) | Essential? | Recommendation |
|------|-------------|-----------------|----------------|------------|----------------|
| 1 | vision | +0.00% | 0.00% | × | REDUNDANT: vision has minimal impact (0.00%). Consider remov... |
| 2 | text | +0.00% | 0.00% | × | REDUNDANT: text has minimal impact (0.00%). Consider removin... |
| 3 | multimodal | +0.00% | 0.00% | × | REDUNDANT: multimodal has minimal impact (0.00%). Consider r... |
| 4 | specialized | +0.00% | 0.00% | × | REDUNDANT: specialized has minimal impact (0.00%). Consider ... |

### 3.2 Pairwise Expert Synergy

| Expert A | Expert B | Pair Score (%) | Synergy (Δ%) | Synergistic? |
|----------|----------|----------------|--------------|--------------|
| multimodal | specialized | 6.12% | +6.12% | ✓ |
| text | multimodal | 4.99% | +4.99% | ✓ |
| vision | multimodal | 4.72% | +4.72% | ✓ |
| vision | specialized | 4.65% | +4.65% | ✓ |
| vision | text | 0.00% | +0.00% | – |
| text | specialized | 0.00% | +0.00% | – |

## 4. Router Analysis

- **Best router:** Router=topk | K=4 | lb=0.0 (6.45%)
- **Worst router:** Router=topk | K=2 | lb=0.0 (6.39%)
- **Sensitivity:** low
- **Best top_k:** 4

### Router Type Comparison

| Router Type | Avg Score (%) |
|-------------|---------------|
| topk | 6.42% |

## 5. Full Metric Table

| Experiment | Bleu | Meteor | Rouge L | Cider | Exact Match | Precision | Recall | F1 | Loss | Perplexity |
|---|---|---|---|---|---|---|---|---|---|---|
| `full__topk_k2` | 6.39 | 28.02 | 41.78 | 68.93 | 6.63 | 55.49 | 41.06 | 45.60 | 484.05 | 126.53 |
| `subset_multimodal_specialized_vision__noisy_topk_k2` | 4.67 | 25.72 | 40.32 | 60.05 | 6.26 | 54.62 | 38.56 | 43.54 | 497.43 | 144.65 |
| `full__topk_k4` | 6.45 | 28.23 | 41.99 | 69.08 | 6.23 | 55.66 | 41.44 | 45.85 | 483.38 | 125.69 |
| `subset_multimodal_specialized__noisy_topk_k2` | 6.12 | 28.04 | 41.98 | 67.37 | 6.85 | 55.79 | 40.97 | 45.58 | 484.88 | 127.59 |
| `subset_multimodal_text__noisy_topk_k2` | 4.99 | 26.09 | 40.30 | 61.92 | 6.36 | 54.32 | 38.96 | 43.78 | 496.32 | 143.05 |
| `subset_multimodal_text_vision__noisy_topk_k2` | 4.72 | 25.98 | 40.39 | 60.43 | 6.15 | 54.65 | 39.10 | 43.99 | 496.26 | 142.97 |
| `subset_multimodal_specialized_text__noisy_topk_k2` | 6.12 | 28.07 | 42.02 | 67.42 | 6.85 | 55.80 | 41.00 | 45.59 | 484.87 | 127.58 |
| `subset_multimodal_vision__noisy_topk_k2` | 4.72 | 25.98 | 40.39 | 60.43 | 6.15 | 54.65 | 39.10 | 43.99 | 496.26 | 142.97 |
| `subset_specialized_vision__noisy_topk_k2` | 4.65 | 25.94 | 40.29 | 61.08 | 6.36 | 54.40 | 38.83 | 43.68 | 497.34 | 144.52 |

## 6. Performance Delta from Baseline

| Experiment | Δ (%) |
|------------|-------|
| `full__topk_k4` | +0.05% ↑ |
| `full__topk_k2` | +0.00% = |
| `subset_multimodal_specialized__noisy_topk_k2` | -0.27% ↓ |
| `subset_multimodal_specialized_text__noisy_topk_k2` | -0.27% ↓ |
| `subset_multimodal_text__noisy_topk_k2` | -1.40% ↓ |
| `subset_multimodal_text_vision__noisy_topk_k2` | -1.67% ↓ |
| `subset_multimodal_vision__noisy_topk_k2` | -1.67% ↓ |
| `subset_multimodal_specialized_vision__noisy_topk_k2` | -1.73% ↓ |
| `subset_specialized_vision__noisy_topk_k2` | -1.75% ↓ |

## 7. Optimal Configuration Recommendation

- **Recommended experts:** 
- **Recommended router:** topk
- **Recommended top_k:** 4
- **Expected score:** 6.45%
- **Reasoning:** Redundant experts (can remove): vision, text, multimodal, specialized. Best router: topk with top_k=4.

## 8. MOE-Specific Metrics

| Experiment | Load Balance Loss | Routing Entropy | Load Imbalance | Active Experts |
|---|---|---|---|---|
| `full__topk_k2` | 0.0193 | 0.5262 | 0.0000 | 4 |
| `subset_multimodal_specialized_vision__noisy_topk_k2` | 0.0204 | 0.9219 | 0.0000 | 3 |
| `full__topk_k4` | 0.0400 | 0.5435 | 0.0000 | 4 |
| `subset_multimodal_specialized__noisy_topk_k2` | 0.0202 | 0.9252 | 0.0000 | 2 |
| `subset_multimodal_text__noisy_topk_k2` | 0.0203 | 0.9437 | 0.0000 | 2 |
| `subset_multimodal_text_vision__noisy_topk_k2` | 0.0217 | 0.6752 | 0.0000 | 3 |
| `subset_multimodal_specialized_text__noisy_topk_k2` | 0.0202 | 0.9258 | 0.0000 | 3 |
| `subset_multimodal_vision__noisy_topk_k2` | 0.0217 | 0.6752 | 0.0000 | 2 |
| `subset_specialized_vision__noisy_topk_k2` | 0.0211 | 0.7985 | 0.0000 | 2 |

---
*Report generated by AutoViVQA Ablation Pipeline — 2026-02-25 17:40*