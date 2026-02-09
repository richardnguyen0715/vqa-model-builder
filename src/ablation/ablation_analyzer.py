"""
Ablation Analyzer Module
==========================

High-level analysis engine that interprets ablation results and generates
actionable research insights. Builds on AblationEvaluator's raw metrics.

Computes:
- Expert contribution analysis (individual, pairwise synergy)
- Router sensitivity report
- Expert × Question-type effectiveness (if question type info available)
- Optimal MOE configuration recommendations
- Auto-generated key findings text
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.ablation.ablation_evaluator import (
    AblationEvaluator,
    ExpertPerformanceProfile,
    STANDARD_METRICS,
)


# ══════════════════════════════════════════════════════════════════════
# Analysis Dataclasses
# ══════════════════════════════════════════════════════════════════════

@dataclass
class ExpertContribution:
    """Quantified contribution of each expert type."""
    expert_type: str
    importance_rank: int = 0
    importance_score: float = 0.0       # baseline - LOO
    solo_score: float = 0.0             # single-expert performance
    marginal_contribution: float = 0.0  # importance_score normalized
    is_essential: bool = False          # If removal causes > threshold drop
    is_redundant: bool = False          # If removal causes < threshold drop
    recommendation: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "expert_type": self.expert_type,
            "importance_rank": self.importance_rank,
            "importance_score": round(self.importance_score, 6),
            "solo_score": round(self.solo_score, 6),
            "marginal_contribution": round(self.marginal_contribution, 6),
            "is_essential": self.is_essential,
            "is_redundant": self.is_redundant,
            "recommendation": self.recommendation,
        }


@dataclass
class PairwiseSynergy:
    """Synergy analysis between pairs of expert types."""
    expert_a: str
    expert_b: str
    pair_score: float = 0.0              # Subset(a,b) performance
    sum_solo: float = 0.0               # solo(a) + solo(b)
    synergy: float = 0.0                # pair_score - max(solo(a), solo(b))
    is_synergistic: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "expert_a": self.expert_a,
            "expert_b": self.expert_b,
            "pair_score": round(self.pair_score, 6),
            "sum_solo": round(self.sum_solo, 6),
            "synergy": round(self.synergy, 6),
            "is_synergistic": self.is_synergistic,
        }


@dataclass
class RouterAnalysis:
    """Analysis of router behaviour across experiments."""
    best_router: str = ""
    best_router_score: float = 0.0
    worst_router: str = ""
    worst_router_score: float = 0.0
    router_scores: Dict[str, float] = field(default_factory=dict)
    best_top_k: int = 2
    top_k_scores: Dict[int, float] = field(default_factory=dict)
    best_lb_weight: float = 0.01
    sensitivity_rating: str = ""  # "low", "medium", "high"
    recommendation: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "best_router": self.best_router,
            "best_router_score": round(self.best_router_score, 6),
            "worst_router": self.worst_router,
            "worst_router_score": round(self.worst_router_score, 6),
            "router_scores": self.router_scores,
            "best_top_k": self.best_top_k,
            "top_k_scores": self.top_k_scores,
            "best_lb_weight": self.best_lb_weight,
            "sensitivity_rating": self.sensitivity_rating,
            "recommendation": self.recommendation,
        }


@dataclass
class MOERecommendation:
    """Optimal MOE configuration recommendation."""
    recommended_experts: List[str] = field(default_factory=list)
    recommended_router: str = ""
    recommended_top_k: int = 2
    recommended_lb_weight: float = 0.01
    expected_score: float = 0.0
    reasoning: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "recommended_experts": self.recommended_experts,
            "recommended_router": self.recommended_router,
            "recommended_top_k": self.recommended_top_k,
            "recommended_lb_weight": self.recommended_lb_weight,
            "expected_score": round(self.expected_score, 6),
            "reasoning": self.reasoning,
        }


# ══════════════════════════════════════════════════════════════════════
# Ablation Analyzer
# ══════════════════════════════════════════════════════════════════════

class AblationAnalyzer:
    """Analyzes ablation results and generates insights.

    Takes an AblationEvaluator (or raw results dict) and produces:
    1. Expert contribution analysis with rankings
    2. Pairwise synergy matrix
    3. Router sensitivity analysis  
    4. Optimal configuration recommendation
    5. Auto-generated findings summary (text)
    """

    def __init__(
        self,
        evaluator: AblationEvaluator,
        primary_metric: str = "vqa_accuracy",
        essential_threshold: float = 0.02,     # >2% drop → essential
        redundant_threshold: float = 0.005,    # <0.5% drop → redundant
        synergy_threshold: float = 0.01,       # >1% above best solo → synergy
    ):
        self.evaluator = evaluator
        self.primary_metric = primary_metric
        self.essential_threshold = essential_threshold
        self.redundant_threshold = redundant_threshold
        self.synergy_threshold = synergy_threshold

    # ── Expert Contribution ───────────────────────────────────────

    def analyze_expert_contributions(self) -> List[ExpertContribution]:
        """Compute expert contribution analysis from LOO + single experiments."""
        profiles = self.evaluator.compute_expert_importance()

        total_importance = sum(abs(p.importance_score) for p in profiles) or 1.0
        contributions: List[ExpertContribution] = []

        for rank, profile in enumerate(profiles, 1):
            marginal = profile.importance_score / total_importance

            is_essential = profile.importance_score > self.essential_threshold
            is_redundant = abs(profile.importance_score) < self.redundant_threshold

            # Generate recommendation
            if is_essential:
                rec = f"ESSENTIAL: Removing {profile.expert_type} causes {profile.importance_score*100:.2f}% performance drop. Must keep."
            elif is_redundant:
                rec = f"REDUNDANT: {profile.expert_type} has minimal impact ({profile.importance_score*100:.2f}%). Consider removing to save compute."
            else:
                rec = f"MODERATE: {profile.expert_type} contributes {profile.importance_score*100:.2f}%. Keep if compute allows."

            contributions.append(ExpertContribution(
                expert_type=profile.expert_type,
                importance_rank=rank,
                importance_score=profile.importance_score,
                solo_score=profile.solo_score,
                marginal_contribution=marginal,
                is_essential=is_essential,
                is_redundant=is_redundant,
                recommendation=rec,
            ))

        return contributions

    # ── Pairwise Synergy ──────────────────────────────────────────

    def analyze_pairwise_synergy(self) -> List[PairwiseSynergy]:
        """Compute synergy between all pairs of expert types.

        Synergy = pair_performance - max(solo_a, solo_b)
        Positive synergy means the pair works better together than each alone.
        """
        expert_types = ["vision", "text", "multimodal", "specialized"]

        # Get solo scores
        solo_scores: Dict[str, float] = {}
        for exp_id, result in self.evaluator.results.items():
            ec = result.get("experiment_config", {}).get("expert_config", {})
            if ec.get("mode") == "single_expert":
                active = ec.get("active_experts", [])
                if len(active) == 1:
                    val = result.get("metrics", {}).get(self.primary_metric, 0.0)
                    solo_scores[active[0]] = val

        # Get pair scores (subset experiments with exactly 2 experts)
        pair_scores: Dict[Tuple[str, str], float] = {}
        for exp_id, result in self.evaluator.results.items():
            ec = result.get("experiment_config", {}).get("expert_config", {})
            if ec.get("mode") == "subset":
                active = sorted(ec.get("active_experts", []))
                if len(active) == 2:
                    val = result.get("metrics", {}).get(self.primary_metric, 0.0)
                    pair_scores[tuple(active)] = val

        # Compute synergy
        synergies: List[PairwiseSynergy] = []
        import itertools
        for a, b in itertools.combinations(expert_types, 2):
            key = tuple(sorted([a, b]))
            pair_val = pair_scores.get(key, 0.0)
            solo_a = solo_scores.get(a, 0.0)
            solo_b = solo_scores.get(b, 0.0)
            best_solo = max(solo_a, solo_b)
            synergy_val = pair_val - best_solo

            synergies.append(PairwiseSynergy(
                expert_a=a,
                expert_b=b,
                pair_score=pair_val,
                sum_solo=solo_a + solo_b,
                synergy=synergy_val,
                is_synergistic=synergy_val > self.synergy_threshold,
            ))

        synergies.sort(key=lambda s: s.synergy, reverse=True)
        return synergies

    # ── Router Analysis ───────────────────────────────────────────

    def analyze_routers(self) -> RouterAnalysis:
        """Analyze router performance and sensitivity."""
        router_comparison = self.evaluator.compute_router_comparison()

        analysis = RouterAnalysis()

        if not router_comparison:
            analysis.recommendation = "No router ablation data available."
            return analysis

        # Extract router type scores
        router_type_scores: Dict[str, List[float]] = {}
        top_k_scores: Dict[int, List[float]] = {}
        lb_weight_scores: Dict[float, List[float]] = {}

        best_overall = -1.0
        worst_overall = float("inf")

        for desc, metrics in router_comparison.items():
            val = metrics.get(self.primary_metric, 0.0)

            # Parse description → extract router_type, K, lb
            # Expected format: "Router=noisy_topk | K=2 | noise=on | lb=0.01"
            parts = desc.split(" | ")
            rt = ""
            k_val = 2
            lb_val = 0.01
            for p in parts:
                p = p.strip()
                if p.startswith("Router="):
                    rt = p.replace("Router=", "")
                elif p.startswith("K="):
                    try:
                        k_val = int(p.replace("K=", ""))
                    except ValueError:
                        pass
                elif p.startswith("lb="):
                    try:
                        lb_val = float(p.replace("lb=", ""))
                    except ValueError:
                        pass

            if rt:
                router_type_scores.setdefault(rt, []).append(val)
            top_k_scores.setdefault(k_val, []).append(val)
            lb_weight_scores.setdefault(lb_val, []).append(val)

            if val > best_overall:
                best_overall = val
                analysis.best_router = desc
                analysis.best_router_score = val
            if val < worst_overall:
                worst_overall = val
                analysis.worst_router = desc
                analysis.worst_router_score = val

        # Average per router type
        for rt, vals in router_type_scores.items():
            analysis.router_scores[rt] = sum(vals) / len(vals)

        # Best top_k
        for k, vals in top_k_scores.items():
            avg = sum(vals) / len(vals)
            analysis.top_k_scores[k] = avg
        if analysis.top_k_scores:
            analysis.best_top_k = max(analysis.top_k_scores, key=analysis.top_k_scores.get)

        # Sensitivity rating
        if router_comparison:
            vals = [m.get(self.primary_metric, 0.0) for m in router_comparison.values()]
            spread = max(vals) - min(vals) if vals else 0
            if spread > 0.05:
                analysis.sensitivity_rating = "high"
            elif spread > 0.02:
                analysis.sensitivity_rating = "medium"
            else:
                analysis.sensitivity_rating = "low"

        # Recommendation
        if analysis.router_scores:
            best_rt = max(analysis.router_scores, key=analysis.router_scores.get)
            analysis.recommendation = (
                f"Best router: {best_rt} (avg {analysis.router_scores[best_rt]*100:.2f}%). "
                f"Best top_k: {analysis.best_top_k}. "
                f"Router sensitivity: {analysis.sensitivity_rating}."
            )

        return analysis

    # ── Configuration Recommendation ──────────────────────────────

    def recommend_optimal_config(self) -> MOERecommendation:
        """Synthesize analyses into an optimal MOE configuration recommendation."""
        contributions = self.analyze_expert_contributions()
        router_analysis = self.analyze_routers()
        ranking = self.evaluator.rank_experiments(self.primary_metric)

        rec = MOERecommendation()

        # Recommended experts: keep essential + moderate, remove redundant
        for c in contributions:
            if not c.is_redundant:
                rec.recommended_experts.append(c.expert_type)

        # Router from analysis
        if router_analysis.router_scores:
            rec.recommended_router = max(
                router_analysis.router_scores,
                key=router_analysis.router_scores.get,
            )
        rec.recommended_top_k = router_analysis.best_top_k

        # Expected score: find the closest matching experiment
        for exp_id, val in ranking:
            result = self.evaluator.results[exp_id]
            ec = result.get("experiment_config", {}).get("expert_config", {})
            active = sorted(ec.get("active_experts", []))
            if active == sorted(rec.recommended_experts):
                rec.expected_score = val
                break
        if rec.expected_score == 0.0 and ranking:
            rec.expected_score = ranking[0][1]

        # Reasoning
        essential = [c.expert_type for c in contributions if c.is_essential]
        redundant = [c.expert_type for c in contributions if c.is_redundant]
        reasoning_parts = []
        if essential:
            reasoning_parts.append(f"Essential experts: {', '.join(essential)}")
        if redundant:
            reasoning_parts.append(f"Redundant experts (can remove): {', '.join(redundant)}")
        reasoning_parts.append(
            f"Best router: {rec.recommended_router} with top_k={rec.recommended_top_k}"
        )
        rec.reasoning = ". ".join(reasoning_parts) + "."

        return rec

    # ── Key Findings ──────────────────────────────────────────────

    def generate_key_findings(self) -> List[str]:
        """Auto-generate a list of key findings as natural language statements."""
        findings: List[str] = []
        contributions = self.analyze_expert_contributions()
        synergies = self.analyze_pairwise_synergy()
        router_analysis = self.analyze_routers()
        ranking = self.evaluator.rank_experiments(self.primary_metric)
        deltas = self.evaluator.compute_deltas_from_baseline(self.primary_metric)

        # 1. Baseline performance
        for exp_id, result in self.evaluator.results.items():
            ec = result.get("experiment_config", {}).get("expert_config", {})
            if ec.get("mode") == "full":
                val = result.get("metrics", {}).get(self.primary_metric, 0)
                findings.append(
                    f"Baseline (all experts, default router): {self.primary_metric} = {val*100:.2f}%"
                )
                break

        # 2. No-MOE comparison
        for exp_id, result in self.evaluator.results.items():
            ec = result.get("experiment_config", {}).get("expert_config", {})
            if ec.get("mode") == "no_moe":
                val = result.get("metrics", {}).get(self.primary_metric, 0)
                delta = deltas.get(exp_id, 0)
                direction = "improvement" if delta > 0 else "degradation"
                findings.append(
                    f"Disabling MOE entirely: {self.primary_metric} = {val*100:.2f}% "
                    f"({delta*100:+.2f}% {direction} from baseline)"
                )
                break

        # 3. Expert importance ranking
        if contributions:
            ranking_str = ", ".join(
                f"{c.expert_type} ({c.importance_score*100:+.2f}%)"
                for c in contributions
            )
            findings.append(f"Expert importance ranking (LOO drops): {ranking_str}")

        # 4. Most and least important
        if contributions:
            most = contributions[0]
            findings.append(
                f"Most important expert: {most.expert_type} "
                f"(removing it drops {self.primary_metric} by {most.importance_score*100:.2f}%)"
            )
            least = contributions[-1]
            if least.is_redundant:
                findings.append(
                    f"Least important expert: {least.expert_type} "
                    f"(removing it changes {self.primary_metric} by only {least.importance_score*100:.2f}%)"
                )

        # 5. Best pair synergy
        best_synergy = synergies[0] if synergies else None
        if best_synergy and best_synergy.is_synergistic:
            findings.append(
                f"Strongest synergy: {best_synergy.expert_a} + {best_synergy.expert_b} "
                f"(synergy = +{best_synergy.synergy*100:.2f}% over best solo)"
            )

        # 6. Router findings
        if router_analysis.router_scores:
            findings.append(
                f"Router sensitivity: {router_analysis.sensitivity_rating}. "
                f"Best: {router_analysis.best_router}"
            )

        # 7. Best overall experiment
        if ranking:
            best_id, best_val = ranking[0]
            findings.append(
                f"Best overall experiment: {best_id} "
                f"({self.primary_metric} = {best_val*100:.2f}%)"
            )

        return findings

    # ── Full Analysis ─────────────────────────────────────────────

    def run_full_analysis(self) -> Dict[str, Any]:
        """Run all analyses and return a complete report dictionary."""
        return {
            "expert_contributions": [
                c.to_dict() for c in self.analyze_expert_contributions()
            ],
            "pairwise_synergy": [
                s.to_dict() for s in self.analyze_pairwise_synergy()
            ],
            "router_analysis": self.analyze_routers().to_dict(),
            "recommendation": self.recommend_optimal_config().to_dict(),
            "key_findings": self.generate_key_findings(),
            "evaluation_data": self.evaluator.to_dict(),
        }

    def save_analysis(self, path: str) -> None:
        """Save full analysis to JSON."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.run_full_analysis(), f, indent=2, default=str)
