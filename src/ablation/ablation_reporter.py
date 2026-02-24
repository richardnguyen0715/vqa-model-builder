"""
Ablation Reporter Module
=========================

Generates human-readable reports from ablation analysis results.
Output formats:
- Markdown (.md) — primary, research-paper quality
- CSV — for spreadsheet analysis
- JSON — machine-readable (already in evaluator/analyzer)
- LaTeX tables (optional, for paper submission)
"""

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.ablation.ablation_analyzer import AblationAnalyzer
from src.ablation.ablation_evaluator import (
    AblationEvaluator,
    STANDARD_METRICS,
    GENERATIVE_METRICS,
    CLASSIFICATION_METRICS,
)


class AblationReporter:
    """Generates reports from ablation study results.

    Takes an AblationAnalyzer (which wraps an AblationEvaluator) and
    generates formatted reports in multiple output formats.
    """

    def __init__(
        self,
        analyzer: AblationAnalyzer,
        output_dir: str = "outputs/ablation",
        study_name: str = "moe_ablation_study",
    ):
        self.analyzer = analyzer
        self.evaluator = analyzer.evaluator
        self.output_dir = Path(output_dir)
        self.study_name = study_name
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ══════════════════════════════════════════════════════════════
    # Markdown Report
    # ══════════════════════════════════════════════════════════════

    def generate_markdown_report(self) -> str:
        """Generate a comprehensive Markdown ablation report."""
        analysis = self.analyzer.run_full_analysis()
        lines: List[str] = []

        # Header
        lines.append(f"# Ablation Study Report: {self.study_name}")
        lines.append(f"")
        lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"**Total experiments:** {analysis['evaluation_data']['num_experiments']}")
        lines.append(f"**Primary metric:** `{self.analyzer.primary_metric}`")
        lines.append(f"")

        # ── Key Findings ──
        lines.append("## 1. Key Findings")
        lines.append("")
        for i, finding in enumerate(analysis["key_findings"], 1):
            lines.append(f"{i}. {finding}")
        lines.append("")

        # ── Experiment Ranking ──
        lines.append("## 2. Experiment Ranking")
        lines.append("")
        ranking = analysis["evaluation_data"]["ranking"]
        lines.append("| Rank | Experiment ID | Score |")
        lines.append("|------|---------------|-------|")
        for rank, (exp_id, score) in enumerate(ranking, 1):
            lines.append(f"| {rank} | `{exp_id}` | {score*100:.2f}% |")
        lines.append("")

        # ── Expert Contribution Analysis ──
        lines.append("## 3. Expert Contribution Analysis")
        lines.append("")
        contributions = analysis["expert_contributions"]
        lines.append("### 3.1 Importance Ranking (Leave-One-Out)")
        lines.append("")
        lines.append("| Rank | Expert Type | Importance (Δ%) | Solo Score (%) | Essential? | Recommendation |")
        lines.append("|------|-------------|-----------------|----------------|------------|----------------|")
        for c in contributions:
            ess = "✓" if c["is_essential"] else ("×" if c["is_redundant"] else "–")
            lines.append(
                f"| {c['importance_rank']} | {c['expert_type']} | "
                f"{c['importance_score']*100:+.2f}% | {c['solo_score']*100:.2f}% | "
                f"{ess} | {c['recommendation'][:60]}... |"
            )
        lines.append("")

        # ── Pairwise Synergy ──
        lines.append("### 3.2 Pairwise Expert Synergy")
        lines.append("")
        synergies = analysis["pairwise_synergy"]
        lines.append("| Expert A | Expert B | Pair Score (%) | Synergy (Δ%) | Synergistic? |")
        lines.append("|----------|----------|----------------|--------------|--------------|")
        for s in synergies:
            syn = "✓" if s["is_synergistic"] else "–"
            lines.append(
                f"| {s['expert_a']} | {s['expert_b']} | "
                f"{s['pair_score']*100:.2f}% | {s['synergy']*100:+.2f}% | {syn} |"
            )
        lines.append("")

        # ── Router Analysis ──
        lines.append("## 4. Router Analysis")
        lines.append("")
        router = analysis["router_analysis"]
        if router["router_scores"]:
            lines.append(f"- **Best router:** {router['best_router']} ({router['best_router_score']*100:.2f}%)")
            lines.append(f"- **Worst router:** {router['worst_router']} ({router['worst_router_score']*100:.2f}%)")
            lines.append(f"- **Sensitivity:** {router['sensitivity_rating']}")
            lines.append(f"- **Best top_k:** {router['best_top_k']}")
            lines.append("")
            lines.append("### Router Type Comparison")
            lines.append("")
            lines.append("| Router Type | Avg Score (%) |")
            lines.append("|-------------|---------------|")
            for rt, score in sorted(
                router["router_scores"].items(),
                key=lambda x: x[1],
                reverse=True,
            ):
                lines.append(f"| {rt} | {score*100:.2f}% |")
            lines.append("")
        else:
            lines.append("*No router ablation data available.*")
            lines.append("")

        # ── Full Metric Table ──
        lines.append("## 5. Full Metric Table")
        lines.append("")
        metric_table = analysis["evaluation_data"]["metric_table"]
        if metric_table:
            # Use model-type-aware metric list
            all_metrics = self.evaluator.relevant_metrics
            header = "| Experiment | " + " | ".join(m.replace("_", " ").title() for m in all_metrics) + " |"
            sep = "|" + "|".join(["---"] * (len(all_metrics) + 1)) + "|"
            lines.append(header)
            lines.append(sep)
            for exp_id, metrics in metric_table.items():
                vals = []
                for m in all_metrics:
                    v = metrics.get(m, 0)
                    # Format perplexity without multiplying by 100
                    if m == "perplexity":
                        vals.append(f"{v:.2f}")
                    else:
                        vals.append(f"{v*100:.2f}")
                lines.append(f"| `{exp_id}` | " + " | ".join(vals) + " |")
            lines.append("")

        # ── Delta from Baseline ──
        lines.append("## 6. Performance Delta from Baseline")
        lines.append("")
        deltas = analysis["evaluation_data"]["deltas"]
        lines.append("| Experiment | Δ (%) |")
        lines.append("|------------|-------|")
        for exp_id, delta in sorted(deltas.items(), key=lambda x: x[1], reverse=True):
            marker = "↑" if delta > 0 else ("↓" if delta < 0 else "=")
            lines.append(f"| `{exp_id}` | {delta*100:+.2f}% {marker} |")
        lines.append("")

        # ── Recommendation ──
        lines.append("## 7. Optimal Configuration Recommendation")
        lines.append("")
        rec = analysis["recommendation"]
        lines.append(f"- **Recommended experts:** {', '.join(rec['recommended_experts'])}")
        lines.append(f"- **Recommended router:** {rec['recommended_router']}")
        lines.append(f"- **Recommended top_k:** {rec['recommended_top_k']}")
        lines.append(f"- **Expected score:** {rec['expected_score']*100:.2f}%")
        lines.append(f"- **Reasoning:** {rec['reasoning']}")
        lines.append("")

        # ── MOE Metrics ──
        lines.append("## 8. MOE-Specific Metrics")
        lines.append("")
        moe_table = analysis["evaluation_data"]["moe_metric_table"]
        if moe_table:
            moe_keys = ["load_balance_loss", "routing_entropy", "load_imbalance", "active_experts"]
            header = "| Experiment | " + " | ".join(k.replace("_", " ").title() for k in moe_keys) + " |"
            sep = "|" + "|".join(["---"] * (len(moe_keys) + 1)) + "|"
            lines.append(header)
            lines.append(sep)
            for exp_id, metrics in moe_table.items():
                vals = []
                for k in moe_keys:
                    v = metrics.get(k)
                    if v is None:
                        vals.append("N/A")
                    elif isinstance(v, float):
                        vals.append(f"{v:.4f}")
                    else:
                        vals.append(str(v))
                lines.append(f"| `{exp_id}` | " + " | ".join(vals) + " |")
            lines.append("")

        lines.append("---")
        lines.append(
            f"*Report generated by AutoViVQA Ablation Pipeline — "
            f"{datetime.now().strftime('%Y-%m-%d %H:%M')}*"
        )

        return "\n".join(lines)

    # ══════════════════════════════════════════════════════════════
    # CSV Export
    # ══════════════════════════════════════════════════════════════

    def export_csv(self, filename: str = "ablation_results.csv") -> str:
        """Export experiment metrics to CSV."""
        path = self.output_dir / filename
        metric_table = self.evaluator.build_metric_table()

        if not metric_table:
            return str(path)

        all_metrics = self.evaluator.relevant_metrics
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["experiment_id"] + all_metrics)
            for exp_id, metrics in metric_table.items():
                row = [exp_id] + [metrics.get(m, 0.0) for m in all_metrics]
                writer.writerow(row)

        return str(path)

    def export_expert_contributions_csv(
        self, filename: str = "expert_contributions.csv"
    ) -> str:
        """Export expert contribution analysis to CSV."""
        path = self.output_dir / filename
        contributions = self.analyzer.analyze_expert_contributions()

        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "rank", "expert_type", "importance_score",
                "solo_score", "marginal_contribution",
                "is_essential", "is_redundant", "recommendation",
            ])
            for c in contributions:
                writer.writerow([
                    c.importance_rank, c.expert_type,
                    c.importance_score, c.solo_score,
                    c.marginal_contribution, c.is_essential,
                    c.is_redundant, c.recommendation,
                ])

        return str(path)

    # ══════════════════════════════════════════════════════════════
    # LaTeX Tables
    # ══════════════════════════════════════════════════════════════

    def generate_latex_table(self) -> str:
        """Generate LaTeX table for paper submission."""
        ranking = self.evaluator.rank_experiments(self.analyzer.primary_metric)
        metric_table = self.evaluator.build_metric_table()

        lines: List[str] = []

        # Select key metrics based on model type
        if self.evaluator.model_type == "generative":
            selected_metrics = ["bleu", "meteor", "rouge_l", "cider", "exact_match", "f1"]
        else:
            selected_metrics = ["vqa_accuracy", "accuracy", "exact_match", "f1"]

        lines.append(r"\begin{table}[h]")
        lines.append(r"\centering")
        lines.append(r"\caption{Ablation Study Results for MOE Configurations}")
        lines.append(r"\label{tab:ablation}")
        cols = "l" + "c" * len(selected_metrics)
        lines.append(r"\begin{tabular}{" + cols + "}")
        lines.append(r"\toprule")

        # Header
        header = "Experiment & " + " & ".join(
            m.replace("_", " ").title() for m in selected_metrics
        ) + r" \\"
        lines.append(header)
        lines.append(r"\midrule")

        # Find best values for bolding
        best_vals = {}
        for m in selected_metrics:
            vals = [metric_table[e].get(m, 0) for e in metric_table if e in metric_table]
            if "loss" in m:
                best_vals[m] = min(vals) if vals else 0
            else:
                best_vals[m] = max(vals) if vals else 0

        # Rows (use ranking order)
        for exp_id, _ in ranking:
            if exp_id not in metric_table:
                continue
            metrics = metric_table[exp_id]
            vals = []
            for m in selected_metrics:
                v = metrics.get(m, 0.0)
                formatted = f"{v*100:.2f}"
                if abs(v - best_vals.get(m, 0)) < 1e-6:
                    formatted = r"\textbf{" + formatted + "}"
                vals.append(formatted)
            # Shorten experiment ID for LaTeX
            short_id = exp_id.replace("__", "/").replace("_", r"\_")
            lines.append(f"{short_id} & " + " & ".join(vals) + r" \\")

        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")
        lines.append(r"\end{table}")

        return "\n".join(lines)

    # ══════════════════════════════════════════════════════════════
    # Save All Reports
    # ══════════════════════════════════════════════════════════════

    def save_all_reports(self) -> Dict[str, str]:
        """Generate and save all report formats.

        Returns:
            Dict mapping report type to file path.
        """
        paths: Dict[str, str] = {}

        # Markdown
        md_path = self.output_dir / f"{self.study_name}_report.md"
        md_content = self.generate_markdown_report()
        md_path.write_text(md_content)
        paths["markdown"] = str(md_path)

        # CSV
        paths["csv_results"] = self.export_csv()
        paths["csv_contributions"] = self.export_expert_contributions_csv()

        # LaTeX
        latex_path = self.output_dir / f"{self.study_name}_table.tex"
        latex_content = self.generate_latex_table()
        latex_path.write_text(latex_content)
        paths["latex"] = str(latex_path)

        # Full analysis JSON
        analysis_path = self.output_dir / f"{self.study_name}_analysis.json"
        self.analyzer.save_analysis(str(analysis_path))
        paths["json_analysis"] = str(analysis_path)

        # Raw results JSON
        raw_path = self.output_dir / f"{self.study_name}_raw_results.json"
        self.evaluator.save_json(str(raw_path))
        paths["json_raw"] = str(raw_path)

        return paths
