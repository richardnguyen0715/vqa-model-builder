"""
Ablation Runner Module
=======================

Main orchestrator for the ablation study pipeline.
Single entry point that:
1. Loads config (YAML / dict / AblationConfig)
2. Generates experiment matrix
3. Runs experiments sequentially (with resume support)
4. Collects results into evaluator
5. Runs analyzer
6. Generates reports

Usage:
    runner = AblationRunner.from_yaml("configs/ablation_config.yaml")
    runner.run()
"""

import gc
import json
import logging
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.ablation.ablation_analyzer import AblationAnalyzer
from src.ablation.ablation_config import AblationConfig, ExperimentConfig
from src.ablation.ablation_evaluator import AblationEvaluator
from src.ablation.ablation_reporter import AblationReporter
from src.ablation.ablation_trainer import AblationTrainer, ExperimentResult
from src.core.pipeline_logger import PipelineLogger, get_pipeline_logger


logger = logging.getLogger(__name__)


class AblationRunner:
    """Orchestrates the complete ablation study.

    Lifecycle:
        1. __init__  — receive model, data, config
        2. run()     — execute all experiments + analysis + reporting
        3. Results saved to output_dir/

    Features:
        - Resume support: skips experiments whose results are already on disk
        - Progress tracking: writes progress.json after each experiment
        - Error isolation: failed experiments don't crash the pipeline
        - Memory management: deep-copies model per experiment, clears GPU cache
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: AblationConfig,
        device: Optional[torch.device] = None,
        pipeline_logger: Optional[PipelineLogger] = None,
        model_config: Optional[Any] = None,
        vocabulary: Optional[Dict[str, int]] = None,
        id2answer: Optional[Dict[int, str]] = None,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.logger = pipeline_logger or get_pipeline_logger()
        self.model_config = model_config
        self.vocabulary = vocabulary
        self.id2answer = id2answer or {}

        # Paths
        self.output_dir = Path(config.output_dir)
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.log_dir = Path(config.log_dir)
        self.progress_path = self.output_dir / "progress.json"
        self.results_dir = self.output_dir / "experiment_results"

        # State
        self.experiments: List[ExperimentConfig] = []
        self.results: List[ExperimentResult] = []
        self.completed_ids: set = set()

    # ── Factory ───────────────────────────────────────────────────

    @classmethod
    def from_yaml(
        cls,
        yaml_path: str,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        **kwargs,
    ) -> "AblationRunner":
        """Create runner from YAML config file."""
        config = AblationConfig.from_yaml(yaml_path)
        return cls(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            **kwargs,
        )

    # ── Main Entry Point ──────────────────────────────────────────

    def run(self) -> Dict[str, Any]:
        """Execute the complete ablation study.

        Returns:
            Dict with summary including report paths and key findings.
        """
        self.logger.start_stage("ABLATION STUDY")
        start_time = time.time()

        try:
            # 1. Setup directories
            self._setup_directories()

            # 2. Generate experiment matrix
            self.experiments = self.config.generate_experiment_matrix()
            self.logger.info(f"Generated {len(self.experiments)} experiments")
            self._save_experiment_manifest()

            # 3. Load any previously completed results (resume)
            if self.config.resume:
                self._load_completed_results()

            # 4. Run experiments
            self._run_all_experiments()

            # 5. Evaluate & Analyze
            evaluator = self._build_evaluator()
            analyzer = AblationAnalyzer(evaluator, primary_metric="vqa_accuracy")

            # 6. Generate reports
            reporter = AblationReporter(
                analyzer=analyzer,
                output_dir=str(self.output_dir),
                study_name=self.config.name,
            )
            report_paths = reporter.save_all_reports()

            # 7. Summary
            total_time = time.time() - start_time
            key_findings = analyzer.generate_key_findings()

            summary = {
                "study_name": self.config.name,
                "total_experiments": len(self.experiments),
                "completed": sum(1 for r in self.results if r.status == "completed"),
                "failed": sum(1 for r in self.results if r.status == "failed"),
                "skipped": sum(1 for r in self.results if r.status == "skipped"),
                "total_time_seconds": total_time,
                "total_time_human": self._format_time(total_time),
                "key_findings": key_findings,
                "report_paths": report_paths,
                "best_experiment": self._get_best_experiment(),
            }

            # Save summary
            summary_path = self.output_dir / f"{self.config.name}_summary.json"
            with open(summary_path, "w") as f:
                json.dump(summary, f, indent=2, default=str)

            self._log_final_summary(summary)
            self.logger.end_stage("ABLATION STUDY", success=True)

            return summary

        except Exception as e:
            self.logger.error(f"Ablation study failed: {e}\n{traceback.format_exc()}")
            self.logger.end_stage("ABLATION STUDY", success=False)
            raise

    # ── Internal Methods ──────────────────────────────────────────

    def _setup_directories(self):
        """Create all output directories."""
        for d in [self.output_dir, self.checkpoint_dir, self.log_dir, self.results_dir]:
            d.mkdir(parents=True, exist_ok=True)

    def _save_experiment_manifest(self):
        """Save the experiment matrix to disk for reference."""
        manifest = {
            "study_name": self.config.name,
            "generated_at": datetime.now().isoformat(),
            "num_experiments": len(self.experiments),
            "experiments": [e.to_dict() for e in self.experiments],
            "config": self.config.to_dict(),
        }
        manifest_path = self.output_dir / "experiment_manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
        self.logger.info(f"Experiment manifest saved: {manifest_path}")

    def _load_completed_results(self):
        """Load results from previous runs for resume support."""
        loaded = 0
        for result_file in self.results_dir.glob("*.json"):
            try:
                with open(result_file) as f:
                    data = json.load(f)
                if data.get("status") == "completed":
                    result = ExperimentResult(
                        experiment_id=data["experiment_id"],
                        experiment_config=data.get("experiment_config"),
                        metrics=data.get("metrics", {}),
                        moe_metrics=data.get("moe_metrics", {}),
                        training_history=data.get("training_history", []),
                        validation_history=data.get("validation_history", []),
                        total_epochs=data.get("total_epochs", 0),
                        training_time=data.get("training_time", 0.0),
                        best_metric=data.get("best_metric", 0.0),
                        checkpoint_path=data.get("checkpoint_path", ""),
                        status="completed",
                    )
                    self.results.append(result)
                    self.completed_ids.add(data["experiment_id"])
                    loaded += 1
            except Exception as e:
                self.logger.warning(f"Failed to load result {result_file}: {e}")

        if loaded:
            self.logger.info(f"Resumed {loaded} completed experiments from disk")

    def _run_all_experiments(self):
        """Run all experiments in the matrix."""
        trainer = AblationTrainer(
            base_model=self.model,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            ablation_config=self.config,
            device=self.device,
            logger=self.logger,
            model_config=self.model_config,
            vocabulary=self.vocabulary,
            id2answer=self.id2answer,
        )

        total = len(self.experiments)
        completed_count = len(self.completed_ids)

        for idx, experiment in enumerate(self.experiments, 1):
            # Skip completed
            if experiment.experiment_id in self.completed_ids:
                self.logger.info(
                    f"[{idx}/{total}] Skipping (already completed): "
                    f"{experiment.experiment_id}"
                )
                continue

            self.logger.section(
                f"Experiment [{idx}/{total}]: {experiment.experiment_id}"
            )
            self.logger.info(f"Expert: {experiment.expert_config.description}")
            self.logger.info(f"Router: {experiment.router_config.description}")

            # Run
            result = trainer.run_experiment(experiment)
            self.results.append(result)

            # Save result to disk (for resume)
            self._save_experiment_result(result)

            # Update progress
            completed_count += 1
            self._save_progress(idx, total, completed_count)

            # Memory cleanup between experiments
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            self.logger.info(
                f"Progress: {completed_count}/{total} experiments "
                f"({completed_count/total*100:.0f}%)"
            )

    def _save_experiment_result(self, result: ExperimentResult):
        """Save single experiment result to disk."""
        safe_id = result.experiment_id.replace("/", "_").replace("\\", "_")
        result_path = self.results_dir / f"{safe_id}.json"
        with open(result_path, "w") as f:
            json.dump(result.to_dict(), f, indent=2, default=str)

    def _save_progress(self, current: int, total: int, completed: int):
        """Save progress state."""
        progress = {
            "current_index": current,
            "total_experiments": total,
            "completed": completed,
            "failed": sum(1 for r in self.results if r.status == "failed"),
            "last_updated": datetime.now().isoformat(),
            "completed_ids": list(self.completed_ids)
            + [r.experiment_id for r in self.results if r.status == "completed"],
        }
        with open(self.progress_path, "w") as f:
            json.dump(progress, f, indent=2)

    def _build_evaluator(self) -> AblationEvaluator:
        """Build evaluator from all collected results."""
        evaluator = AblationEvaluator(primary_metric="vqa_accuracy")
        for result in self.results:
            if result.status == "completed":
                evaluator.add_result(result.to_dict())
        return evaluator

    def _get_best_experiment(self) -> Dict[str, Any]:
        """Find the best experiment by primary metric."""
        best_result = None
        best_val = -1.0
        for r in self.results:
            if r.status == "completed":
                val = r.metrics.get("vqa_accuracy", 0.0)
                if val > best_val:
                    best_val = val
                    best_result = r
        if best_result:
            return {
                "experiment_id": best_result.experiment_id,
                "vqa_accuracy": best_val,
                "checkpoint_path": best_result.checkpoint_path,
            }
        return {}

    def _log_final_summary(self, summary: Dict[str, Any]):
        """Log the final ablation study summary."""
        self.logger.section("ABLATION STUDY SUMMARY")
        self.logger.info("=" * 70)
        self.logger.info(f"  Study: {summary['study_name']}")
        self.logger.info(f"  Total experiments: {summary['total_experiments']}")
        self.logger.info(f"  Completed: {summary['completed']}")
        self.logger.info(f"  Failed: {summary['failed']}")
        self.logger.info(f"  Total time: {summary['total_time_human']}")
        self.logger.info("")

        best = summary.get("best_experiment", {})
        if best:
            self.logger.info(f"  Best experiment: {best.get('experiment_id', 'N/A')}")
            self.logger.info(f"  Best VQA accuracy: {best.get('vqa_accuracy', 0)*100:.2f}%")
        self.logger.info("")

        self.logger.info("  Key Findings:")
        for i, finding in enumerate(summary.get("key_findings", []), 1):
            self.logger.info(f"    {i}. {finding}")
        self.logger.info("")

        self.logger.info("  Report files:")
        for rtype, rpath in summary.get("report_paths", {}).items():
            self.logger.info(f"    [{rtype}] {rpath}")
        self.logger.info("=" * 70)

    @staticmethod
    def _format_time(seconds: float) -> str:
        """Format seconds to human-readable string."""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            return f"{seconds / 60:.1f}m"
        else:
            h = int(seconds // 3600)
            m = int((seconds % 3600) // 60)
            return f"{h}h {m}m"
