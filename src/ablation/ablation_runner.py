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


# ── Graceful interrupt exception ──────────────────────────────────

class GracefulInterrupt(Exception):
    """Raised when the ablation loop is interrupted by Ctrl+C.

    Carries context about what was running so the caller can
    report progress cleanly.
    """

    def __init__(
        self,
        experiment_id: str = "",
        completed_count: int = 0,
        total: int = 0,
    ):
        self.experiment_id = experiment_id
        self.completed_count = completed_count
        self.total = total
        super().__init__(
            f"Interrupted at {experiment_id} "
            f"({completed_count}/{total} completed)"
        )

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
        tokenizer: Optional[Any] = None,
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
        self.tokenizer = tokenizer  # Required for generative model

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

    def run(
        self,
        selected_indices: Optional[List[int]] = None,
        rerun_indices: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        """Execute the complete ablation study.

        Args:
            selected_indices: Optional 1-based list of experiment numbers to run.
                If None, all experiments are run. Example: [1, 3, 5] runs only
                the 1st, 3rd, and 5th experiments from the generated matrix.
            rerun_indices: Optional 1-based list of experiment numbers to force
                re-run from scratch. Old results for these experiments are deleted
                before running. Can overlap with selected_indices.

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

            # 3. Delete results for forced re-run experiments
            if rerun_indices:
                self._delete_rerun_results(rerun_indices)

            # 4. Load any previously completed results (resume)
            if self.config.resume:
                self._load_completed_results()

            # 5. Run experiments (with incremental reporting)
            self._run_all_experiments(
                selected_indices=selected_indices,
                rerun_indices=rerun_indices,
            )

            # 6. Final evaluate & analyze & report
            summary = self._generate_full_report(start_time)

            self._log_final_summary(summary)
            self.logger.end_stage("ABLATION STUDY", success=True)

            return summary

        except GracefulInterrupt as gi:
            # ── Ctrl+C: generate report with completed results and exit ──
            self.logger.info("Generating final report after interruption...")
            try:
                summary = self._generate_full_report(start_time)
                summary["interrupted"] = True
                summary["interrupted_at"] = gi.experiment_id
            except Exception:
                summary = {
                    "study_name": self.config.name,
                    "interrupted": True,
                    "interrupted_at": gi.experiment_id,
                    "completed": gi.completed_count,
                    "total_experiments": gi.total,
                }

            self.logger.end_stage("ABLATION STUDY", success=False)
            return summary

        except KeyboardInterrupt:
            # ── Ctrl+C outside of an experiment (between experiments) ──
            completed_count = sum(
                1 for r in self.results if r.status == "completed"
            )
            total = len(self.experiments) if self.experiments else 0
            self.logger.warning(
                f"\n{'='*70}\n"
                f"  INTERRUPTED BY USER (Ctrl+C) — between experiments\n"
                f"  Completed experiments so far: {completed_count}/{total}\n"
                f"{'='*70}"
            )
            try:
                summary = self._generate_full_report(start_time)
                summary["interrupted"] = True
            except Exception:
                summary = {
                    "study_name": self.config.name,
                    "interrupted": True,
                    "completed": completed_count,
                    "total_experiments": total,
                }

            self.logger.end_stage("ABLATION STUDY", success=False)
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

    def _delete_rerun_results(self, rerun_indices: List[int]):
        """Delete previous results for experiments that should be re-run.

        Args:
            rerun_indices: 1-based indices of experiments to force re-run.
        """
        deleted = 0
        for idx in rerun_indices:
            if 1 <= idx <= len(self.experiments):
                exp_id = self.experiments[idx - 1].experiment_id
                safe_id = exp_id.replace("/", "_").replace("\\", "_")
                result_path = self.results_dir / f"{safe_id}.json"
                if result_path.exists():
                    result_path.unlink()
                    self.logger.info(
                        f"Deleted old result for re-run: [{idx}] {exp_id}"
                    )
                    deleted += 1
                # Also remove from completed_ids if already loaded
                self.completed_ids.discard(exp_id)
        if deleted:
            self.logger.info(f"Cleared {deleted} old result(s) for re-run")

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

    def _run_all_experiments(
        self,
        selected_indices: Optional[List[int]] = None,
        rerun_indices: Optional[List[int]] = None,
    ):
        """Run experiments in the matrix.

        Args:
            selected_indices: Optional 1-based indices to run. If None, run all.
            rerun_indices: Optional 1-based indices to force re-run (skip resume).
        """
        # Convert selected_indices to a set of 0-based indices for fast lookup
        if selected_indices is not None:
            selected_set = {i - 1 for i in selected_indices}  # 1-based → 0-based
            self.logger.info(
                f"Running selected experiments: "
                f"{sorted(selected_indices)} ({len(selected_indices)}/{len(self.experiments)})"
            )
        else:
            selected_set = None  # means run all

        # Convert rerun_indices to set of experiment IDs for fast lookup
        rerun_ids: set = set()
        if rerun_indices:
            for idx in rerun_indices:
                if 1 <= idx <= len(self.experiments):
                    rerun_ids.add(self.experiments[idx - 1].experiment_id)

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
            tokenizer=self.tokenizer,
        )

        total = len(self.experiments)
        completed_count = len(self.completed_ids)

        for idx, experiment in enumerate(self.experiments, 1):
            # Skip if not in user selection
            if selected_set is not None and (idx - 1) not in selected_set:
                continue

            # Skip completed (resume) — but NOT if this experiment is in the rerun set
            if (
                experiment.experiment_id in self.completed_ids
                and experiment.experiment_id not in rerun_ids
            ):
                self.logger.info(
                    f"[{idx}/{total}] Skipping (already completed): "
                    f"{experiment.experiment_id}"
                )
                continue

            if experiment.experiment_id in rerun_ids:
                self.logger.info(
                    f"[{idx}/{total}] Force RE-RUN: {experiment.experiment_id}"
                )

            self.logger.section(
                f"Experiment [{idx}/{total}]: {experiment.experiment_id}"
            )
            self.logger.info(f"Expert: {experiment.expert_config.description}")
            self.logger.info(f"Router: {experiment.router_config.description}")

            # Run (may raise KeyboardInterrupt)
            try:
                result = trainer.run_experiment(experiment)
            except KeyboardInterrupt:
                # The trainer already saved an emergency checkpoint.
                # Save the interrupted result and break out of the loop.
                result = trainer._last_interrupted_result
                if result is None:
                    result = ExperimentResult(
                        experiment_id=experiment.experiment_id,
                        experiment_config=experiment.to_dict(),
                        status="interrupted",
                        error_message="Interrupted by user (Ctrl+C)",
                    )

                self.results.append(result)
                self._save_experiment_result(result)

                completed_count = sum(
                    1 for r in self.results if r.status == "completed"
                )
                self._save_progress(idx, total, completed_count)

                self.logger.warning(
                    f"\n{'='*70}\n"
                    f"  INTERRUPTED BY USER (Ctrl+C)\n"
                    f"  Experiment: {experiment.experiment_id}\n"
                    f"  Completed experiments so far: {completed_count}/{total}\n"
                    f"{'='*70}"
                )

                # Generate report with whatever is completed
                if completed_count > 0:
                    self.logger.info("Generating report with completed results...")
                    try:
                        self._run_incremental_report(idx, total)
                    except Exception:
                        pass

                # Signal the outer run() to shut down gracefully
                raise GracefulInterrupt(
                    experiment_id=experiment.experiment_id,
                    completed_count=completed_count,
                    total=total,
                )

            # If re-running, replace old result in self.results
            if experiment.experiment_id in rerun_ids:
                self.results = [
                    r for r in self.results
                    if r.experiment_id != experiment.experiment_id
                ]

            self.results.append(result)

            # Save result to disk (for resume)
            self._save_experiment_result(result)

            # Update progress
            if result.status == "completed":
                self.completed_ids.add(experiment.experiment_id)
            completed_count = sum(
                1 for r in self.results if r.status == "completed"
            )
            self._save_progress(idx, total, completed_count)

            # ── Incremental report after each completed experiment ──
            if result.status == "completed":
                self._run_incremental_report(idx, total)

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
        primary_metric = self.config.primary_metric
        evaluator = AblationEvaluator(
            primary_metric=primary_metric,
            model_type=self.config.model_type,
        )
        for result in self.results:
            if result.status == "completed":
                evaluator.add_result(result.to_dict())
        return evaluator

    def _run_incremental_report(self, current_idx: int, total: int):
        """Generate an incremental report after each completed experiment.

        Includes all completed results so far (both resumed and newly run).
        Reports are saved with an 'incremental_' prefix and also overwrite
        the main report files so they always reflect the latest state.
        """
        completed_results = [r for r in self.results if r.status == "completed"]
        if not completed_results:
            return

        try:
            evaluator = self._build_evaluator()
            analyzer = AblationAnalyzer(
                evaluator, primary_metric=self.config.primary_metric
            )
            reporter = AblationReporter(
                analyzer=analyzer,
                output_dir=str(self.output_dir),
                study_name=self.config.name,
            )
            report_paths = reporter.save_all_reports()

            # Also save a timestamped incremental summary
            best = self._get_best_experiment()
            primary = self.config.primary_metric
            inc_summary = {
                "type": "incremental",
                "after_experiment": current_idx,
                "total_experiments": total,
                "completed_so_far": len(completed_results),
                "failed_so_far": sum(1 for r in self.results if r.status == "failed"),
                "best_experiment": best,
                "completed_ids": sorted(self.completed_ids),
                "timestamp": datetime.now().isoformat(),
                "report_paths": report_paths,
            }

            inc_path = self.output_dir / f"{self.config.name}_summary.json"
            with open(inc_path, "w") as f:
                json.dump(inc_summary, f, indent=2, default=str)

            self.logger.info(
                f"  Incremental report updated: "
                f"{len(completed_results)} completed experiments | "
                f"Best {primary}: "
                f"{best.get(primary, 0):.4f} ({best.get('experiment_id', 'N/A')})"
            )

        except Exception as e:
            # Incremental report failure should not crash the pipeline
            self.logger.warning(f"Incremental report failed (non-fatal): {e}")

    def _generate_full_report(self, start_time: float) -> Dict[str, Any]:
        """Generate final full report with all results.

        Args:
            start_time: Pipeline start timestamp for total time calculation.

        Returns:
            Summary dict with key findings, report paths, etc.
        """
        evaluator = self._build_evaluator()
        analyzer = AblationAnalyzer(
            evaluator, primary_metric=self.config.primary_metric
        )

        reporter = AblationReporter(
            analyzer=analyzer,
            output_dir=str(self.output_dir),
            study_name=self.config.name,
        )
        report_paths = reporter.save_all_reports()

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

        summary_path = self.output_dir / f"{self.config.name}_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)

        return summary

    def _get_best_experiment(self) -> Dict[str, Any]:
        """Find the best experiment by primary metric."""
        primary_metric = self.config.primary_metric
        best_result = None
        best_val = -1.0
        for r in self.results:
            if r.status == "completed":
                val = r.metrics.get(primary_metric, 0.0)
                if val > best_val:
                    best_val = val
                    best_result = r
        if best_result:
            return {
                "experiment_id": best_result.experiment_id,
                primary_metric: best_val,
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
            primary = self.config.primary_metric
            self.logger.info(f"  Best experiment: {best.get('experiment_id', 'N/A')}")
            self.logger.info(f"  Best {primary}: {best.get(primary, 0):.4f}")
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
