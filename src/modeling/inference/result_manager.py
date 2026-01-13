"""
Inference Result Manager Module
Handles saving, loading, and formatting of VQA inference results.
"""

import os
import json
import csv
import logging
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import torch
from PIL import Image

from src.middleware.config_loader import get_config

logger = logging.getLogger(__name__)


class InferenceResultManager:
    """
    Manages inference results saving and formatting.
    
    Features:
    - Save results in multiple formats (JSON, CSV, JSONL)
    - Include configurable fields
    - Generate sample visualizations
    - Aggregate results for analysis
    
    Attributes:
        output_dir: Directory for saving results.
        format: Output format (json, csv, jsonl).
        prefix: Filename prefix.
    """
    
    def __init__(
        self,
        output_dir: Optional[str] = None,
        format: Optional[str] = None,
        prefix: Optional[str] = None,
        include_timestamp: Optional[bool] = None,
    ) -> None:
        """
        Initialize inference result manager.
        
        Args:
            output_dir: Directory for results (from config if None).
            format: Output format (from config if None).
            prefix: Filename prefix (from config if None).
            include_timestamp: Include timestamp in filename (from config if None).
        """
        self.output_dir = Path(
            output_dir or get_config("inference", "output.output_dir", "inference_results")
        )
        self.format = format or get_config("inference", "output.format", "json")
        self.prefix = prefix or get_config("inference", "output.prefix", "inference_results")
        self.include_timestamp = (
            include_timestamp if include_timestamp is not None
            else get_config("inference", "output.include_timestamp", True)
        )
        
        # Configuration for what to include
        self.include_predictions = get_config("inference", "output.include_predictions", True)
        self.include_probabilities = get_config("inference", "output.include_probabilities", True)
        self.include_top_k = get_config("inference", "output.include_top_k", True)
        self.include_question = get_config("inference", "output.include_question", True)
        self.include_image_path = get_config("inference", "output.include_image_path", True)
        self.include_ground_truth = get_config("inference", "output.include_ground_truth", False)
        self.include_metadata = get_config("inference", "output.include_metadata", True)
        
        # Sample saving settings
        self.save_samples = get_config("inference", "output.save_samples", True)
        self.num_samples = get_config("inference", "output.num_samples", 10)
        self.sample_dir = get_config("inference", "output.sample_dir", "samples")
        
        # Results storage
        self.results: List[Dict[str, Any]] = []
        self.metadata: Dict[str, Any] = {}
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"InferenceResultManager initialized. Output dir: {self.output_dir}")
    
    def add_result(
        self,
        prediction: str,
        question: Optional[str] = None,
        image_path: Optional[str] = None,
        probabilities: Optional[List[float]] = None,
        top_k_predictions: Optional[List[Dict[str, Any]]] = None,
        ground_truth: Optional[str] = None,
        confidence: Optional[float] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Add single inference result.
        
        Args:
            prediction: Predicted answer.
            question: Input question.
            image_path: Path to input image.
            probabilities: Prediction probabilities.
            top_k_predictions: Top-k predictions with scores.
            ground_truth: Ground truth answer (if available).
            confidence: Prediction confidence score.
            extra: Additional data to include.
        """
        result = {"id": len(self.results)}
        
        if self.include_predictions:
            result["prediction"] = prediction
        
        if self.include_question and question is not None:
            result["question"] = question
        
        if self.include_image_path and image_path is not None:
            result["image_path"] = str(image_path)
        
        if self.include_probabilities and probabilities is not None:
            result["probabilities"] = probabilities
        
        if self.include_top_k and top_k_predictions is not None:
            result["top_k_predictions"] = top_k_predictions
        
        if self.include_ground_truth and ground_truth is not None:
            result["ground_truth"] = ground_truth
        
        if confidence is not None:
            result["confidence"] = confidence
        
        if extra:
            result["extra"] = extra
        
        self.results.append(result)
    
    def add_batch_results(
        self,
        predictions: List[str],
        questions: Optional[List[str]] = None,
        image_paths: Optional[List[str]] = None,
        probabilities: Optional[List[List[float]]] = None,
        top_k_predictions: Optional[List[List[Dict[str, Any]]]] = None,
        ground_truths: Optional[List[str]] = None,
        confidences: Optional[List[float]] = None,
    ) -> None:
        """
        Add batch of inference results.
        
        Args:
            predictions: List of predicted answers.
            questions: List of input questions.
            image_paths: List of image paths.
            probabilities: List of probability distributions.
            top_k_predictions: List of top-k prediction lists.
            ground_truths: List of ground truth answers.
            confidences: List of confidence scores.
        """
        batch_size = len(predictions)
        
        for i in range(batch_size):
            self.add_result(
                prediction=predictions[i],
                question=questions[i] if questions else None,
                image_path=image_paths[i] if image_paths else None,
                probabilities=probabilities[i] if probabilities else None,
                top_k_predictions=top_k_predictions[i] if top_k_predictions else None,
                ground_truth=ground_truths[i] if ground_truths else None,
                confidence=confidences[i] if confidences else None,
            )
    
    def set_metadata(
        self,
        model_name: Optional[str] = None,
        checkpoint_path: Optional[str] = None,
        total_samples: Optional[int] = None,
        inference_time: Optional[float] = None,
        device: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Set inference metadata.
        
        Args:
            model_name: Name of the model.
            checkpoint_path: Path to model checkpoint.
            total_samples: Total number of samples processed.
            inference_time: Total inference time in seconds.
            device: Device used for inference.
            extra: Additional metadata.
        """
        self.metadata = {
            "timestamp": datetime.now().isoformat(),
            "model_name": model_name,
            "checkpoint_path": str(checkpoint_path) if checkpoint_path else None,
            "total_samples": total_samples or len(self.results),
            "inference_time_seconds": inference_time,
            "device": device,
        }
        
        if extra:
            self.metadata.update(extra)
    
    def _generate_filename(self, extension: str) -> str:
        """
        Generate output filename.
        
        Args:
            extension: File extension.
            
        Returns:
            Generated filename.
        """
        parts = [self.prefix]
        
        if self.include_timestamp:
            timestamp = datetime.now().strftime(
                get_config("inference", "output.timestamp_format", "%Y%m%d_%H%M%S")
            )
            parts.append(timestamp)
        
        return "_".join(parts) + f".{extension}"
    
    def save(self, filename: Optional[str] = None) -> Path:
        """
        Save results to file.
        
        Args:
            filename: Optional custom filename.
            
        Returns:
            Path to saved file.
        """
        if self.format == "json":
            return self._save_json(filename)
        elif self.format == "csv":
            return self._save_csv(filename)
        elif self.format == "jsonl":
            return self._save_jsonl(filename)
        else:
            logger.warning(f"Unknown format {self.format}, defaulting to JSON")
            return self._save_json(filename)
    
    def _save_json(self, filename: Optional[str] = None) -> Path:
        """
        Save results as JSON.
        
        Args:
            filename: Optional custom filename.
            
        Returns:
            Path to saved file.
        """
        filename = filename or self._generate_filename("json")
        filepath = self.output_dir / filename
        
        output = {
            "results": self.results,
        }
        
        if self.include_metadata:
            output["metadata"] = self.metadata
        
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Saved {len(self.results)} results to {filepath}")
        return filepath
    
    def _save_csv(self, filename: Optional[str] = None) -> Path:
        """
        Save results as CSV.
        
        Args:
            filename: Optional custom filename.
            
        Returns:
            Path to saved file.
        """
        filename = filename or self._generate_filename("csv")
        filepath = self.output_dir / filename
        
        if not self.results:
            logger.warning("No results to save")
            return filepath
        
        # Get all field names from results
        fieldnames = set()
        for result in self.results:
            fieldnames.update(result.keys())
        fieldnames = sorted(fieldnames)
        
        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for result in self.results:
                # Flatten complex fields for CSV
                row = {}
                for key, value in result.items():
                    if isinstance(value, (list, dict)):
                        row[key] = json.dumps(value, ensure_ascii=False)
                    else:
                        row[key] = value
                writer.writerow(row)
        
        logger.info(f"Saved {len(self.results)} results to {filepath}")
        return filepath
    
    def _save_jsonl(self, filename: Optional[str] = None) -> Path:
        """
        Save results as JSON Lines.
        
        Args:
            filename: Optional custom filename.
            
        Returns:
            Path to saved file.
        """
        filename = filename or self._generate_filename("jsonl")
        filepath = self.output_dir / filename
        
        with open(filepath, "w", encoding="utf-8") as f:
            for result in self.results:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
        
        logger.info(f"Saved {len(self.results)} results to {filepath}")
        return filepath
    
    def save_samples(
        self,
        images: Optional[List[Any]] = None,
        questions: Optional[List[str]] = None,
        predictions: Optional[List[str]] = None,
        ground_truths: Optional[List[str]] = None,
        num_samples: Optional[int] = None,
    ) -> Path:
        """
        Save sample visualizations.
        
        Args:
            images: List of images (PIL Images or paths).
            questions: List of questions.
            predictions: List of predictions.
            ground_truths: List of ground truths.
            num_samples: Number of samples to save.
            
        Returns:
            Path to samples directory.
        """
        if not self.save_samples:
            return self.output_dir
        
        sample_dir = self.output_dir / self.sample_dir
        sample_dir.mkdir(parents=True, exist_ok=True)
        
        num_samples = num_samples or self.num_samples
        num_samples = min(num_samples, len(self.results))
        
        # Create sample info file
        sample_info = []
        
        for i in range(num_samples):
            result = self.results[i]
            
            sample_entry = {
                "sample_id": i,
                "prediction": result.get("prediction"),
                "question": result.get("question"),
                "ground_truth": result.get("ground_truth"),
                "confidence": result.get("confidence"),
            }
            
            # Copy image if available
            if images and i < len(images):
                image = images[i]
                if isinstance(image, str) or isinstance(image, Path):
                    # Copy from path
                    image_path = Path(image)
                    if image_path.exists():
                        dest_path = sample_dir / f"sample_{i}{image_path.suffix}"
                        import shutil
                        shutil.copy(image_path, dest_path)
                        sample_entry["image_file"] = dest_path.name
                elif hasattr(image, "save"):
                    # PIL Image
                    dest_path = sample_dir / f"sample_{i}.png"
                    image.save(dest_path)
                    sample_entry["image_file"] = dest_path.name
            
            sample_info.append(sample_entry)
        
        # Save sample info
        info_path = sample_dir / "sample_info.json"
        with open(info_path, "w", encoding="utf-8") as f:
            json.dump(sample_info, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Saved {len(sample_info)} samples to {sample_dir}")
        return sample_dir
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary of inference results.
        
        Returns:
            Summary dictionary.
        """
        summary = {
            "total_samples": len(self.results),
            "output_dir": str(self.output_dir),
            "format": self.format,
        }
        
        # Calculate accuracy if ground truth available
        if self.include_ground_truth:
            correct = sum(
                1 for r in self.results
                if r.get("prediction") == r.get("ground_truth")
            )
            if len(self.results) > 0:
                summary["accuracy"] = correct / len(self.results)
        
        # Average confidence
        confidences = [r.get("confidence") for r in self.results if r.get("confidence")]
        if confidences:
            summary["average_confidence"] = sum(confidences) / len(confidences)
        
        return summary
    
    def clear(self) -> None:
        """Clear all stored results."""
        self.results.clear()
        self.metadata.clear()
        logger.debug("Cleared all stored results")
    
    @classmethod
    def load_results(cls, filepath: Union[str, Path]) -> Dict[str, Any]:
        """
        Load results from file.
        
        Args:
            filepath: Path to results file.
            
        Returns:
            Dictionary with results and metadata.
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Results file not found: {filepath}")
        
        if filepath.suffix == ".json":
            with open(filepath, "r", encoding="utf-8") as f:
                return json.load(f)
        elif filepath.suffix == ".jsonl":
            results = []
            with open(filepath, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        results.append(json.loads(line))
            return {"results": results}
        elif filepath.suffix == ".csv":
            results = []
            with open(filepath, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    results.append(row)
            return {"results": results}
        else:
            raise ValueError(f"Unknown file format: {filepath.suffix}")


# =============================================================================
# Convenience Functions
# =============================================================================

def save_inference_results(
    predictions: List[str],
    output_path: Union[str, Path],
    questions: Optional[List[str]] = None,
    image_paths: Optional[List[str]] = None,
    ground_truths: Optional[List[str]] = None,
    confidences: Optional[List[float]] = None,
    format: str = "json",
) -> Path:
    """
    Simple function to save inference results.
    
    Args:
        predictions: List of predicted answers.
        output_path: Output file path.
        questions: List of questions.
        image_paths: List of image paths.
        ground_truths: List of ground truths.
        confidences: List of confidence scores.
        format: Output format.
        
    Returns:
        Path to saved file.
    """
    output_path = Path(output_path)
    
    results = []
    for i, pred in enumerate(predictions):
        result = {
            "id": i,
            "prediction": pred,
        }
        if questions:
            result["question"] = questions[i]
        if image_paths:
            result["image_path"] = image_paths[i]
        if ground_truths:
            result["ground_truth"] = ground_truths[i]
        if confidences:
            result["confidence"] = confidences[i]
        results.append(result)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if format == "json":
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump({"results": results}, f, ensure_ascii=False, indent=2)
    elif format == "csv":
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            if results:
                writer = csv.DictWriter(f, fieldnames=results[0].keys())
                writer.writeheader()
                writer.writerows(results)
    elif format == "jsonl":
        with open(output_path, "w", encoding="utf-8") as f:
            for result in results:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
    
    logger.info(f"Saved {len(results)} results to {output_path}")
    return output_path


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    "InferenceResultManager",
    "save_inference_results",
]
