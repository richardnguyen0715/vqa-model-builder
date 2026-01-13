"""
VQA Predictor Module
Provides inference engine for Vietnamese VQA models.
Includes resource monitoring for inference operations.
"""

import logging
from typing import Optional, Dict, Any, List, Union, Tuple
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from .inference_config import (
    VQAInferenceConfig,
    InferenceMode,
    DecodingStrategy,
    get_default_inference_config
)

# Resource management imports
try:
    from src.resource_management import (
        ResourceMonitor,
        load_resource_config,
    )
    RESOURCE_MANAGEMENT_AVAILABLE = True
except ImportError:
    RESOURCE_MANAGEMENT_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class PredictionResult:
    """
    VQA prediction result.
    
    Attributes:
        answer: Predicted answer string
        answer_id: Predicted answer index
        confidence: Prediction confidence score
        top_answers: List of top-n answers with scores
        question: Original question
        attention_weights: Optional attention visualization data
        retrieved_knowledge: Optional retrieved knowledge documents
        resource_metrics: Optional resource usage during inference
    """
    answer: str
    answer_id: int
    confidence: float
    top_answers: List[Tuple[str, float]] = field(default_factory=list)
    question: str = ""
    attention_weights: Optional[Dict[str, Any]] = None
    retrieved_knowledge: Optional[List[str]] = None
    resource_metrics: Optional[Dict[str, Any]] = None


@dataclass
class BatchPredictionResult:
    """
    Batch VQA prediction results.
    
    Attributes:
        predictions: List of individual prediction results
        batch_size: Number of predictions
        inference_time: Total inference time in seconds
        resource_metrics: Optional resource usage during batch inference
    """
    predictions: List[PredictionResult]
    batch_size: int
    inference_time: float = 0.0
    resource_metrics: Optional[Dict[str, Any]] = None


class VQAPredictor:
    """
    Vietnamese VQA inference engine.
    
    Provides:
    - Single and batch prediction
    - Multiple decoding strategies
    - Knowledge-augmented inference
    - Attention visualization
    - Resource monitoring during inference
    
    Attributes:
        model: VQA model
        config: Inference configuration
        device: Inference device
        answer_vocabulary: Answer ID to string mapping
        tokenizer: Text tokenizer
        image_transform: Image transformation pipeline
        resource_monitor: Optional resource monitor for tracking usage
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: Optional[VQAInferenceConfig] = None,
        answer_vocabulary: Optional[List[str]] = None,
        tokenizer: Any = None,
        image_transform: Any = None,
        enable_resource_monitoring: bool = False
    ):
        """
        Initialize VQA predictor.
        
        Args:
            model: Trained VQA model
            config: Inference configuration
            answer_vocabulary: List of answer strings
            tokenizer: Text tokenizer for questions
            image_transform: Image preprocessing transform
            enable_resource_monitoring: Enable resource monitoring during inference
        """
        self.config = config or get_default_inference_config()
        self.device = self._get_device()
        
        # Setup model
        self.model = model.to(self.device)
        self.model.eval()
        
        # Compile model if enabled
        if self.config.inference.use_compile and hasattr(torch, "compile"):
            self.model = torch.compile(self.model)
        
        # Answer vocabulary
        self.answer_vocabulary = answer_vocabulary or []
        if self.config.answer_vocabulary_path:
            self._load_answer_vocabulary()
        
        # Tokenizer
        self.tokenizer = tokenizer
        if tokenizer is None:
            self._setup_default_tokenizer()
        
        # Image transform
        self.image_transform = image_transform
        if image_transform is None:
            self._setup_default_transform()
        
        # Setup resource monitoring
        self.resource_monitor = self._setup_resource_monitor(enable_resource_monitoring)
        
        logger.info(f"Initialized VQA predictor on {self.device}")
    
    def _setup_resource_monitor(
        self,
        enable: bool = False
    ) -> Optional["ResourceMonitor"]:
        """
        Setup resource monitor for tracking usage during inference.
        
        Args:
            enable: Whether to enable resource monitoring.
            
        Returns:
            ResourceMonitor instance or None if not available.
        """
        if not enable or not RESOURCE_MANAGEMENT_AVAILABLE:
            return None
        
        try:
            resource_config = load_resource_config()
            monitor = ResourceMonitor(resource_config)
            logger.info("Resource monitoring enabled for inference")
            return monitor
        except Exception as e:
            logger.warning(f"Failed to setup resource monitor: {e}")
            return None
    
    def _get_resource_snapshot(self) -> Optional[Dict[str, Any]]:
        """
        Get current resource usage snapshot.
        
        Returns:
            Dictionary with resource metrics or None if monitoring not available.
        """
        if self.resource_monitor is None:
            return None
        
        try:
            metrics = self.resource_monitor.get_latest_metrics()
            if metrics is not None:
                return metrics.to_dict()
        except Exception:
            pass
        return None
    
    def _get_device(self) -> torch.device:
        """Get inference device."""
        device_str = self.config.inference.device
        
        if device_str == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        
        return torch.device(device_str)
    
    def _load_answer_vocabulary(self) -> None:
        """Load answer vocabulary from file."""
        path = self.config.answer_vocabulary_path
        if path and Path(path).exists():
            with open(path, "r", encoding="utf-8") as f:
                self.answer_vocabulary = [line.strip() for line in f]
            logger.info(f"Loaded {len(self.answer_vocabulary)} answers from {path}")
    
    def _setup_default_tokenizer(self) -> None:
        """Setup default Vietnamese tokenizer."""
        try:
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
            logger.info("Using PhoBERT tokenizer")
        except Exception as e:
            logger.warning(f"Could not load PhoBERT tokenizer: {e}")
            self.tokenizer = None
    
    def _setup_default_transform(self) -> None:
        """Setup default image transform."""
        try:
            from torchvision import transforms
            
            preprocess = self.config.preprocess
            self.image_transform = transforms.Compose([
                transforms.Resize((preprocess.image_size, preprocess.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=preprocess.normalize_mean,
                    std=preprocess.normalize_std
                )
            ])
        except ImportError:
            logger.warning("torchvision not available, image transform not set")
            self.image_transform = None
    
    def _preprocess_image(self, image: Union[str, Path, Image.Image, torch.Tensor]) -> torch.Tensor:
        """
        Preprocess image for inference.
        
        Args:
            image: Image path, PIL Image, or tensor
            
        Returns:
            Preprocessed image tensor
        """
        if isinstance(image, torch.Tensor):
            if image.dim() == 3:
                image = image.unsqueeze(0)
            return image.to(self.device)
        
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")
        
        if self.image_transform:
            image = self.image_transform(image)
        
        if image.dim() == 3:
            image = image.unsqueeze(0)
        
        return image.to(self.device)
    
    def _preprocess_question(self, question: str) -> Dict[str, torch.Tensor]:
        """
        Preprocess question for inference.
        
        Args:
            question: Question string
            
        Returns:
            Dictionary with input_ids and attention_mask
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer not initialized")
        
        # Vietnamese text preprocessing
        if self.config.preprocess.use_vietnamese_processor:
            try:
                from underthesea import word_tokenize
                question = word_tokenize(question, format="text")
            except ImportError:
                pass
        
        encoded = self.tokenizer(
            question,
            max_length=self.config.preprocess.max_question_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            "input_ids": encoded["input_ids"].to(self.device),
            "attention_mask": encoded["attention_mask"].to(self.device)
        }
    
    def _decode_answer(
        self,
        logits: torch.Tensor,
        return_attention: bool = False
    ) -> Tuple[int, float, List[Tuple[str, float]]]:
        """
        Decode answer from logits.
        
        Args:
            logits: Model output logits
            return_attention: Whether to include attention
            
        Returns:
            Tuple of (answer_id, confidence, top_answers)
        """
        answer_config = self.config.answer
        
        # Apply temperature
        if answer_config.temperature != 1.0:
            logits = logits / answer_config.temperature
        
        # Get probabilities
        probs = F.softmax(logits, dim=-1)
        
        # Decode based on strategy
        strategy = answer_config.decoding_strategy
        
        if strategy == DecodingStrategy.GREEDY:
            answer_id = probs.argmax(dim=-1).item()
            confidence = probs[0, answer_id].item()
        
        elif strategy == DecodingStrategy.TOP_K:
            top_k_probs, top_k_ids = probs.topk(answer_config.top_k, dim=-1)
            # Sample from top-k
            sample_probs = top_k_probs / top_k_probs.sum()
            sample_idx = torch.multinomial(sample_probs[0], 1).item()
            answer_id = top_k_ids[0, sample_idx].item()
            confidence = top_k_probs[0, sample_idx].item()
        
        elif strategy == DecodingStrategy.TOP_P:
            sorted_probs, sorted_ids = probs.sort(dim=-1, descending=True)
            cumsum_probs = sorted_probs.cumsum(dim=-1)
            mask = cumsum_probs <= answer_config.top_p
            mask[..., 1:] = mask[..., :-1].clone()
            mask[..., 0] = True
            filtered_probs = sorted_probs * mask.float()
            filtered_probs = filtered_probs / filtered_probs.sum()
            sample_idx = torch.multinomial(filtered_probs[0], 1).item()
            answer_id = sorted_ids[0, sample_idx].item()
            confidence = sorted_probs[0, sample_idx].item()
        
        else:  # BEAM_SEARCH - simplified as greedy for classification
            answer_id = probs.argmax(dim=-1).item()
            confidence = probs[0, answer_id].item()
        
        # Get top-n answers
        top_n = answer_config.return_top_n
        top_probs, top_ids = probs.topk(top_n, dim=-1)
        
        top_answers = []
        for i in range(top_n):
            aid = top_ids[0, i].item()
            prob = top_probs[0, i].item()
            answer_str = self._id_to_answer(aid)
            top_answers.append((answer_str, prob))
        
        return answer_id, confidence, top_answers
    
    def _id_to_answer(self, answer_id: int) -> str:
        """Convert answer ID to string."""
        if 0 <= answer_id < len(self.answer_vocabulary):
            return self.answer_vocabulary[answer_id]
        return f"answer_{answer_id}"
    
    @torch.no_grad()
    def predict(
        self,
        image: Union[str, Path, Image.Image, torch.Tensor],
        question: str,
        return_attention: bool = False,
        use_knowledge: Optional[bool] = None
    ) -> PredictionResult:
        """
        Predict answer for single image-question pair.
        
        Args:
            image: Input image
            question: Question string
            return_attention: Whether to return attention weights
            use_knowledge: Whether to use knowledge augmentation
            
        Returns:
            PredictionResult with answer and metadata
        """
        # Preprocess inputs
        image_tensor = self._preprocess_image(image)
        question_inputs = self._preprocess_question(question)
        
        # Use FP16 if enabled
        if self.config.inference.use_fp16 and self.device.type == "cuda":
            with torch.cuda.amp.autocast():
                outputs = self.model(
                    image=image_tensor,
                    input_ids=question_inputs["input_ids"],
                    attention_mask=question_inputs["attention_mask"]
                )
        else:
            outputs = self.model(
                image=image_tensor,
                input_ids=question_inputs["input_ids"],
                attention_mask=question_inputs["attention_mask"]
            )
        
        # Get logits
        logits = outputs.logits if hasattr(outputs, "logits") else outputs["logits"]
        
        # Decode answer
        answer_id, confidence, top_answers = self._decode_answer(
            logits, return_attention
        )
        
        # Get attention weights if requested
        attention_weights = None
        if return_attention and hasattr(outputs, "attention_weights"):
            attention_weights = {
                k: v.cpu().numpy() if isinstance(v, torch.Tensor) else v
                for k, v in outputs.attention_weights.items()
            }
        
        # Get retrieved knowledge if available
        retrieved_knowledge = None
        if hasattr(outputs, "retrieved_docs"):
            retrieved_knowledge = outputs.retrieved_docs
        
        return PredictionResult(
            answer=self._id_to_answer(answer_id),
            answer_id=answer_id,
            confidence=confidence,
            top_answers=top_answers,
            question=question,
            attention_weights=attention_weights,
            retrieved_knowledge=retrieved_knowledge
        )
    
    @torch.no_grad()
    def predict_batch(
        self,
        images: List[Union[str, Path, Image.Image, torch.Tensor]],
        questions: List[str]
    ) -> BatchPredictionResult:
        """
        Predict answers for batch of image-question pairs.
        
        Tracks resource usage during batch inference if monitoring is enabled.
        
        Args:
            images: List of input images
            questions: List of question strings
            
        Returns:
            BatchPredictionResult with all predictions and resource metrics
        """
        import time
        start_time = time.time()
        
        assert len(images) == len(questions), "Images and questions must have same length"
        
        # Start resource monitoring for batch inference
        if self.resource_monitor is not None:
            self.resource_monitor.start()
            initial_resources = self._get_resource_snapshot()
        else:
            initial_resources = None
        
        predictions = []
        batch_size = self.config.inference.batch_size
        
        try:
            # Process in batches
            for i in range(0, len(images), batch_size):
                batch_images = images[i:i + batch_size]
                batch_questions = questions[i:i + batch_size]
                
                # Preprocess batch
                image_tensors = torch.cat([
                    self._preprocess_image(img) for img in batch_images
                ], dim=0)
                
                # Tokenize questions
                question_inputs = self.tokenizer(
                    batch_questions,
                    max_length=self.config.preprocess.max_question_length,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt"
                )
                question_inputs = {
                    k: v.to(self.device) for k, v in question_inputs.items()
                }
                
                # Inference
                if self.config.inference.use_fp16 and self.device.type == "cuda":
                    with torch.cuda.amp.autocast():
                        outputs = self.model(
                            image=image_tensors,
                            input_ids=question_inputs["input_ids"],
                            attention_mask=question_inputs["attention_mask"]
                        )
                else:
                    outputs = self.model(
                        image=image_tensors,
                        input_ids=question_inputs["input_ids"],
                        attention_mask=question_inputs["attention_mask"]
                    )
                
                logits = outputs.logits if hasattr(outputs, "logits") else outputs["logits"]
                probs = F.softmax(logits, dim=-1)
                
                # Process each prediction
                for j, (q, p) in enumerate(zip(batch_questions, probs)):
                    answer_id = p.argmax().item()
                    confidence = p[answer_id].item()
                    
                    top_n = self.config.answer.return_top_n
                    top_probs, top_ids = p.topk(top_n)
                    top_answers = [
                        (self._id_to_answer(tid.item()), tp.item())
                        for tid, tp in zip(top_ids, top_probs)
                    ]
                    
                    predictions.append(PredictionResult(
                        answer=self._id_to_answer(answer_id),
                        answer_id=answer_id,
                        confidence=confidence,
                        top_answers=top_answers,
                        question=q
                    ))
        finally:
            # Stop resource monitoring
            if self.resource_monitor is not None:
                self.resource_monitor.stop()
        
        inference_time = time.time() - start_time
        
        # Collect resource metrics
        final_resources = self._get_resource_snapshot()
        resource_metrics = None
        if initial_resources or final_resources:
            resource_metrics = {
                "initial": initial_resources,
                "final": final_resources,
                "inference_time": inference_time,
            }
        
        return BatchPredictionResult(
            predictions=predictions,
            batch_size=len(images),
            inference_time=inference_time,
            resource_metrics=resource_metrics
        )
    
    def save_predictions(
        self,
        results: Union[PredictionResult, BatchPredictionResult],
        output_path: str
    ) -> None:
        """
        Save predictions to file.
        
        Includes resource metrics if available.
        
        Args:
            results: Prediction results
            output_path: Output file path
        """
        import json
        
        if isinstance(results, PredictionResult):
            data = {
                "answer": results.answer,
                "answer_id": results.answer_id,
                "confidence": results.confidence,
                "top_answers": results.top_answers,
                "question": results.question
            }
            # Add resource metrics if available
            if results.resource_metrics:
                data["resource_metrics"] = results.resource_metrics
        else:
            data = {
                "predictions": [
                    {
                        "answer": p.answer,
                        "answer_id": p.answer_id,
                        "confidence": p.confidence,
                        "top_answers": p.top_answers,
                        "question": p.question
                    }
                    for p in results.predictions
                ],
                "batch_size": results.batch_size,
                "inference_time": results.inference_time
            }
            # Add resource metrics if available
            if results.resource_metrics:
                data["resource_metrics"] = results.resource_metrics
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Saved predictions to {output_path}")


def load_predictor(
    checkpoint_path: str,
    model_config: Optional[Any] = None,
    inference_config: Optional[VQAInferenceConfig] = None,
    answer_vocabulary: Optional[List[str]] = None
) -> VQAPredictor:
    """
    Load VQA predictor from checkpoint.
    
    Args:
        checkpoint_path: Path to model checkpoint
        model_config: Model configuration
        inference_config: Inference configuration
        answer_vocabulary: Answer vocabulary
        
    Returns:
        Initialized VQAPredictor
    """
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    
    # Create model
    if model_config is None:
        from ..meta_arch import create_vqa_model, get_default_vietnamese_vqa_config
        model_config = get_default_vietnamese_vqa_config()
    
    from ..meta_arch import create_vqa_model
    model = create_vqa_model(model_config)
    
    # Load weights
    model.load_state_dict(checkpoint["model_state_dict"])
    
    # Create predictor
    config = inference_config or get_default_inference_config()
    config.checkpoint_path = checkpoint_path
    
    return VQAPredictor(
        model=model,
        config=config,
        answer_vocabulary=answer_vocabulary
    )
