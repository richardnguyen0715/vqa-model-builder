"""
Inference Configuration Module
Provides configuration for VQA model inference.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from enum import Enum


class InferenceMode(Enum):
    """Inference mode enumeration."""
    SINGLE = "single"
    BATCH = "batch"
    STREAMING = "streaming"


class DecodingStrategy(Enum):
    """Answer decoding strategy."""
    GREEDY = "greedy"
    TOP_K = "top_k"
    TOP_P = "top_p"
    BEAM_SEARCH = "beam_search"


@dataclass
class InferenceConfig:
    """
    Inference configuration.
    
    Attributes:
        mode: Inference mode (single, batch, streaming)
        batch_size: Batch size for batch inference
        device: Device for inference (auto, cuda, cpu, mps)
        use_fp16: Whether to use FP16 inference
        use_compile: Whether to use torch.compile
        num_workers: Number of data loading workers
        prefetch_factor: Data prefetch factor
    """
    mode: InferenceMode = InferenceMode.SINGLE
    batch_size: int = 32
    device: str = "auto"
    use_fp16: bool = True
    use_compile: bool = False
    num_workers: int = 4
    prefetch_factor: int = 2


@dataclass
class AnswerConfig:
    """
    Answer generation configuration.
    
    Attributes:
        decoding_strategy: Strategy for answer selection
        top_k: K for top-k sampling
        top_p: P for nucleus sampling
        temperature: Sampling temperature
        num_beams: Number of beams for beam search
        return_top_n: Number of top answers to return
        confidence_threshold: Minimum confidence threshold
        use_knowledge: Whether to use knowledge augmentation
        num_retrieved: Number of documents to retrieve
    """
    decoding_strategy: DecodingStrategy = DecodingStrategy.GREEDY
    top_k: int = 5
    top_p: float = 0.9
    temperature: float = 1.0
    num_beams: int = 4
    return_top_n: int = 5
    confidence_threshold: float = 0.0
    use_knowledge: bool = False
    num_retrieved: int = 3


@dataclass
class PreprocessConfig:
    """
    Preprocessing configuration.
    
    Attributes:
        image_size: Target image size
        normalize_mean: Image normalization mean
        normalize_std: Image normalization std
        max_question_length: Maximum question length
        use_vietnamese_processor: Whether to process Vietnamese text
    """
    image_size: int = 224
    normalize_mean: List[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])
    normalize_std: List[float] = field(default_factory=lambda: [0.229, 0.224, 0.225])
    max_question_length: int = 64
    use_vietnamese_processor: bool = True


@dataclass
class VQAInferenceConfig:
    """
    Complete VQA inference configuration.
    
    Attributes:
        inference: Inference settings
        answer: Answer generation settings
        preprocess: Preprocessing settings
        checkpoint_path: Path to model checkpoint
        answer_vocabulary_path: Path to answer vocabulary file
    """
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    answer: AnswerConfig = field(default_factory=AnswerConfig)
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    checkpoint_path: Optional[str] = None
    answer_vocabulary_path: Optional[str] = None


def get_default_inference_config() -> VQAInferenceConfig:
    """
    Get default inference configuration.
    
    Returns:
        Default VQAInferenceConfig instance
    """
    return VQAInferenceConfig(
        inference=InferenceConfig(
            mode=InferenceMode.SINGLE,
            use_fp16=True
        ),
        answer=AnswerConfig(
            decoding_strategy=DecodingStrategy.GREEDY,
            return_top_n=5
        ),
        preprocess=PreprocessConfig(
            image_size=224,
            max_question_length=64
        )
    )
