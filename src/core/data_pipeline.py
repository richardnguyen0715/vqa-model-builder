"""
Data Pipeline Module
Handles complete data loading, validation, and preprocessing with comprehensive logging.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
import numpy as np
from collections import Counter

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from src.core.pipeline_logger import get_pipeline_logger, PipelineLogger
from src.schema.data_schema import OneSample
from src.data.dataset import vqa_collate_fn


@dataclass
class DataPipelineConfig:
    """Configuration for data pipeline."""
    
    # Data paths
    images_dir: str = "data/raw/images"
    text_file: str = "data/raw/texts/evaluate_60k_data_balanced_preprocessed.csv"
    
    # Split ratios
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    
    # DataLoader settings
    batch_size: int = 32
    eval_batch_size: int = 64
    num_workers: int = 4
    pin_memory: bool = True
    
    # Image settings
    image_size: Tuple[int, int] = (224, 224)
    normalize_mean: List[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])
    normalize_std: List[float] = field(default_factory=lambda: [0.229, 0.224, 0.225])
    
    # Augmentation
    augmentation_strength: str = "medium"  # light, medium, strong
    
    # Tokenizer settings
    tokenizer_name: str = "vinai/phobert-base"
    max_seq_length: int = 64
    
    # Answer vocabulary
    min_answer_freq: int = 5
    
    # Validation
    validate_samples: int = 5  # Number of samples to log for validation
    
    # Reproducibility
    seed: int = 42
    

@dataclass
class DataPipelineOutput:
    """Output from data pipeline."""
    
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: Optional[DataLoader]
    
    train_samples: int
    val_samples: int
    test_samples: int
    
    answer2id: Dict[str, int]
    id2answer: Dict[int, str]
    num_classes: int
    
    tokenizer: Any
    train_transform: transforms.Compose
    eval_transform: transforms.Compose


class DataPipeline:
    """
    Complete data pipeline for VQA.
    
    Handles:
    - Data loading from images + CSV
    - Data validation and statistics
    - Train/val/test splitting
    - Answer vocabulary building
    - Tokenizer initialization
    - Augmentation and transforms
    - DataLoader creation
    
    All steps are logged with comprehensive validation output.
    """
    
    def __init__(
        self,
        config: Optional[DataPipelineConfig] = None,
        logger: Optional[PipelineLogger] = None
    ):
        """
        Initialize data pipeline.
        
        Args:
            config: Data pipeline configuration
            logger: Pipeline logger instance
        """
        self.config = config or DataPipelineConfig()
        self.logger = logger or get_pipeline_logger()
        
        # State
        self.raw_data: List[OneSample] = []
        self.train_data: List[OneSample] = []
        self.val_data: List[OneSample] = []
        self.test_data: List[OneSample] = []
        
        self.answer2id: Dict[str, int] = {}
        self.id2answer: Dict[int, str] = {}
        
        self.tokenizer = None
        self.train_transform = None
        self.eval_transform = None
        
    def run(self) -> DataPipelineOutput:
        """
        Execute the complete data pipeline.
        
        Returns:
            DataPipelineOutput with all components
        """
        self.logger.start_stage("DATA PIPELINE")
        
        try:
            # Step 1: Load raw data
            self._load_raw_data()
            
            # Step 2: Validate data
            self._validate_data()
            
            # Step 3: Compute data statistics
            self._compute_statistics()
            
            # Step 4: Split data
            self._split_data()
            
            # Step 5: Build answer vocabulary
            self._build_answer_vocabulary()
            
            # Step 6: Initialize tokenizer
            self._initialize_tokenizer()
            
            # Step 7: Build transforms
            self._build_transforms()
            
            # Step 8: Create datasets and dataloaders
            output = self._create_dataloaders()
            
            # Step 9: Validate dataloaders
            self._validate_dataloaders(output)
            
            self.logger.end_stage("DATA PIPELINE", success=True)
            
            return output
            
        except Exception as e:
            self.logger.error(f"Data pipeline failed: {e}")
            self.logger.end_stage("DATA PIPELINE", success=False)
            raise
            
    def _load_raw_data(self):
        """Load raw data from images and text file."""
        self.logger.subsection("Loading Raw Data")
        
        # Import data loading functions
        from src.data.data_actions import load_raw_data
        
        images_dir = self.config.images_dir
        text_file = self.config.text_file
        
        self.logger.info(f"Images directory: {images_dir}")
        self.logger.info(f"Text file: {text_file}")
        
        # Check paths exist
        if not os.path.exists(images_dir):
            raise FileNotFoundError(f"Images directory not found: {images_dir}")
        if not os.path.exists(text_file):
            raise FileNotFoundError(f"Text file not found: {text_file}")
            
        # Count images
        image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
        image_count = sum(1 for f in os.listdir(images_dir) 
                        if f.lower().endswith(image_extensions))
        self.logger.info(f"Found {image_count} images")
        
        # Load data
        self.logger.info("Loading data samples...")
        self.raw_data = load_raw_data(images_dir, text_file)
        
        self.logger.success(f"Loaded {len(self.raw_data)} data samples")
        
        # Log sample examples
        self.logger.subsection("Sample Data Validation")
        for i in range(min(self.config.validate_samples, len(self.raw_data))):
            self.logger.log_data_sample(self.raw_data[i], idx=i)
            
    def _validate_data(self):
        """Validate loaded data."""
        self.logger.subsection("Data Validation")
        
        # Import validation function
        from src.data.data_actions import validate_data
        
        self.logger.info("Running data validation checks...")
        
        issues = []
        valid_count = 0
        
        for i, sample in enumerate(self.raw_data):
            try:
                # Check image path exists
                if not hasattr(sample, 'image_path') or not sample.image_path:
                    issues.append(f"Sample {i}: Missing image path")
                    continue
                
                if not os.path.exists(sample.image_path):
                    issues.append(f"Sample {i}: Image file not found: {sample.image_path}")
                    continue
                    
                # Check question
                if not sample.question or len(sample.question.strip()) == 0:
                    issues.append(f"Sample {i}: Empty question")
                    continue
                    
                # Check answers
                if not sample.answers or len(sample.answers) == 0:
                    issues.append(f"Sample {i}: No answers")
                    continue
                    
                valid_count += 1
                
            except Exception as e:
                issues.append(f"Sample {i}: Validation error - {e}")
                
        # Log results
        self.logger.key_value("Total samples", len(self.raw_data))
        self.logger.key_value("Valid samples", valid_count)
        self.logger.key_value("Invalid samples", len(issues))
        
        if issues:
            self.logger.warning(f"Found {len(issues)} validation issues")
            for issue in issues[:10]:  # Show first 10 issues
                self.logger.bullet(issue, indent=4)
            if len(issues) > 10:
                self.logger.bullet(f"... and {len(issues) - 10} more issues", indent=4)
        else:
            self.logger.success("All samples passed validation")
            
    def _compute_statistics(self):
        """Compute and log data statistics."""
        self.logger.subsection("Data Statistics")
        
        # Image statistics - load a few samples to get shape info
        image_shapes = []
        from PIL import Image
        for i, sample in enumerate(self.raw_data[:100]):  # Sample first 100 only
            try:
                if os.path.exists(sample.image_path):
                    with Image.open(sample.image_path) as img:
                        image_shapes.append((img.height, img.width, len(img.getbands())))
            except:
                pass
            if i >= 100:  # Limit to avoid memory issues
                break
                
        if image_shapes:
            shapes_array = np.array(image_shapes)
            self.logger.info("Image Statistics (sampled 100 images):")
            self.logger.key_value("Min shape (H,W,C)", tuple(shapes_array.min(axis=0)))
            self.logger.key_value("Max shape (H,W,C)", tuple(shapes_array.max(axis=0)))
            self.logger.key_value("Mean shape (H,W,C)", tuple(shapes_array.mean(axis=0).astype(int)))
            
        # Question statistics
        question_lengths = [len(sample.question.split()) for sample in self.raw_data]
        self.logger.info("Question Statistics:")
        self.logger.key_value("Min length (words)", min(question_lengths))
        self.logger.key_value("Max length (words)", max(question_lengths))
        self.logger.key_value("Mean length (words)", f"{np.mean(question_lengths):.1f}")
        
        # Answer statistics
        all_answers = []
        answer_counts_per_sample = []
        for sample in self.raw_data:
            all_answers.extend(sample.answers)
            answer_counts_per_sample.append(len(sample.answers))
            
        answer_counter = Counter(all_answers)
        self.logger.info("Answer Statistics:")
        self.logger.key_value("Total answers", len(all_answers))
        self.logger.key_value("Unique answers", len(answer_counter))
        self.logger.key_value("Answers per sample (mean)", f"{np.mean(answer_counts_per_sample):.1f}")
        
        # Top answers
        self.logger.info("Top 10 Most Common Answers:")
        for answer, count in answer_counter.most_common(10):
            self.logger.bullet(f'"{answer}": {count} ({count/len(self.raw_data)*100:.1f}%)', indent=4)
            
    def _split_data(self):
        """Split data into train/val/test sets."""
        self.logger.subsection("Data Splitting")
        
        from src.data.data_actions import split_data
        
        self.logger.key_value("Train ratio", self.config.train_ratio)
        self.logger.key_value("Validation ratio", self.config.val_ratio)
        self.logger.key_value("Test ratio", self.config.test_ratio)
        self.logger.key_value("Seed", self.config.seed)
        
        self.train_data, self.val_data, self.test_data = split_data(
            self.raw_data,
            train_ratio=self.config.train_ratio,
            val_ratio=self.config.val_ratio,
            is_random=True,
            seed=self.config.seed
        )
        
        self.logger.success("Data split completed:")
        self.logger.key_value("Training samples", len(self.train_data))
        self.logger.key_value("Validation samples", len(self.val_data))
        self.logger.key_value("Test samples", len(self.test_data))
        
        # Verify split
        total_split = len(self.train_data) + len(self.val_data) + len(self.test_data)
        if total_split != len(self.raw_data):
            self.logger.warning(f"Split mismatch: {total_split} != {len(self.raw_data)}")
        else:
            self.logger.success("Split verification passed")
            
    def _build_answer_vocabulary(self):
        """Build answer vocabulary from training data."""
        self.logger.subsection("Building Answer Vocabulary")
        
        from src.data.dataset import build_answer_vocab
        
        self.logger.key_value("Minimum frequency", self.config.min_answer_freq)
        
        self.answer2id = build_answer_vocab(
            self.train_data,
            min_freq=self.config.min_answer_freq
        )
        
        self.id2answer = {v: k for k, v in self.answer2id.items()}
        
        self.logger.success(f"Built vocabulary with {len(self.answer2id)} answer classes")
        self.logger.key_value("Vocabulary size", len(self.answer2id))
        self.logger.key_value("Includes <unk>", '<unk>' in self.answer2id)
        
        # Log sample vocabulary entries
        self.logger.info("Sample vocabulary entries:")
        for i, (answer, idx) in enumerate(list(self.answer2id.items())[:10]):
            self.logger.bullet(f'{idx}: "{answer}"', indent=4)
            
    def _initialize_tokenizer(self):
        """Initialize text tokenizer."""
        self.logger.subsection("Initializing Tokenizer")
        
        from src.modeling.tokenizer.pre_trained_tokenizer import PretrainedTokenizer
        
        self.logger.key_value("Tokenizer model", self.config.tokenizer_name)
        self.logger.key_value("Max sequence length", self.config.max_seq_length)
        
        self.tokenizer = PretrainedTokenizer(
            model_name=self.config.tokenizer_name,
            max_len=self.config.max_seq_length
        )
        
        self.logger.success("Tokenizer initialized")
        self.logger.key_value("Vocabulary size", self.tokenizer.vocab_size)
        
        # Test tokenizer
        self.logger.info("Tokenizer validation:")
        test_text = "Đây là một câu hỏi tiếng Việt mẫu"
        tokens = self.tokenizer(test_text)
        self.logger.key_value("Test input", f'"{test_text}"')
        self.logger.key_value("Input IDs shape", list(tokens['input_ids'].shape))
        self.logger.key_value("Attention mask shape", list(tokens['attention_mask'].shape))
        
        # Decode back
        decoded = self.tokenizer.decode(tokens['input_ids'])
        self.logger.key_value("Decoded text", f'"{decoded}"')
        
    def _build_transforms(self):
        """Build image transforms for training and evaluation."""
        self.logger.subsection("Building Image Transforms")
        
        from src.data.augmentation import ImageAugmentation
        
        self.logger.key_value("Image size", self.config.image_size)
        self.logger.key_value("Augmentation strength", self.config.augmentation_strength)
        self.logger.key_value("Normalize mean", self.config.normalize_mean)
        self.logger.key_value("Normalize std", self.config.normalize_std)
        
        # Training transform with augmentation
        train_aug = ImageAugmentation(
            mode="train",
            image_size=self.config.image_size,
            normalize_mean=self.config.normalize_mean,
            normalize_std=self.config.normalize_std,
            augmentation_strength=self.config.augmentation_strength
        )
        self.train_transform = train_aug.transform
        
        # Evaluation transform (no augmentation)
        eval_aug = ImageAugmentation(
            mode="eval",
            image_size=self.config.image_size,
            normalize_mean=self.config.normalize_mean,
            normalize_std=self.config.normalize_std
        )
        self.eval_transform = eval_aug.transform
        
        self.logger.success("Transforms built")
        
        # Log transform details
        self.logger.info("Training transforms:")
        for t in self.train_transform.transforms:
            self.logger.bullet(t.__class__.__name__, indent=4)
            
        self.logger.info("Evaluation transforms:")
        for t in self.eval_transform.transforms:
            self.logger.bullet(t.__class__.__name__, indent=4)
            
        # Validate transforms
        self.logger.info("Transform validation:")
        if len(self.train_data) > 0:
            sample = self.train_data[0]
            from PIL import Image
            
            # Load image from path
            try:
                pil_image = Image.open(sample.image_path).convert('RGB')
            except:
                # Use dummy image if loading fails
                pil_image = Image.new('RGB', (224, 224), (0, 0, 0))
                
            original_size = pil_image.size
            transformed = self.train_transform(pil_image)
            
            self.logger.log_augmentation_result(
                original_shape=original_size,
                augmented_shape=tuple(transformed.shape),
                augmentations_applied=[t.__class__.__name__ for t in self.train_transform.transforms]
            )
            
    def _create_dataloaders(self) -> DataPipelineOutput:
        """Create datasets and dataloaders."""
        self.logger.subsection("Creating DataLoaders")
        
        from src.data.dataset import VQADataset, create_data_loaders
        
        # Create datasets
        self.logger.info("Creating training dataset...")
        train_dataset = VQADataset(
            data_list=self.train_data,
            img_dir=self.config.images_dir,
            tokenizer=self.tokenizer,
            answer2id=self.answer2id,
            transform=self.train_transform,
            mode='train'
        )
        
        self.logger.info("Creating validation dataset...")
        val_dataset = VQADataset(
            data_list=self.val_data,
            img_dir=self.config.images_dir,
            tokenizer=self.tokenizer,
            answer2id=self.answer2id,
            transform=self.eval_transform,
            mode='val'
        )
        
        self.logger.info("Creating test dataset...")
        test_dataset = VQADataset(
            data_list=self.test_data,
            img_dir=self.config.images_dir,
            tokenizer=self.tokenizer,
            answer2id=self.answer2id,
            transform=self.eval_transform,
            mode='val'
        ) if self.test_data else None
        
        # Create dataloaders with custom collate function for VQA
        self.logger.info("Creating dataloaders...")
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            drop_last=True,
            collate_fn=vqa_collate_fn
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.eval_batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            collate_fn=vqa_collate_fn
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.eval_batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            collate_fn=vqa_collate_fn
        ) if test_dataset else None
        
        # Log dataloader info
        self.logger.log_dataloader_info(
            name="Training",
            num_samples=len(train_dataset),
            batch_size=self.config.batch_size,
            num_batches=len(train_loader),
            num_workers=self.config.num_workers
        )
        
        self.logger.log_dataloader_info(
            name="Validation",
            num_samples=len(val_dataset),
            batch_size=self.config.eval_batch_size,
            num_batches=len(val_loader),
            num_workers=self.config.num_workers
        )
        
        if test_loader:
            self.logger.log_dataloader_info(
                name="Test",
                num_samples=len(test_dataset),
                batch_size=self.config.eval_batch_size,
                num_batches=len(test_loader),
                num_workers=self.config.num_workers
            )
            
        return DataPipelineOutput(
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            train_samples=len(train_dataset),
            val_samples=len(val_dataset),
            test_samples=len(test_dataset) if test_dataset else 0,
            answer2id=self.answer2id,
            id2answer=self.id2answer,
            num_classes=len(self.answer2id),
            tokenizer=self.tokenizer,
            train_transform=self.train_transform,
            eval_transform=self.eval_transform
        )
        
    def _validate_dataloaders(self, output: DataPipelineOutput):
        """Validate dataloaders by getting sample batches."""
        self.logger.subsection("DataLoader Validation")
        
        self.logger.info("Fetching sample batch from training loader...")
        
        try:
            # Get one batch
            batch = next(iter(output.train_loader))
            
            self.logger.log_batch_sample(batch, batch_idx=0)
            
            # Validate batch structure
            required_keys = ['image', 'input_ids', 'attention_mask', 'label']
            missing_keys = [k for k in required_keys if k not in batch]
            
            if missing_keys:
                self.logger.warning(f"Missing keys in batch: {missing_keys}")
            else:
                self.logger.success("Batch structure validated")
                
            # Check batch size
            batch_size = batch['image'].shape[0]
            if batch_size != self.config.batch_size:
                self.logger.warning(f"Batch size mismatch: {batch_size} != {self.config.batch_size}")
            else:
                self.logger.success(f"Batch size verified: {batch_size}")
                
            # Check image dimensions
            expected_shape = (self.config.batch_size, 3, *self.config.image_size)
            actual_shape = tuple(batch['image'].shape)
            if actual_shape != expected_shape:
                self.logger.warning(f"Image shape mismatch: {actual_shape} != {expected_shape}")
            else:
                self.logger.success(f"Image shape verified: {actual_shape}")
                
            # Check label range
            max_label = batch['label'].max().item()
            min_label = batch['label'].min().item()
            self.logger.key_value("Label range", f"[{min_label}, {max_label}]")
            
            if max_label >= len(self.answer2id):
                self.logger.warning(f"Label {max_label} exceeds vocabulary size {len(self.answer2id)}")
            else:
                self.logger.success("Label range verified")
                
        except Exception as e:
            self.logger.error(f"DataLoader validation failed: {e}")
            raise
