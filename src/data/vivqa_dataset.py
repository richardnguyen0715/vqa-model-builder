"""
VIVQA Dataset
Dataset class for Vietnamese VQA dataset with single answers.
Adapted from GenerativeVQADataset for VIVQA structure.

The VIVQA dataset has:
- question: Vietnamese question text
- answer: Single Vietnamese answer (not multiple choices)
- img_id: COCO image ID (not filename)
- type: Question type (typically 2 for color questions)
"""

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import pandas as pd
from torchvision import transforms
from typing import List, Union, Dict, Optional, Any
from tqdm import tqdm

from src.middleware.logger import data_process_logger


class VivqaDataset(Dataset):
    """
    Dataset class for Vietnamese VQA (VIVQA).
    
    Supports loading from CSV with structure:
    - question: Vietnamese question
    - answer: Single Vietnamese answer
    - img_id: COCO image ID (zero-padded to 12 digits)
    - type: Question type
    
    Returns tokenized question and answer sequences for seq2seq models.
    """
    
    def __init__(
        self,
        csv_path: str,
        images_dir: str = "data/vivqa/images",
        tokenizer: Any = None,
        question_tokenizer: Any = None,
        answer_tokenizer: Any = None,
        transform: Optional[transforms.Compose] = None,
        mode: str = 'inference',
        max_question_length: int = 64,
        max_answer_length: int = 64,
        use_logging: bool = True
    ):
        """
        Initialize VIVQA Dataset.
        
        Args:
            csv_path: Path to CSV file (train.csv or test.csv)
            images_dir: Directory containing COCO images
            tokenizer: Main tokenizer (used for both question and answer if not specified)
            question_tokenizer: Tokenizer for questions (optional)
            answer_tokenizer: Tokenizer for answers (optional)
            transform: Image transforms (transforms.Compose)
            mode: Dataset mode ('train', 'val', or 'inference')
            max_question_length: Maximum question token length
            max_answer_length: Maximum answer token length
            use_logging: Whether to log information
        """
        if use_logging:
            data_process_logger.info(f"Loading VIVQA dataset from: {csv_path}")
        
        # Load CSV file
        try:
            self.df = pd.read_csv(csv_path, index_col=0)
            if use_logging:
                data_process_logger.info(f"Loaded {len(self.df)} samples from {csv_path}")
        except Exception as e:
            if use_logging:
                data_process_logger.error(f"Failed to load CSV {csv_path}: {e}")
            raise
        
        self.images_dir = images_dir
        self.mode = mode
        self.max_question_length = max_question_length
        self.max_answer_length = max_answer_length
        self.use_logging = use_logging
        
        # Setup tokenizers
        self.tokenizer = tokenizer
        self.question_tokenizer = question_tokenizer if question_tokenizer is not None else tokenizer
        self.answer_tokenizer = answer_tokenizer if answer_tokenizer is not None else tokenizer
        
        if self.tokenizer is None and self.question_tokenizer is None:
            raise ValueError("Tokenizer must be provided")
        
        # Get special token IDs
        tokenizer_for_ids = self.answer_tokenizer or self.tokenizer
        self.pad_token_id = getattr(tokenizer_for_ids, 'pad_token_id', 1)
        self.bos_token_id = getattr(tokenizer_for_ids, 'bos_token_id', 0)
        self.eos_token_id = getattr(tokenizer_for_ids, 'eos_token_id', 2)
        
        # Fallback if special tokens not available
        if self.bos_token_id is None:
            self.bos_token_id = getattr(tokenizer_for_ids, 'cls_token_id', 0)
        if self.eos_token_id is None:
            self.eos_token_id = getattr(tokenizer_for_ids, 'sep_token_id', 2)
        if self.pad_token_id is None:
            self.pad_token_id = 1
        
        # Setup image transforms (default to CLIP normalization)
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.48145466, 0.4578275, 0.40821073],
                    std=[0.26862954, 0.26130258, 0.27577711]
                )
            ])
        else:
            self.transform = transform
        
        if use_logging:
            data_process_logger.info(
                f"VivqaDataset initialized: mode={mode}, "
                f"max_question_length={max_question_length}, "
                f"max_answer_length={max_answer_length}"
            )
    
    def __len__(self) -> int:
        """Return number of samples in dataset."""
        return len(self.df)
    
    def _load_image(self, img_id: int) -> Image.Image:
        """
        Load image from COCO image ID.
        
        Tries multiple filename formats:
        1. Standard COCO format: COCO_train2014_000000XXXXXX.jpg
        2. Without prefix
        3. With alternative extensions
        
        Args:
            img_id: COCO image ID
            
        Returns:
            PIL Image (RGB)
        """
        img_path = None
        
        # Try standard COCO format
        for ext in ['.jpg', '.jpeg', '.png', '.JPEG', '.JPG', '.PNG']:
            # Try zero-padded (standard COCO format)
            padded_id = str(img_id).zfill(12)
            potential_path = os.path.join(self.images_dir, f"COCO_train2014_{padded_id}{ext}")
            if os.path.exists(potential_path):
                img_path = potential_path
                break
            
            # Try with just ID
            potential_path = os.path.join(self.images_dir, f"{padded_id}{ext}")
            if os.path.exists(potential_path):
                img_path = potential_path
                break
        
        # Load or create placeholder
        try:
            if img_path and os.path.exists(img_path):
                image = Image.open(img_path).convert('RGB')
            else:
                if self.use_logging and img_path:
                    data_process_logger.warning(f"Image not found for img_id {img_id}: {img_path}")
                # Create gray placeholder
                image = Image.new('RGB', (224, 224), (128, 128, 128))
        except Exception as e:
            if self.use_logging:
                data_process_logger.warning(f"Failed to load image for img_id {img_id}: {e}")
            image = Image.new('RGB', (224, 224), (128, 128, 128))
        
        return image
    
    def _tokenize_question(self, question: str) -> Dict[str, torch.Tensor]:
        """
        Tokenize question text.
        
        Args:
            question: Question text (Vietnamese)
            
        Returns:
            Dictionary with input_ids and attention_mask
        """
        tokenized = self.question_tokenizer(
            question,
            max_length=self.max_question_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': tokenized['input_ids'].squeeze(0),
            'attention_mask': tokenized['attention_mask'].squeeze(0)
        }
    
    def _tokenize_answer(self, answer: str) -> Dict[str, torch.Tensor]:
        """
        Tokenize answer text for seq2seq training/inference.
        
        For training: creates decoder_input_ids and labels for teacher forcing
        For inference: only creates input_ids (no labels)
        
        Args:
            answer: Answer text (Vietnamese)
            
        Returns:
            Dictionary with answer tokenization
        """
        # Tokenize without special tokens
        if hasattr(self.answer_tokenizer, 'encode'):
            answer_ids = self.answer_tokenizer.encode(
                answer,
                add_special_tokens=False,
                max_length=self.max_answer_length - 2,
                truncation=True
            )
        else:
            tokenized = self.answer_tokenizer(
                answer,
                add_special_tokens=False,
                max_length=self.max_answer_length - 2,
                truncation=True
            )
            answer_ids = tokenized['input_ids'] if isinstance(tokenized['input_ids'], list) else tokenized['input_ids'].tolist()
        
        # Create decoder_input_ids: [BOS] + answer
        decoder_input_ids = [self.bos_token_id] + answer_ids
        
        # Create labels: answer + [EOS]
        labels = answer_ids + [self.eos_token_id]
        
        # Pad to max_answer_length
        decoder_length = len(decoder_input_ids)
        labels_length = len(labels)
        
        # Pad decoder_input_ids
        if decoder_length < self.max_answer_length:
            padding_length = self.max_answer_length - decoder_length
            decoder_input_ids = decoder_input_ids + [self.pad_token_id] * padding_length
        else:
            decoder_input_ids = decoder_input_ids[:self.max_answer_length]
        
        # Pad labels (use -100 for ignored positions in loss)
        if labels_length < self.max_answer_length:
            padding_length = self.max_answer_length - labels_length
            labels = labels + [-100] * padding_length
        else:
            labels = labels[:self.max_answer_length]
        
        # Create attention mask
        decoder_attention_mask = [1 if i < decoder_length else 0 for i in range(self.max_answer_length)]
        
        return {
            'decoder_input_ids': torch.tensor(decoder_input_ids, dtype=torch.long),
            'decoder_attention_mask': torch.tensor(decoder_attention_mask, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long)
        }
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get dataset sample.
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary containing:
            - image: Transformed image tensor [C, H, W]
            - input_ids: Question token IDs
            - attention_mask: Question attention mask
            - decoder_input_ids: Answer decoder input IDs
            - decoder_attention_mask: Answer decoder attention mask
            - labels: Answer labels for loss computation
            - question: Raw question text
            - answer: Raw answer text
            - img_id: COCO image ID
            - type: Question type
        """
        row = self.df.iloc[idx]
        
        # Extract fields
        question = str(row['question']).strip()
        answer = str(row['answer']).strip()
        img_id = int(row['img_id'])
        question_type = int(row.get('type', 2))  # Default to type 2 if not present
        
        # Load image
        image = self._load_image(img_id)
        if self.transform:
            image = self.transform(image)
        
        # Tokenize question
        question_tokenized = self._tokenize_question(question)
        
        # Tokenize answer
        answer_tokenized = self._tokenize_answer(answer)
        
        return {
            # Image
            'image': image,
            
            # Question (encoder input)
            'input_ids': question_tokenized['input_ids'],
            'attention_mask': question_tokenized['attention_mask'],
            
            # Answer (decoder input/output)
            'decoder_input_ids': answer_tokenized['decoder_input_ids'],
            'decoder_attention_mask': answer_tokenized['decoder_attention_mask'],
            'labels': answer_tokenized['labels'],
            
            # Raw data (for evaluation/logging)
            'question': question,
            'answer': answer,
            'img_id': img_id,
            'type': question_type
        }


def vivqa_collate_fn(batch: List[Dict]) -> Dict[str, Any]:
    """
    Custom collate function for VivqaDataset.
    
    Args:
        batch: List of sample dictionaries
        
    Returns:
        Batched data with stacked tensors and lists
    """
    # Stack tensor fields
    images = torch.stack([s['image'] for s in batch])
    input_ids = torch.stack([s['input_ids'] for s in batch])
    attention_masks = torch.stack([s['attention_mask'] for s in batch])
    decoder_input_ids = torch.stack([s['decoder_input_ids'] for s in batch])
    decoder_attention_masks = torch.stack([s['decoder_attention_mask'] for s in batch])
    labels = torch.stack([s['labels'] for s in batch])
    
    # Keep lists for metadata
    questions = [s['question'] for s in batch]
    answers = [s['answer'] for s in batch]
    img_ids = [s['img_id'] for s in batch]
    types = [s['type'] for s in batch]
    
    return {
        'image': images,
        'input_ids': input_ids,
        'attention_mask': attention_masks,
        'decoder_input_ids': decoder_input_ids,
        'decoder_attention_mask': decoder_attention_masks,
        'labels': labels,
        'question': questions,
        'answer': answers,
        'img_id': img_ids,
        'type': types
    }


def create_vivqa_dataloader(
    dataset: VivqaDataset,
    batch_size: int = 32,
    num_workers: int = 4,
    shuffle: bool = False,
    pin_memory: bool = True
) -> DataLoader:
    """
    Create DataLoader for VivqaDataset.
    
    Args:
        dataset: VivqaDataset instance
        batch_size: Batch size
        num_workers: Number of data loading workers
        shuffle: Whether to shuffle data
        pin_memory: Whether to pin memory
        
    Returns:
        DataLoader instance
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=vivqa_collate_fn
    )
