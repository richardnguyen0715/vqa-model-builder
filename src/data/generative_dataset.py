"""
Generative VQA Dataset
Dataset for sequence-to-sequence VQA training.
Returns tokenized answer sequences instead of class labels.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import pandas as pd
from collections import Counter
from torchvision import transforms
from typing import List, Union, Dict, Optional, Any
from tqdm import tqdm

from src.schema.data_schema import OneSample
from src.middleware.logger import data_process_logger
from src.exception.data_exception_handling import MemoryOverflowException
from src.middleware.monitor import memory_monitor


class GenerativeVQADataset(Dataset):
    """
    Dataset for Generative VQA training.
    
    Returns tokenized question and answer sequences for seq2seq training.
    Uses teacher forcing: decoder_input_ids = [BOS] + answer[:-1]
                         labels = answer + [EOS]
    
    Supports both DataFrame and list input formats.
    """
    
    def __init__(
        self,
        df: Optional[pd.DataFrame] = None,
        data_list: Optional[List[Union[OneSample, Dict]]] = None,
        images_dir: str = "data/raw/images",
        img_dir: str = None,  # Alias for images_dir
        tokenizer: Any = None,
        question_tokenizer: Any = None,  # Alias
        answer_tokenizer: Any = None,  # Can be different from question tokenizer
        transform: Optional[transforms.Compose] = None,
        mode: str = 'train',
        max_question_length: int = 64,
        max_answer_length: int = 64,
        answer_selection: str = 'majority',  # 'majority', 'random', 'all'
        enable_memory_monitor: bool = True
    ):
        """
        Args:
            df: DataFrame with columns [image, question, answers, answer1-5]
            data_list: List of data samples (OneSample or dict) - alternative to df
            images_dir: Directory containing images
            img_dir: Alias for images_dir
            tokenizer: Main tokenizer (used for both question and answer if not specified separately)
            question_tokenizer: Tokenizer for questions (optional)
            answer_tokenizer: Tokenizer for answers (optional)
            transform: Image transforms
            mode: 'train', 'val', or 'test'
            max_question_length: Maximum question token length
            max_answer_length: Maximum answer token length
            answer_selection: How to select answer during training
            enable_memory_monitor: Whether to monitor memory
        """
        # Handle DataFrame input
        if df is not None:
            self.data = df.to_dict('records')
            self.use_df_format = True
        elif data_list is not None:
            self.data = data_list
            self.use_df_format = False
        else:
            raise ValueError("Either df or data_list must be provided")
        
        # Handle path alias
        self.img_dir = img_dir if img_dir is not None else images_dir
        
        # Handle tokenizer aliases
        self.tokenizer = tokenizer
        self.question_tokenizer = question_tokenizer if question_tokenizer is not None else tokenizer
        self.answer_tokenizer = answer_tokenizer if answer_tokenizer is not None else tokenizer
        
        if self.tokenizer is None and self.question_tokenizer is None:
            raise ValueError("Tokenizer must be provided")
        
        self.transform = transform
        self.mode = mode
        self.max_question_length = max_question_length
        self.max_answer_length = max_answer_length
        self.answer_selection = answer_selection
        self.enable_memory_monitor = enable_memory_monitor
        
        # Memory monitor
        global memory_monitor
        self.memory_monitor = memory_monitor if enable_memory_monitor else None
        
        # Get special token IDs from answer tokenizer
        tokenizer_for_ids = self.answer_tokenizer or self.tokenizer
        self.pad_token_id = getattr(tokenizer_for_ids, 'pad_token_id', 1)
        self.bos_token_id = getattr(tokenizer_for_ids, 'bos_token_id', 0)
        self.eos_token_id = getattr(tokenizer_for_ids, 'eos_token_id', 2)
        
        # If tokenizer doesn't have bos/eos, use cls/sep
        if self.bos_token_id is None:
            self.bos_token_id = getattr(tokenizer_for_ids, 'cls_token_id', 0)
        if self.eos_token_id is None:
            self.eos_token_id = getattr(tokenizer_for_ids, 'sep_token_id', 2)
        if self.pad_token_id is None:
            self.pad_token_id = 1
        
        # Default transform using CLIP processor format
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.48145466, 0.4578275, 0.40821073],  # CLIP normalization
                    std=[0.26862954, 0.26130258, 0.27577711]
                )
            ])
    
    def __len__(self):
        return len(self.data)
    
    def _load_image(self, item: Union[OneSample, Dict]) -> Image.Image:
        """Load and return PIL Image.
        
        Supports multiple formats:
        - 'image': direct filename or path
        - 'image_id': COCO-style ID (will search for common extensions)
        - OneSample with image_path
        """
        img_path = None
        
        if isinstance(item, OneSample):
            img_path = item.image_path
        else:
            # Try 'image' field first
            img_filename = item.get('image', '')
            
            # If empty, try 'image_id'
            if not img_filename:
                image_id = item.get('image_id', '')
                if image_id:
                    # Try common extensions
                    for ext in ['.jpg', '.jpeg', '.png', '.JPEG', '.JPG', '.PNG']:
                        # Try with zero-padded ID (COCO format: 000000xxxxxx.jpg)
                        padded_id = str(image_id).zfill(12)
                        potential_path = os.path.join(self.img_dir, f"{padded_id}{ext}")
                        if os.path.exists(potential_path):
                            img_path = potential_path
                            break
                        
                        # Try without padding
                        potential_path = os.path.join(self.img_dir, f"{image_id}{ext}")
                        if os.path.exists(potential_path):
                            img_path = potential_path
                            break
            else:
                if os.path.isabs(img_filename):
                    img_path = img_filename
                else:
                    img_path = os.path.join(self.img_dir, img_filename)
        
        # Try to load image
        try:
            if img_path and os.path.exists(img_path):
                image = Image.open(img_path).convert('RGB')
            else:
                data_process_logger.warning(f"Image not found: {img_path or item}")
                image = Image.new('RGB', (224, 224), (128, 128, 128))  # Gray placeholder
        except Exception as e:
            data_process_logger.warning(f"Failed to load image {img_path}: {e}")
            image = Image.new('RGB', (224, 224), (128, 128, 128))
        
        return image
    
    def _get_question(self, item: Union[OneSample, Dict]) -> str:
        """Get question text."""
        if isinstance(item, OneSample):
            return item.question
        return item.get('question', '')
    
    def _get_answers(self, item: Union[OneSample, Dict]) -> List[str]:
        """Get all answers (supports both 'answers' list and 'answer1-5' columns)."""
        if isinstance(item, OneSample):
            return item.answers if item.answers else []
        
        # Check for 'answers' key first
        if 'answers' in item and item['answers']:
            answers = item['answers']
            if isinstance(answers, str):
                # Parse string representation of list
                try:
                    import ast
                    answers = ast.literal_eval(answers)
                except:
                    answers = [answers]
            return answers
        
        # Check for individual answer columns (answer1, answer2, ..., answer5)
        answers = []
        for i in range(1, 6):
            key = f'answer{i}'
            if key in item:
                ans = item[key]
                if ans and str(ans).strip() and str(ans).lower() != 'nan':
                    answers.append(str(ans).strip())
        
        return answers if answers else []
    
    def _select_answer(self, answers: List[str]) -> str:
        """Select answer based on strategy."""
        if not answers:
            return ""
        
        if self.answer_selection == 'majority':
            # Most common answer
            return Counter(answers).most_common(1)[0][0]
        elif self.answer_selection == 'random':
            # Random answer (useful for data augmentation)
            import random
            return random.choice(answers)
        else:
            # 'all' - return first (for deterministic behavior)
            return answers[0]
    
    def _tokenize_question(self, question: str) -> Dict[str, torch.Tensor]:
        """Tokenize question."""
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
        Tokenize answer for seq2seq training.
        
        Returns:
            decoder_input_ids: [BOS] + answer_tokens (teacher forcing input)
            decoder_attention_mask: attention mask for decoder input
            labels: answer_tokens + [EOS] (target for loss computation)
        """
        # Tokenize answer without special tokens
        # We'll add BOS/EOS manually for more control
        if hasattr(self.answer_tokenizer, 'encode'):
            # For raw tokenizer
            answer_ids = self.answer_tokenizer.encode(
                answer,
                add_special_tokens=False,
                max_length=self.max_answer_length - 2,  # Leave room for BOS/EOS
                truncation=True
            )
        else:
            # For wrapped tokenizer (PretrainedTokenizer)
            tokenized = self.answer_tokenizer(answer)
            answer_ids = tokenized['input_ids'].tolist()
            # Remove existing special tokens if any
            if answer_ids and answer_ids[0] == self.bos_token_id:
                answer_ids = answer_ids[1:]
            if answer_ids and answer_ids[-1] == self.eos_token_id:
                answer_ids = answer_ids[:-1]
            # Truncate
            answer_ids = answer_ids[:self.max_answer_length - 2]
        
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
            labels = labels + [-100] * padding_length  # -100 is ignored by CrossEntropyLoss
        else:
            labels = labels[:self.max_answer_length]
        
        # Create attention mask (1 for real tokens, 0 for padding)
        decoder_attention_mask = [1 if i < decoder_length else 0 for i in range(self.max_answer_length)]
        
        return {
            'decoder_input_ids': torch.tensor(decoder_input_ids, dtype=torch.long),
            'decoder_attention_mask': torch.tensor(decoder_attention_mask, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long)
        }
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.data[idx]
        
        # Memory check
        if self.memory_monitor is not None and idx % 100 == 0:
            try:
                self.memory_monitor.check_memory_usage(f"Loading item {idx}")
            except MemoryOverflowException as e:
                data_process_logger.critical(f"Memory overflow at item {idx}: {e}")
                raise
        
        # 1. Load and transform image
        image = self._load_image(item)
        if self.transform:
            image = self.transform(image)
        
        # 2. Get and tokenize question
        question = self._get_question(item)
        question_tokenized = self._tokenize_question(question)
        
        # 3. Get all answers
        all_answers = self._get_answers(item)
        
        # 4. Select and tokenize target answer
        selected_answer = self._select_answer(all_answers)
        answer_tokenized = self._tokenize_answer(selected_answer)
        
        return {
            # Image
            'image': image,  # [C, H, W]
            
            # Question (encoder input)
            'input_ids': question_tokenized['input_ids'],  # [q_seq]
            'attention_mask': question_tokenized['attention_mask'],  # [q_seq]
            
            # Answer (decoder input/output)
            'decoder_input_ids': answer_tokenized['decoder_input_ids'],  # [a_seq]
            'decoder_attention_mask': answer_tokenized['decoder_attention_mask'],  # [a_seq]
            'labels': answer_tokenized['labels'],  # [a_seq]
            
            # Raw text (for evaluation)
            'question': question,
            'answer': selected_answer,
            'all_answers': all_answers
        }


def generative_vqa_collate_fn(batch: List[Dict]) -> Dict[str, Any]:
    """
    Custom collate function for GenerativeVQADataset.
    
    Args:
        batch: List of sample dictionaries
        
    Returns:
        Collated batch with stacked tensors and lists
    """
    # Tensor fields
    images = torch.stack([s['image'] for s in batch])
    input_ids = torch.stack([s['input_ids'] for s in batch])
    attention_masks = torch.stack([s['attention_mask'] for s in batch])
    decoder_input_ids = torch.stack([s['decoder_input_ids'] for s in batch])
    decoder_attention_masks = torch.stack([s['decoder_attention_mask'] for s in batch])
    labels = torch.stack([s['labels'] for s in batch])
    
    # List fields (for evaluation)
    questions = [s['question'] for s in batch]
    answers = [s['answer'] for s in batch]
    all_answers = [s['all_answers'] for s in batch]
    
    return {
        'image': images,
        'input_ids': input_ids,
        'attention_mask': attention_masks,
        'decoder_input_ids': decoder_input_ids,
        'decoder_attention_mask': decoder_attention_masks,
        'labels': labels,
        'question': questions,
        'answer': answers,
        'all_answers': all_answers
    }


def create_generative_dataloader(
    dataset: GenerativeVQADataset,
    batch_size: int = 32,
    num_workers: int = 4,
    shuffle: bool = True,
    pin_memory: bool = True
) -> DataLoader:
    """
    Create DataLoader for GenerativeVQADataset.
    
    Args:
        dataset: GenerativeVQADataset instance
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
        collate_fn=generative_vqa_collate_fn
    )
