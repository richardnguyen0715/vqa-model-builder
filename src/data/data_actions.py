import os
import cv2
import numpy as np
import pandas as pd
import ast
from typing import List
from tqdm import tqdm

from src.middleware.logger import data_process_logger
from src.schema.data_schema import OneSample
from utils.path_management import PROCESSED_DATA_DIR
from src.exception.data_exception_handling import MemoryOverflowException


def load_image(image_path: str) -> np.ndarray:
    """Load an image from the specified file path.

    Args:
        image_path: Path to the image file.
    Returns:
        Loaded image as a NumPy array.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at path: {image_path}")
    return image


def load_text_data(text_file_path: str) -> pd.DataFrame:
    """Load text data from a CSV file.

    Args:
        text_file_path: Path to the CSV text file.
    Returns:
        DataFrame containing the text data.
    """
    try:
        df = pd.read_csv(text_file_path)
        return df
    except Exception as e:
        data_process_logger.error(f"Failed to load text data from {text_file_path}: {e}")
        raise


def get_all_image_paths(directory: str) -> list:
    """Get a list of all image file paths in the specified directory.

    Args:
        directory: Path to the directory containing images.
    Returns:
        List of image file paths.
    """
    image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
    image_paths = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(image_extensions):
                image_paths.append(os.path.join(root, file))
    data_process_logger.info(f"Found {len(image_paths)} images in directory: {directory}")
    return image_paths


def load_raw_data(images_dir: str, text_file_path: str) -> List[OneSample]:
    """Load raw data samples from images and text data.

    Args:
        images_dir: Directory containing raw images.
        text_file_path: Path to the text data file (CSV).
    Returns:
        List of OneSample instances.
    """
    # Lazy import to avoid circular dependency
    from src.middleware.monitor import memory_monitor
    
    data_samples = []
    
    # Load text data
    data_process_logger.info(f"Loading text data from: {text_file_path}")
    memory_monitor.check_memory_usage("Before loading text data")
    
    text_data = load_text_data(text_file_path)
    
    # Check required columns
    required_columns = ['image_link', 'question', 'answers']
    missing_columns = [col for col in required_columns if col not in text_data.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    # Get all image paths
    data_process_logger.info(f"Retrieving image paths from directory: {images_dir}")
    memory_monitor.check_memory_usage("Before retrieving image paths")
    
    image_paths = get_all_image_paths(images_dir)
    # Create a mapping from filename to full path
    image_path_map = {os.path.basename(p): p for p in image_paths}

    # Iterate through text data and match with images with progress bar
    for idx, row in tqdm(text_data.iterrows(), total=len(text_data), desc="Loading data samples"):
        try:
            # Check memory every 100 samples
            if idx % 100 == 0:
                memory_monitor.check_memory_usage(f"Loading sample {idx}")
            
            # Extract image filename from image_link URL
            # Example: http://images.cocodataset.org/train2017/000000581569.jpg -> 000000581569.jpg
            image_link = row['image_link']
            image_filename = os.path.basename(image_link)
            
            question = row['question']
            
            # Parse answers from string representation to list
            answers_str = row['answers']
            if isinstance(answers_str, str):
                answers = ast.literal_eval(answers_str)
            else:
                answers = answers_str
            
            # Check if answers is a list
            if not isinstance(answers, list):
                data_process_logger.warning(f"Row {idx}: answers is not a list, skipping")
                continue

            # Find corresponding image
            if image_filename in image_path_map:
                image_path = image_path_map[image_filename]
                image = load_image(image_path)

                sample = OneSample(
                    image=image,
                    question=question,
                    answers=answers,
                    metadata={
                        "image_path": image_path,
                        "answer_count": len(answers)
                    }
                )
                data_samples.append(sample)
            else:
                data_process_logger.warning(f"Image file not found for entry {idx}: {image_filename}")
        
        except MemoryOverflowException as e:
            data_process_logger.critical(f"Memory overflow at sample {idx}: {e}")
            memory_report = memory_monitor.get_memory_report()
            data_process_logger.critical(f"Memory report: {memory_report}")
            raise
        except Exception as e:
            data_process_logger.error(f"Error processing row {idx}: {e}")
            continue

    memory_monitor.check_memory_usage("After loading all samples")
    memory_report = memory_monitor.get_memory_report()
    data_process_logger.info(f"Loaded {len(data_samples)} data samples successfully.")
    data_process_logger.info(f"Memory report: {memory_report}")
    
    return data_samples


def validate_data(raw_data):
    """Validate the loaded raw data samples."""
    if not raw_data:
        data_process_logger.error("No data samples loaded.")
        raise ValueError("No data samples loaded.")
    for i, sample in enumerate(raw_data):
        if sample.image.size == 0:
            data_process_logger.error(f"Sample {i} has empty image data.")
            raise ValueError(f"Sample {i} has empty image data.")
        if not sample.question:
            data_process_logger.error(f"Sample {i} has empty question.")
            raise ValueError(f"Sample {i} has empty question.")
        if not sample.answers:
            data_process_logger.error(f"Sample {i} has empty answers.")
            raise ValueError(f"Sample {i} has empty answers.")
        
        
def split_data(raw_data, train_ratio=0.8, val_ratio=0.1, is_random=True):
    """Split raw data into training, validation, and test sets.

    Args:
        raw_data: List of OneSample instances.
        train_ratio: Proportion of data to use for training.
        val_ratio: Proportion of data to use for validation.
        is_random: Whether to shuffle data before splitting.
    Returns:
        Tuple of (train_data, val_data, test_data).
    """
    total_samples = len(raw_data)
    if is_random:
        import random
        random.shuffle(raw_data)

    train_end = int(total_samples * train_ratio)
    val_end = train_end + int(total_samples * val_ratio)

    train_data = raw_data[:train_end]
    val_data = raw_data[train_end:val_end]
    test_data = raw_data[val_end:]

    data_process_logger.info(f"Data split into {len(train_data)} training, {len(val_data)} validation, and {len(test_data)} test samples. IS_RANDOM={is_random}")
    return train_data, val_data, test_data


def load_data_split(images_dir: str, text_file_path: str, split_type: str = 'train', 
                    train_ratio: float = 0.8, val_ratio: float = 0.1, 
                    start_idx: int = None, end_idx: int = None) -> List[OneSample]:
    """
    Load a specific data split (train/val/test) without loading all data into memory at once.
    
    Args:
        images_dir: Directory containing raw images.
        text_file_path: Path to the text data file (CSV).
        split_type: 'train', 'val', or 'test' - which split to load
        train_ratio: Proportion of data for training
        val_ratio: Proportion of data for validation
        start_idx: Start index for custom range (if None, auto-calculated)
        end_idx: End index for custom range (if None, auto-calculated)
        
    Returns:
        List of OneSample instances for the specified split.
    """
    # Lazy import to avoid circular dependency
    from src.middleware.monitor import memory_monitor
    
    data_samples = []
    
    # Load text data
    data_process_logger.info(f"Loading text data from: {text_file_path}")
    memory_monitor.check_memory_usage(f"Before loading {split_type} split")
    
    text_data = load_text_data(text_file_path)
    total_samples = len(text_data)
    
    # Check required columns
    required_columns = ['image_link', 'question', 'answers']
    missing_columns = [col for col in required_columns if col not in text_data.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    # Calculate split indices if not provided
    if start_idx is None or end_idx is None:
        train_end = int(total_samples * train_ratio)
        val_end = train_end + int(total_samples * val_ratio)
        
        if split_type == 'train':
            start_idx, end_idx = 0, train_end
        elif split_type == 'val':
            start_idx, end_idx = train_end, val_end
        elif split_type == 'test':
            start_idx, end_idx = val_end, total_samples
        else:
            raise ValueError(f"Invalid split_type: {split_type}")

    # Get all image paths
    data_process_logger.info(f"Retrieving image paths from directory: {images_dir}")
    image_paths = get_all_image_paths(images_dir)
    image_path_map = {os.path.basename(p): p for p in image_paths}

    # Load only the specified split
    data_process_logger.info(f"Loading {split_type} split (indices {start_idx}-{end_idx})")
    
    for idx in tqdm(range(start_idx, end_idx), desc=f"Loading {split_type} data", total=(end_idx - start_idx)):
        try:
            # Check memory every 100 samples
            if (idx - start_idx) % 100 == 0:
                memory_monitor.check_memory_usage(f"Loading {split_type} sample {idx - start_idx}")
            
            row = text_data.iloc[idx]
            
            # Extract image filename from image_link URL
            image_link = row['image_link']
            image_filename = os.path.basename(image_link)
            
            question = row['question']
            
            # Parse answers from string representation to list
            answers_str = row['answers']
            if isinstance(answers_str, str):
                answers = ast.literal_eval(answers_str)
            else:
                answers = answers_str
            
            # Check if answers is a list
            if not isinstance(answers, list):
                data_process_logger.warning(f"Row {idx}: answers is not a list, skipping")
                continue

            # Find corresponding image
            if image_filename in image_path_map:
                image_path = image_path_map[image_filename]
                image = load_image(image_path)

                sample = OneSample(
                    image=image,
                    question=question,
                    answers=answers,
                    metadata={
                        "image_path": image_path,
                        "answer_count": len(answers)
                    }
                )
                data_samples.append(sample)
            else:
                data_process_logger.warning(f"Image file not found for entry {idx}: {image_filename}")
        
        except MemoryOverflowException as e:
            data_process_logger.critical(f"Memory overflow at {split_type} sample {idx}: {e}")
            memory_report = memory_monitor.get_memory_report()
            data_process_logger.critical(f"Memory report: {memory_report}")
            raise
        except Exception as e:
            data_process_logger.error(f"Error processing row {idx}: {e}")
            continue

    memory_monitor.check_memory_usage(f"After loading {split_type} split")
    memory_report = memory_monitor.get_memory_report()
    data_process_logger.info(f"Loaded {len(data_samples)} {split_type} samples successfully.")
    data_process_logger.info(f"Memory report: {memory_report}")
    
    return data_samples


def save_data(train_data, val_data, test_data):
    """Save the split data into processed data directory.

    Args:
        train_data: List of training data samples.
        val_data: List of validation data samples.
        test_data: List of test data samples.
        
    Action:
        - Move all images to raw/train, raw/val, raw/test directories.
        - Save text metadata as json 
    """
    
    import os
    import json
    from pathlib import Path
    from shutil import copy2

    splits = {
        'train': train_data,
        'val': val_data,
        'test': test_data
    }

    for split_name, data_samples in splits.items():
        split_image_dir = PROCESSED_DATA_DIR / 'raw' / split_name
        split_image_dir.mkdir(parents=True, exist_ok=True)
        metadata_list = []

        for sample in tqdm(data_samples, desc=f"Saving {split_name} data"):
            image_filename = os.path.basename(sample.image_path)
            dest_image_path = split_image_dir / image_filename
            copy2(sample.image_path, dest_image_path)

            metadata = {
                'image_path': str(dest_image_path),
                'question': sample.question,
                'answers': sample.answers,
                'metadata': sample.metadata
            }
            metadata_list.append(metadata)

        metadata_file_path = PROCESSED_DATA_DIR / f'{split_name}_metadata.json'
        with open(metadata_file_path, 'w') as f:
            json.dump(metadata_list, f, indent=4)

        data_process_logger.info(f"Saved {len(data_samples)} samples to {split_name} split at {split_image_dir} and metadata to {metadata_file_path}")
    