import os
import cv2
import numpy as np
import pandas as pd
import ast
from typing import List


from src.middleware.logger import data_process_logger
from src.schema.data_schema import OneSample


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
    data_samples = []
    
    # Load text data
    data_process_logger.info(f"Loading text data from: {text_file_path}")
    text_data = load_text_data(text_file_path)
    
    # Check required columns
    required_columns = ['image_link', 'question', 'answers']
    missing_columns = [col for col in required_columns if col not in text_data.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    # Get all image paths
    data_process_logger.info(f"Retrieving image paths from directory: {images_dir}")
    image_paths = get_all_image_paths(images_dir)
    # Create a mapping from filename to full path
    image_path_map = {os.path.basename(p): p for p in image_paths}

    # Iterate through text data and match with images
    for idx, row in text_data.iterrows():
        try:
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
        
        except Exception as e:
            data_process_logger.error(f"Error processing row {idx}: {e}")
            continue

    data_process_logger.info(f"Loaded {len(data_samples)} data samples successfully.")
    return data_samples