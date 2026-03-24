"""
Data Download and Organization Script
Handles downloading datasets from Kaggle and organizing them into project structure.
Supports both VQA and VIVQA datasets.
"""

import kagglehub
import shutil
from pathlib import Path
from typing import Optional
import logging

from src.middleware.logger import data_loader_logger

# Try to load config, but make it optional for flexibility
try:
    from src.middleware.config import data_config
    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False


def download_and_organize_vqa_data():
    """
    Download VQA dataset from Kaggle and organize into project structure.
    
    This function handles the original VQA dataset structure.
    Requires data_config to be available.
    """
    if not CONFIG_AVAILABLE:
        data_loader_logger.error("Config module not available. Cannot download VQA dataset.")
        raise ImportError("src.middleware.config not available")
    
    # Load configuration
    data_loader_logger.info("Loading data configuration from configs/data_configs.yaml")
    config = data_config
    
    data_paths = config['data_path']
    kaggle_setup = config['kaggle_setup']
    
    # Step 1: Download from Kaggle
    data_loader_logger.info(f"Starting Kaggle dataset download: {kaggle_setup['kaggle_project']}")
    try:
        download_path = kagglehub.dataset_download(kaggle_setup['kaggle_project'])
        data_loader_logger.info(f"Dataset downloaded successfully to: {download_path}")
    except Exception as e:
        data_loader_logger.error(f"Failed to download dataset: {e}")
        raise
    
    download_path = Path(download_path)
    
    # Step 2: Create target directories
    raw_images_dir = Path(data_paths['raw_images'])
    raw_texts_dir = Path(data_paths['raw_texts'])
    
    data_loader_logger.info(f"Creating target directories: {raw_images_dir}, {raw_texts_dir}")
    raw_images_dir.mkdir(parents=True, exist_ok=True)
    raw_texts_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 3: Move images
    source_images = download_path / kaggle_setup['images_folder']
    if source_images.exists():
        data_loader_logger.info(f"Moving images from {source_images} to {raw_images_dir}")
        image_files = list(source_images.glob('*'))
        data_loader_logger.info(f"Found {len(image_files)} files in images folder")
        
        moved_count = 0
        for img_file in image_files:
            if img_file.is_file():
                target = raw_images_dir / img_file.name
                shutil.move(str(img_file), str(target))
                moved_count += 1
        
        data_loader_logger.info(f"Successfully moved {moved_count} image files")
    else:
        data_loader_logger.warning(f"Images folder not found: {source_images}")
    
    # Step 4: Move text file
    source_text = download_path / kaggle_setup['text_file']
    if source_text.exists():
        target_text = raw_texts_dir / source_text.name
        data_loader_logger.info(f"Moving text file from {source_text} to {target_text}")
        shutil.move(str(source_text), str(target_text))
        data_loader_logger.info("Text file moved successfully")
    else:
        data_loader_logger.warning(f"Text file not found: {source_text}")
    
    # Step 5: Clean up downloaded folder
    data_loader_logger.info(f"Cleaning up temporary download folder: {download_path}")
    try:
        if download_path.exists():
            shutil.rmtree(download_path)
            data_loader_logger.info("Temporary folder removed successfully")
    except Exception as e:
        data_loader_logger.warning(f"Could not remove temporary folder: {e}")
    
    data_loader_logger.info("VQA dataset download and organization completed successfully")


def download_and_organize_vivqa_data(
    kaggle_dataset: str = "dngtrungngha/vivqa",
    output_dir: str = "data/vivqa"
) -> str:
    """
    Download VIVQA dataset from Kaggle and organize into project structure.
    
    This function handles the VIVQA dataset structure which includes:
    - data/images/train/ (training images)
    - data/images/test/ (test images)
    - data/train.csv (training data)
    - data/test.csv (test data)
    
    The function combines all images into a single 'images' folder and moves
    CSV files to the output directory.
    
    Args:
        kaggle_dataset: Kaggle dataset identifier (format: username/dataset_name)
        output_dir: Target directory for organized data (default: data/vivqa)
        
    Returns:
        Path to the organized dataset directory
        
    Raises:
        Exception: If download or organization fails
    """
    
    data_loader_logger.info("=" * 80)
    data_loader_logger.info("VIVQA DATASET DOWNLOAD AND ORGANIZATION")
    data_loader_logger.info("=" * 80)
    
    # Step 1: Download from Kaggle
    data_loader_logger.info(f"Downloading VIVQA dataset from Kaggle: {kaggle_dataset}")
    try:
        download_path = kagglehub.dataset_download(kaggle_dataset)
        data_loader_logger.info(f"Dataset downloaded successfully to: {download_path}")
    except Exception as e:
        data_loader_logger.error(f"Failed to download VIVQA dataset: {e}")
        raise
    
    download_path = Path(download_path)
    output_dir = Path(output_dir)
    
    # Step 2: Create output directories
    data_loader_logger.info(f"Creating output directory structure: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    data_loader_logger.info(f"Output directories created successfully")
    
    # Step 3: Extract and merge images from train and test folders
    data_loader_logger.info("Extracting and merging images from train and test sets")
    
    train_images_src = download_path / "data" / "images" / "train"
    test_images_src = download_path / "data" / "images" / "test"
    
    total_images_moved = 0
    
    # Move training images
    if train_images_src.exists():
        train_images = list(train_images_src.glob('*'))
        data_loader_logger.info(f"Found {len(train_images)} training images")
        
        for img_file in train_images:
            if img_file.is_file():
                target = images_dir / img_file.name
                shutil.copy2(str(img_file), str(target))
                total_images_moved += 1
        
        data_loader_logger.info(f"Successfully copied {total_images_moved} training images")
    else:
        data_loader_logger.warning(f"Training images folder not found: {train_images_src}")
    
    # Move test images
    if test_images_src.exists():
        test_images = list(test_images_src.glob('*'))
        data_loader_logger.info(f"Found {len(test_images)} test images")
        
        for img_file in test_images:
            if img_file.is_file():
                target = images_dir / img_file.name
                # Check for duplicates (same filename)
                if not target.exists():
                    shutil.copy2(str(img_file), str(target))
                    total_images_moved += 1
                else:
                    data_loader_logger.debug(f"Image already exists (duplicate): {img_file.name}")
        
        data_loader_logger.info(f"Successfully copied {len(test_images)} test images (duplicates skipped)")
    else:
        data_loader_logger.warning(f"Test images folder not found: {test_images_src}")
    
    data_loader_logger.info(f"Total images organized: {total_images_moved}")
    
    # Step 4: Move CSV files
    data_loader_logger.info("Organizing CSV data files")
    
    src_train_csv = download_path / "data" / "train.csv"
    src_test_csv = download_path / "data" / "test.csv"
    
    dst_train_csv = output_dir / "train.csv"
    dst_test_csv = output_dir / "test.csv"
    
    if src_train_csv.exists():
        shutil.copy2(str(src_train_csv), str(dst_train_csv))
        data_loader_logger.info(f"Copied train.csv successfully")
    else:
        data_loader_logger.warning(f"Training CSV not found: {src_train_csv}")
    
    if src_test_csv.exists():
        shutil.copy2(str(src_test_csv), str(dst_test_csv))
        data_loader_logger.info(f"Copied test.csv successfully")
    else:
        data_loader_logger.warning(f"Test CSV not found: {src_test_csv}")
    
    # Step 5: Verify organized structure
    data_loader_logger.info("Verifying organized dataset structure")
    
    images_count = len(list(images_dir.glob('*.jpg'))) + len(list(images_dir.glob('*.png')))
    train_csv_exists = dst_train_csv.exists()
    test_csv_exists = dst_test_csv.exists()
    
    data_loader_logger.info(f"Verification results:")
    data_loader_logger.info(f"  Images in output directory: {images_count}")
    data_loader_logger.info(f"  train.csv exists: {train_csv_exists}")
    data_loader_logger.info(f"  test.csv exists: {test_csv_exists}")
    
    # Step 6: Clean up temporary downloaded folder
    data_loader_logger.info(f"Cleaning up temporary download folder: {download_path}")
    try:
        if download_path.exists():
            shutil.rmtree(download_path)
            data_loader_logger.info("Temporary folder removed successfully")
    except Exception as e:
        data_loader_logger.warning(f"Could not remove temporary folder: {e}")
    
    data_loader_logger.info("=" * 80)
    data_loader_logger.info("VIVQA dataset download and organization completed successfully")
    data_loader_logger.info(f"Output directory: {output_dir}")
    data_loader_logger.info("=" * 80)
    
    return str(output_dir)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Download and organize datasets from Kaggle",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download and organize VIVQA dataset
  python -m src.data.download_data --dataset vivqa
  
  # Download and organize VQA dataset
  python -m src.data.download_data --dataset vqa
  
  # Download VIVQA with custom output directory
  python -m src.data.download_data --dataset vivqa --output-dir /path/to/vivqa
        """
    )
    
    parser.add_argument(
        "--dataset",
        type=str,
        default="vivqa",
        choices=["vqa", "vivqa"],
        help="Dataset to download and organize"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/vivqa",
        help="Output directory for organized data (only for VIVQA)"
    )
    
    parser.add_argument(
        "--kaggle-dataset",
        type=str,
        default="dngtrungngha/vivqa",
        help="Kaggle dataset identifier (format: username/dataset_name, only for VIVQA)"
    )
    
    args = parser.parse_args()
    
    try:
        if args.dataset == "vivqa":
            output_path = download_and_organize_vivqa_data(
                kaggle_dataset=args.kaggle_dataset,
                output_dir=args.output_dir
            )
            print(f"\nVIVQA dataset successfully organized at: {output_path}")
        elif args.dataset == "vqa":
            download_and_organize_vqa_data()
            print("\nVQA dataset successfully organized")
    except Exception as e:
        print(f"\nError during dataset download and organization: {e}")
        raise