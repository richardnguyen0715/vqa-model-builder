import kagglehub
import shutil
from pathlib import Path
from src.middleware.logger import data_loader_logger
from src.middleware.config import data_config


def download_and_organize_data():
    """Download dataset from Kaggle and organize into project structure."""
    
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
        data_loader_logger.info(f"Text file moved successfully")
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
    
    data_loader_logger.info("Data download and organization completed successfully")


if __name__ == "__main__":
    download_and_organize_data()