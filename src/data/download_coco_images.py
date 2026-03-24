"""
Download Images from MSCOCO Dataset
Downloads COCO images based on image IDs from VIVQA dataset.

Features:
- Batch downloads from COCO image server
- Automatic retry on failures
- Progress tracking with tqdm
- Logging of successful and failed downloads
- Duplicate image handling
"""

import os
import pandas as pd
from pathlib import Path
from typing import List, Set
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from tqdm import tqdm
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# COCO image URL formats - try multiple sources
COCO_IMAGE_URL_FORMATS = [
    "http://images.cocodataset.org/train2014/COCO_train2014_{img_id:012d}.jpg",
    "http://images.cocodataset.org/val2014/COCO_val2014_{img_id:012d}.jpg",
    "http://images.cocodataset.org/train2017/000000{img_id:06d}.jpg",
    "http://images.cocodataset.org/val2017/000000{img_id:06d}.jpg",
    "http://images.cocodataset.org/test2014/COCO_test2014_{img_id:012d}.jpg",
    "http://images.cocodataset.org/test2017/000000{img_id:06d}.jpg",
]


class COCOImageDownloader:
    """Download COCO images based on image IDs."""
    
    def __init__(
        self,
        output_dir: str = "data/vivqa/images",
        max_retries: int = 3,
        timeout: int = 30
    ):
        """
        Initialize COCO image downloader.
        
        Args:
            output_dir: Directory to save downloaded images
            max_retries: Maximum number of retries for failed downloads
            timeout: Request timeout in seconds
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_retries = max_retries
        self.timeout = timeout
        
        # Setup session with retry strategy
        self.session = self._create_session()
        
        logger.info(f"COCO image downloader initialized for output directory: {self.output_dir}")
    
    def _create_session(self) -> requests.Session:
        """Create requests session with retry strategy."""
        session = requests.Session()
        retry_strategy = Retry(
            total=self.max_retries,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS"]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        return session
    
    def _get_image_filename(self, img_id: int) -> str:
        """Get standard COCO filename for image ID."""
        return f"COCO_train2014_{img_id:012d}.jpg"
    
    def _get_image_url(self, img_id: int) -> str:
        """Get COCO image URL for given image ID. (Deprecated - use _get_image_urls instead)"""
        return COCO_IMAGE_URL_FORMATS[0].format(img_id=img_id)
    
    def _get_image_urls(self, img_id: int) -> List[str]:
        """Get list of COCO image URLs to try for given image ID."""
        urls = []
        for url_format in COCO_IMAGE_URL_FORMATS:
            try:
                url = url_format.format(img_id=img_id)
                urls.append(url)
            except (KeyError, ValueError):
                continue
        return urls
    
    def download_image(self, img_id: int) -> bool:
        """
        Download single image from COCO.
        
        Tries multiple COCO dataset sources (train2014, val2014, train2017, val2017, etc.).
        
        Args:
            img_id: COCO image ID
            
        Returns:
            True if successful, False otherwise
        """
        filename = self._get_image_filename(img_id)
        filepath = self.output_dir / filename
        
        # Skip if already exists
        if filepath.exists():
            return True
        
        # Try multiple COCO sources
        urls = self._get_image_urls(img_id)
        errors = []
        
        for url in urls:
            try:
                response = self.session.get(url, timeout=self.timeout)
                response.raise_for_status()
                
                # Save image
                with open(filepath, 'wb') as f:
                    f.write(response.content)
                
                return True
            
            except requests.RequestException as e:
                errors.append(f"{url}: {str(e)}")
                continue
        
        # All URLs failed
        logger.debug(f"Failed to download image {img_id} from any source. Errors: {errors}")
        return False
    
    def download_images(self, img_ids: List[int]) -> dict:
        """
        Download multiple images from COCO.
        
        Args:
            img_ids: List of COCO image IDs
            
        Returns:
            Dictionary with download statistics
        """
        logger.info(f"Starting download of {len(img_ids)} images")
        
        # Remove duplicates
        unique_img_ids = list(set(img_ids))
        logger.info(f"Downloading {len(unique_img_ids)} unique images (removed {len(img_ids) - len(unique_img_ids)} duplicates)")
        
        # Track results
        successful = 0
        failed = 0
        already_exist = 0
        failed_ids = []
        
        # Download with progress bar
        for img_id in tqdm(unique_img_ids, desc="Downloading COCO images"):
            filename = self._get_image_filename(img_id)
            filepath = self.output_dir / filename
            
            if filepath.exists():
                already_exist += 1
                continue
            
            if self.download_image(img_id):
                successful += 1
            else:
                failed += 1
                failed_ids.append(img_id)
        
        # Summary
        stats = {
            'total_requested': len(img_ids),
            'unique_requested': len(unique_img_ids),
            'successful_downloads': successful,
            'already_existing': already_exist,
            'failed_downloads': failed,
            'failed_img_ids': failed_ids,
            'total_in_directory': len(list(self.output_dir.glob('*.jpg')))
        }
        
        logger.info("Download statistics:")
        logger.info(f"  Total requested: {stats['total_requested']}")
        logger.info(f"  Unique requested: {stats['unique_requested']}")
        logger.info(f"  Already existing: {stats['already_existing']}")
        logger.info(f"  Successfully downloaded: {stats['successful_downloads']}")
        logger.info(f"  Failed downloads: {stats['failed_downloads']}")
        logger.info(f"  Total images in directory: {stats['total_in_directory']}")
        
        if failed_ids:
            logger.warning(f"Failed to download {len(failed_ids)} images: {failed_ids[:10]}{'...' if len(failed_ids) > 10 else ''}")
        
        return stats
    
    def download_from_csv(
        self,
        csv_path: str,
        img_id_column: str = 'img_id',
        download_train: bool = True,
        download_test: bool = True
    ) -> dict:
        """
        Download images from image IDs listed in CSV file(s).
        
        Args:
            csv_path: Path to directory containing train.csv and test.csv, or path to single CSV file
            img_id_column: Column name containing image IDs
            download_train: Whether to download train set images
            download_test: Whether to download test set images
            
        Returns:
            Dictionary with download statistics
        """
        logger.info(f"Loading image IDs from CSV: {csv_path}")
        
        csv_path = Path(csv_path)
        all_img_ids = []
        
        # Handle directory vs file
        if csv_path.is_dir():
            csv_files = []
            if download_train and (csv_path / 'train.csv').exists():
                csv_files.append(csv_path / 'train.csv')
            if download_test and (csv_path / 'test.csv').exists():
                csv_files.append(csv_path / 'test.csv')
        else:
            csv_files = [csv_path]
        
        # Load image IDs from all CSV files
        for csv_file in csv_files:
            logger.info(f"Reading image IDs from: {csv_file}")
            try:
                df = pd.read_csv(csv_file, index_col=0)
                img_ids = df[img_id_column].tolist()
                all_img_ids.extend(img_ids)
                logger.info(f"  Loaded {len(img_ids)} image IDs from {csv_file.name}")
            except Exception as e:
                logger.error(f"Failed to read CSV file {csv_file}: {e}")
        
        if not all_img_ids:
            logger.warning("No image IDs found in CSV files")
            return {
                'total_requested': 0,
                'successful_downloads': 0,
                'failed_downloads': 0
            }
        
        # Download images
        return self.download_images(all_img_ids)


def download_vivqa_images(
    csv_dir: str = "data/vivqa",
    output_dir: str = "data/vivqa/images",
    download_train: bool = True,
    download_test: bool = True
) -> dict:
    """
    Download all COCO images needed for VIVQA dataset.
    
    Args:
        csv_dir: Directory containing train.csv and test.csv
        output_dir: Directory to save images
        download_train: Whether to download train set images
        download_test: Whether to download test set images
        
    Returns:
        Dictionary with download statistics
    """
    downloader = COCOImageDownloader(output_dir=output_dir)
    
    stats = downloader.download_from_csv(
        csv_path=csv_dir,
        img_id_column='img_id',
        download_train=download_train,
        download_test=download_test
    )
    
    return stats


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Download COCO images for VIVQA dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download all images (train and test)
  python -m src.data.download_coco_images
  
  # Download only train images
  python -m src.data.download_coco_images --skip-test
  
  # Download only test images
  python -m src.data.download_coco_images --skip-train
  
  # Custom output directory
  python -m src.data.download_coco_images --output-dir /path/to/images
        """
    )
    
    parser.add_argument(
        "--csv-dir",
        type=str,
        default="data/vivqa",
        help="Directory containing train.csv and test.csv"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/vivqa/images",
        help="Directory to save downloaded images"
    )
    
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Skip downloading train set images"
    )
    
    parser.add_argument(
        "--skip-test",
        action="store_true",
        help="Skip downloading test set images"
    )
    
    args = parser.parse_args()
    
    stats = download_vivqa_images(
        csv_dir=args.csv_dir,
        output_dir=args.output_dir,
        download_train=not args.skip_train,
        download_test=not args.skip_test
    )
    
    logger.info("Download completed")
