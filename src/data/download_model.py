"""
Hugging Face Model Downloader

Script to download pre-trained models from Hugging Face Hub and save them locally.
Supports different model formats (torch, safetensors) and automatic format detection.

Usage:
    python -m src.data.download_model --model-id RichardNguyen/ViMoE-VQA
    python -m src.data.download_model --model-id MODEL_ID --output-dir path/to/save
    python -m src.data.download_model --help
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Optional

try:
    from huggingface_hub import snapshot_download, model_info, hf_hub_download
    from huggingface_hub import __version__ as hfhub_version
except ImportError:
    print("ERROR: huggingface_hub is not installed.")
    print("Install it with: pip install huggingface_hub")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class HuggingFaceModelDownloader:
    """Download and manage Hugging Face models."""
    
    def __init__(self, output_dir: str = "checkpoints", cache_dir: Optional[str] = None):
        """
        Initialize the downloader.
        
        Args:
            output_dir: Directory to save downloaded models
            cache_dir: Optional custom cache directory for Hugging Face
        """
        self.output_dir = Path(output_dir)
        self.cache_dir = cache_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Output directory: {self.output_dir.absolute()}")
        if cache_dir:
            logger.info(f"Custom cache directory: {cache_dir}")
    
    def get_model_info(self, model_id: str) -> dict:
        """
        Fetch model information from Hugging Face.
        
        Args:
            model_id: Model identifier (e.g., "RichardNguyen/ViMoE-VQA")
            
        Returns:
            Dictionary with model information
        """
        try:
            logger.info(f"Fetching model info for: {model_id}")
            info = model_info(model_id)
            
            return {
                "id": model_id,
                "tags": info.tags,
                "library_name": info.library_name,
                "downloads": info.downloads,
                "last_modified": str(info.last_modified)
            }
        except Exception as e:
            logger.error(f"Failed to fetch model info: {e}")
            raise
    
    def download_model(
        self,
        model_id: str,
        force_download: bool = False
    ) -> Path:
        """
        Download model from Hugging Face Hub.
        
        Args:
            model_id: Model identifier
            force_download: Force download even if already cached
            
        Returns:
            Path to downloaded model directory
        """
        try:
            logger.info(f"Starting download: {model_id}")
            
            # Create subdirectory for this model
            model_name = model_id.replace("/", "_")
            model_output_dir = self.output_dir / model_name
            
            # Download the entire model (all files)
            logger.info(f"Downloading to: {model_output_dir.absolute()}")
            
            local_dir = snapshot_download(
                model_id,
                local_dir=str(model_output_dir),
                cache_dir=self.cache_dir,
                force_download=force_download,
                resume_download=True
            )
            
            logger.info(f"✓ Download completed successfully: {local_dir}")
            
            # List downloaded files
            downloaded_files = list(Path(local_dir).rglob("*"))
            file_count = sum(1 for f in downloaded_files if f.is_file())
            logger.info(f"Total files downloaded: {file_count}")
            
            # Log important files
            important_files = [
                "pytorch_model.bin", "model.safetensors", 
                "config.json", "tokenizer_config.json", "tokenizer.json"
            ]
            found_files = []
            for filename in important_files:
                filepath = Path(local_dir) / filename
                if filepath.exists():
                    size_mb = filepath.stat().st_size / (1024 * 1024)
                    found_files.append(f"{filename} ({size_mb:.2f} MB)")
            
            if found_files:
                logger.info("Key files found:")
                for f in found_files:
                    logger.info(f"  - {f}")
            
            return Path(local_dir)
            
        except Exception as e:
            logger.error(f"Failed to download model: {e}")
            raise
    
    def download_with_fallback(
        self,
        model_id: str,
        force_download: bool = False
    ) -> Path:
        """
        Download model from Hugging Face Hub.
        The function automatically handles different formats available in the model.
        
        Args:
            model_id: Model identifier
            force_download: Force download even if already cached
            
        Returns:
            Path to downloaded model directory
        """
        try:
            logger.info(f"Starting model download for: {model_id}")
            
            return self.download_model(
                model_id,
                force_download=force_download
            )
                
        except Exception as e:
            logger.error(f"Download failed: {e}")
            raise
    
    def verify_model(self, model_dir: Path) -> bool:
        """
        Verify that the model was downloaded correctly.
        
        Args:
            model_dir: Path to model directory
            
        Returns:
            True if model is valid, False otherwise
        """
        try:
            logger.info(f"Verifying model in: {model_dir}")
            
            if not model_dir.exists():
                logger.error(f"Model directory does not exist: {model_dir}")
                return False
            
            # Check for required config file
            config_path = model_dir / "config.json"
            if not config_path.exists():
                logger.warning(f"config.json not found in {model_dir}")
                return False
            
            logger.info("✓ Model verification passed")
            return True
            
        except Exception as e:
            logger.error(f"Model verification failed: {e}")
            return False


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Download Hugging Face models to local directory",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download default model (RichardNguyen/ViMoE-VQA)
  python -m src.data.download_model

  # Download specific model
  python -m src.data.download_model --model-id RichardNguyen/ViMoE-VQA

  # Download to custom directory
  python -m src.data.download_model --output-dir ./my_models

  # Force re-download
  python -m src.data.download_model --force-download

  # Use custom cache directory
  python -m src.data.download_model --cache-dir ~/.cache/huggingface_custom
        """
    )
    
    parser.add_argument(
        "--model-id",
        type=str,
        default="RichardNguyen/ViMoE-VQA",
        help="Model ID from Hugging Face Hub (default: RichardNguyen/ViMoE-VQA)"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="checkpoints",
        help="Output directory to save models (default: checkpoints)"
    )
    
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Custom Hugging Face cache directory (optional)"
    )
    
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Force download even if already cached"
    )
    
    parser.add_argument(
        "--info-only",
        action="store_true",
        help="Only show model information, don't download"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug(f"Verbose logging enabled")
        logger.debug(f"huggingface_hub version: {hfhub_version}")
    
    try:
        downloader = HuggingFaceModelDownloader(
            output_dir=args.output_dir,
            cache_dir=args.cache_dir
        )
        
        logger.info("=" * 80)
        logger.info("Hugging Face Model Downloader")
        logger.info("=" * 80)
        logger.info(f"Model ID: {args.model_id}")
        logger.info(f"Output Directory: {args.output_dir}")
        logger.info("")
        
        # Fetch and display model info
        try:
            model_info_dict = downloader.get_model_info(args.model_id)
            logger.info("Model Information:")
            logger.info(f"  Tags: {', '.join(model_info_dict['tags']) if model_info_dict['tags'] else 'N/A'}")
            logger.info(f"  Library: {model_info_dict['library_name']}")
            logger.info(f"  Downloads: {model_info_dict['downloads']}")
            logger.info(f"  Last Modified: {model_info_dict['last_modified']}")
            logger.info("")
        except Exception as e:
            logger.warning(f"Could not fetch model info: {e}")
        
        if args.info_only:
            logger.info("Info-only mode. Skipping download.")
            return 0
        
        # Download model
        model_dir = downloader.download_with_fallback(
            args.model_id,
            force_download=args.force_download
        )
        
        # Verify model
        if downloader.verify_model(model_dir):
            logger.info("=" * 80)
            logger.info("Download completed successfully!")
            logger.info(f"Model location: {model_dir}")
            logger.info("=" * 80)
            return 0
        else:
            logger.error("Model verification failed!")
            return 1
            
    except KeyboardInterrupt:
        logger.info("\nDownload cancelled by user")
        return 130
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=args.verbose)
        return 1


if __name__ == "__main__":
    sys.exit(main())
