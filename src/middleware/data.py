from src.data.data_utils import load_raw_data
from src.middleware.config import data_config
from src.middleware.logger import data_process_logger


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

raw_data = load_raw_data(
    images_dir=data_config['data_path']['raw_images'],
    text_file_path=data_config['data_path']['raw_texts'] + 'evaluate_60k_data_balanced_preprocessed.csv'
)