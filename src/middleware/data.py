from src.data.data_utils import load_raw_data
from src.middleware.logger import data_process_logger
from src.data.data_utils import validate_data, split_data
from utils.path_management import RAW_TEXT_CSV, RAW_IMAGES_DIR

try:
    raw_data = load_raw_data(
        images_dir=RAW_IMAGES_DIR,
        text_file_path=RAW_TEXT_CSV
    )
    validate_data(raw_data)
    train_data, val_data, test_data = split_data(raw_data, is_random=True)
    data_process_logger.info("Data loading and splitting completed successfully.")
except Exception as e:
    data_process_logger.error(f"Error loading raw data: {e}")
    raise