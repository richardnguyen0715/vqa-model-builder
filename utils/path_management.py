from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]

RAW_TEXT_CSV = ROOT_DIR / 'data' / 'raw' / 'texts' / 'evaluate_60k_data_balanced_preprocessed.csv'
RAW_IMAGES_DIR = ROOT_DIR / 'data' / 'raw' / 'images'
PROCESSED_DATA_DIR = ROOT_DIR / 'data' / 'processed'