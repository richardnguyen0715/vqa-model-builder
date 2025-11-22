from src.data.data_actions import load_raw_data, validate_data, split_data
from src.middleware.logger import data_process_logger
from src.middleware.config import train_ratio, val_ratio, test_ratio, tokenizer_model_name, tokenizer_max_len, image_size, coco_normalize, batch_size, num_workers
from src.data.dataset import VQADataset, build_answer_vocab, create_data_loaders
from src.modeling.tokenizer.pre_trained_tokenizer import PretrainedTokenizer
from src.exception.data_exception_handling import MemoryOverflowException
from utils.path_management import RAW_TEXT_CSV, RAW_IMAGES_DIR

from torchvision import transforms
from src.middleware.monitor import memory_monitor

# --- 1. LOAD RAW DATA ---
try:
    memory_monitor.check_memory_usage("Before loading raw data")
    raw_data = load_raw_data(
        images_dir=RAW_IMAGES_DIR,
        text_file_path=RAW_TEXT_CSV
    )
    validate_data(raw_data)
    memory_monitor.check_memory_usage("After loading raw data")
    data_process_logger.info("Raw data loaded and validated successfully.")
except MemoryOverflowException as e:
    data_process_logger.critical(f"Memory overflow during data loading: {e}")
    memory_report = memory_monitor.get_memory_report()
    data_process_logger.critical(f"Memory report: {memory_report}")
    raise
except Exception as e:
    data_process_logger.error(f"Error loading raw data: {e}")
    raise

# --- 2. SPLIT DATA ---
try:
    memory_monitor.check_memory_usage("Before splitting data")
    train_data, val_data, test_data = split_data(raw_data, is_random=True, train_ratio=train_ratio, val_ratio=val_ratio)
    memory_monitor.check_memory_usage("After splitting data")
    data_process_logger.info("Data splitting completed successfully.")
except Exception as e:
    data_process_logger.error(f"Error splitting data: {e}")
    raise

# --- 3. BUILD ANSWER VOCABULARY ---
try:
    memory_monitor.check_memory_usage("Before building answer vocabulary")
    answer2id = build_answer_vocab(raw_data, min_freq=5)
    memory_monitor.check_memory_usage("After building answer vocabulary")
    data_process_logger.info(f"Answer vocabulary built with {len(answer2id)} classes.")
except Exception as e:
    data_process_logger.error(f"Error building answer vocabulary: {e}")
    raise

# --- 4. INITIALIZE TOKENIZER ---
try:
    memory_monitor.check_memory_usage("Before initializing tokenizer")
    tokenizer = PretrainedTokenizer(model_name=tokenizer_model_name, max_len=tokenizer_max_len)
    memory_monitor.check_memory_usage("After initializing tokenizer")
    data_process_logger.info(f"Tokenizer initialized (vocab_size: {tokenizer.vocab_size}).")
except Exception as e:
    data_process_logger.error(f"Error initializing tokenizer: {e}")
    raise

# --- 5. DEFINE IMAGE TRANSFORMS ---
image_transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=coco_normalize.mean,
                         std=coco_normalize.std)
])

# --- 6. CREATE VQA DATASETS ---
try:
    memory_monitor.check_memory_usage("Before creating datasets")
    train_dataset = VQADataset(
        data_list=train_data,
        img_dir=str(RAW_IMAGES_DIR),
        tokenizer=tokenizer,
        answer2id=answer2id,
        transform=image_transform,
        mode='train'
    )
    
    # val_dataset = VQADataset(
    #     data_list=val_data,
    #     img_dir=str(RAW_IMAGES_DIR),
    #     tokenizer=tokenizer,
    #     answer2id=answer2id,
    #     transform=image_transform,
    #     mode='val'
    # )
    
    # test_dataset = VQADataset(
    #     data_list=test_data,
    #     img_dir=str(RAW_IMAGES_DIR),
    #     tokenizer=tokenizer,
    #     answer2id=answer2id,
    #     transform=image_transform,
    #     mode='val'
    # )
    
    memory_monitor.check_memory_usage("After creating datasets")
    data_process_logger.info(f"Datasets created: train={len(train_dataset)}")
except Exception as e:
    data_process_logger.error(f"Error creating VQA datasets: {e}")
    raise

try:
    memory_monitor.check_memory_usage("Before creating data loaders")
    train_dataloader = create_data_loaders(train_dataset, batch_size=batch_size, num_workers=num_workers, mode='train')
    # val_dataloader = create_data_loaders(val_dataset, batch_size=batch_size, num_workers=num_workers, mode='val')
    # test_dataloader = create_data_loaders(test_dataset, batch_size=batch_size, num_workers=num_workers, mode='val')
    memory_monitor.check_memory_usage("After creating data loaders")
except Exception as e:
    data_process_logger.error(f"Error creating data loaders: {e}")
    raise    
