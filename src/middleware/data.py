from src.data.data_actions import load_raw_data, load_data_split, validate_data, split_data
from src.middleware.logger import data_process_logger
from src.middleware.config import train_ratio, val_ratio, test_ratio, tokenizer_model_name, tokenizer_max_len, image_size, coco_normalize, batch_size, num_workers
from src.data.dataset import VQADataset, build_answer_vocab, create_data_loaders
from src.modeling.tokenizer.pre_trained_tokenizer import PretrainedTokenizer
from src.exception.data_exception_handling import MemoryOverflowException
from utils.path_management import RAW_TEXT_CSV, RAW_IMAGES_DIR

from torchvision import transforms
from src.middleware.monitor import memory_monitor

# --- LOAD DATA SPLITS SEPARATELY ---

def load_data_safely(mode):

    if mode == 'train':
        try:
            memory_monitor.check_memory_usage("Before loading train data")
            train_data = load_data_split(
                images_dir=RAW_IMAGES_DIR,
                text_file_path=RAW_TEXT_CSV,
                split_type='train',
                train_ratio=train_ratio,
                val_ratio=val_ratio
            )
            memory_monitor.check_memory_usage("After loading train data")
            data_process_logger.info(f"Train data loaded: {len(train_data)} samples")
        except MemoryOverflowException as e:
            data_process_logger.critical(f"Memory overflow during train data loading: {e}")
            memory_report = memory_monitor.get_memory_report()
            data_process_logger.critical(f"Memory report: {memory_report}")
            raise
        except Exception as e:
            data_process_logger.error(f"Error loading train data: {e}")
            raise
        
        return train_data

    elif mode == 'val':
        try:
            memory_monitor.check_memory_usage("Before loading val data")
            val_data = load_data_split(
                images_dir=RAW_IMAGES_DIR,
                text_file_path=RAW_TEXT_CSV,
                split_type='val',
                train_ratio=train_ratio,
                val_ratio=val_ratio
            )
            memory_monitor.check_memory_usage("After loading val data")
            data_process_logger.info(f"Val data loaded: {len(val_data)} samples")
        except MemoryOverflowException as e:
            data_process_logger.critical(f"Memory overflow during val data loading: {e}")
            memory_report = memory_monitor.get_memory_report()
            data_process_logger.critical(f"Memory report: {memory_report}")
            raise
        except Exception as e:
            data_process_logger.error(f"Error loading val data: {e}")
            raise
        
        return val_data
    
    elif mode == 'test':
        try:
            memory_monitor.check_memory_usage("Before loading test data")
            test_data = load_data_split(
                images_dir=RAW_IMAGES_DIR,
                text_file_path=RAW_TEXT_CSV,
                split_type='test',
                train_ratio=train_ratio,
                val_ratio=val_ratio
            )
            memory_monitor.check_memory_usage("After loading test data")
            data_process_logger.info(f"Test data loaded: {len(test_data)} samples")
        except MemoryOverflowException as e:
            data_process_logger.critical(f"Memory overflow during test data loading: {e}")
            memory_report = memory_monitor.get_memory_report()
            data_process_logger.critical(f"Memory report: {memory_report}")
            raise
        except Exception as e:
            data_process_logger.error(f"Error loading test data: {e}")
            raise
        
        return test_data
    
train_data = load_data_safely('train')

# --- BUILD ANSWER VOCABULARY FROM TRAIN DATA ONLY ---
try:
    memory_monitor.check_memory_usage("Before building answer vocabulary")
    answer2id = build_answer_vocab(train_data, min_freq=5)
    memory_monitor.check_memory_usage("After building answer vocabulary")
    data_process_logger.info(f"Answer vocabulary built with {len(answer2id)} classes.")
except Exception as e:
    data_process_logger.error(f"Error building answer vocabulary: {e}")
    raise

# --- INITIALIZE TOKENIZER ---
try:
    memory_monitor.check_memory_usage("Before initializing tokenizer")
    tokenizer = PretrainedTokenizer(model_name=tokenizer_model_name, max_len=tokenizer_max_len)
    memory_monitor.check_memory_usage("After initializing tokenizer")
    data_process_logger.info(f"Tokenizer initialized (vocab_size: {tokenizer.vocab_size}).")
except Exception as e:
    data_process_logger.error(f"Error initializing tokenizer: {e}")
    raise

# --- DEFINE IMAGE TRANSFORMS ---
image_transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=coco_normalize['mean'],
                         std=coco_normalize['std'])
])

# --- CREATE VQA DATASETS ---
try:
    memory_monitor.check_memory_usage("Before creating train dataset")
    train_dataset = VQADataset(
        data_list=train_data,
        img_dir=str(RAW_IMAGES_DIR),
        tokenizer=tokenizer,
        answer2id=answer2id,
        transform=image_transform,
        mode='train'
    )
    memory_monitor.check_memory_usage("After creating train dataset")
    data_process_logger.info(f"Train dataset created: {len(train_dataset)} samples")
except Exception as e:
    data_process_logger.error(f"Error creating train dataset: {e}")
    raise

# --- CREATE DATA LOADERS ---
try:
    memory_monitor.check_memory_usage("Before creating data loaders")
    train_dataloader = create_data_loaders(train_dataset, batch_size=batch_size, num_workers=num_workers, mode='train')
    memory_monitor.check_memory_usage("After creating data loaders")
    data_process_logger.info("All data loaders created successfully.")
except Exception as e:
    data_process_logger.error(f"Error creating data loaders: {e}")
    raise    
