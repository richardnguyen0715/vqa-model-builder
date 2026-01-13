from utils.config_loader import load_config


data_config = load_config('configs/data_configs.yaml')
logger_config = load_config('configs/logging_configs.yaml')
model_config = load_config('configs/model_configs.yaml')
warn_exception_config = load_config('configs/warning_configs.yaml')


# Dataconfig parameters ---------------------------------------------
raw_data = data_config.get('data_path', {}).get('raw_data', {})
raw_images = data_config.get('data_path', {}).get('raw_images', {})
raw_texts = data_config.get('data_path', {}).get('raw_texts', {})
processed_data = data_config.get('data_path', {}).get('processed_data', {}).get('folder', '')
train_data_folder = data_config.get('data_path', {}).get('processed_data', {}).get('train_data', {}).get('folder', '')
val_data_folder = data_config.get('data_path', {}).get('processed_data', {}).get('val_data', {}).get('folder', '')
test_data_folder = data_config.get('data_path', {}).get('processed_data', {}).get('test_data', {}).get('folder', '')
train_images_folder = data_config.get('data_path', {}).get('processed_data', {}).get('train_data', {}).get('images_folder', '')
val_images_folder = data_config.get('data_path', {}).get('processed_data', {}).get('val_data', {}).get('images_folder', '')
test_images_folder = data_config.get('data_path', {}).get('processed_data', {}).get('test_data', {}).get('images_folder', '')
train_texts_folder = data_config.get('data_path', {}).get('processed_data', {}).get('train_data', {}).get('texts_folder', '')
val_texts_folder = data_config.get('data_path', {}).get('processed_data', {}).get('val_data', {}).get('texts_folder', '')
test_texts_folder = data_config.get('data_path', {}).get('processed_data', {}).get('test_data', {}).get('texts_folder', '')
kaggle_project = data_config.get('kaggle_setup', {}).get('kaggle_project', '')
kaggle_folder = data_config.get('kaggle_setup', {}).get('images_folder', '')
kaggle_texts_file = data_config.get('kaggle_setup', {}).get('text_file', '')

# Model config parameters ------------------------------------------

# 1. Data parameters
batch_size = model_config.get('data', {}).get('batch_size', 32)
num_workers = model_config.get('data', {}).get('num_workers', 4)
train_ratio = model_config.get('data', {}).get('train_ratio', 0.8)
val_ratio = model_config.get('data', {}).get('val_ratio', 0.1)
test_ratio = model_config.get('data', {}).get('test_ratio', 0.1)

image_size = model_config.get('data', {}).get('image_size', (224, 224))

# 2. Tokenizer parameters
tokenizer_model_name = model_config.get('tokenizer_config', {}).get('model_name', 'bert-base-uncased')
tokenizer_max_len = model_config.get('tokenizer_config', {}).get('max_len', 128)

# 3. Normalize parameters
coco_normalize = model_config.get('coco_normalize', {
    'mean': [0.485, 0.456, 0.406],
    'std': [0.229, 0.224, 0.225]
})

imagenet_normalize = model_config.get('imagenet_normalize', {
    'mean': [0.485, 0.456, 0.406],
    'std': [0.229, 0.224, 0.225]
})

# 4. Training parameters
learning_rate = model_config.get('training_config', {}).get('learning_rate', 1e-4)
num_epochs = model_config.get('training_config', {}).get('num_epochs', 10)
weight_decay = model_config.get('training_config', {}).get('weight_decay', 1e-5)
early_stopping_patience = model_config.get('training_config', {}).get('early_stopping_patience', 5)
gradient_clip_value = model_config.get('training_config', {}).get('gradient_clip_value', 1.0)
lr_scheduler_step_size = model_config.get('training_config', {}).get('lr_scheduler_step_size', 10)
lr_scheduler_gamma = model_config.get('training_config', {}).get('lr_scheduler_gamma', 0.1)

# 5. Model parameters
project_model_name = model_config.get('model_config', {}).get('model_name', 'resnet50')
pretrained_model_path = model_config.get('model_config', {}).get('pretrained_model_path', "")
embedding_dim = model_config.get('model_config', {}).get('embedding_dim', 300)
hidden_dim = model_config.get('model_config', {}).get('hidden_dim', 512)
num_layers = model_config.get('model_config', {}).get('num_layers', 2)
dropout = model_config.get('model_config', {}).get('dropout', 0.3)


# Exception handling parameters ------------------------------------

# 1. Memory monitor parameters
memory_max_ram_percent = warn_exception_config.get('memory_monitor', {}).get('max_ram_percent', 85)
memory_warn_threshold_percent = warn_exception_config.get('memory_monitor', {}).get('warn_threshold_percent', 70)

# 2. CPU and GPU monitor parameters
cpu_memory_check_interval = warn_exception_config.get('cpu_monitor', {}).get('cpu_memory_check_interval', 60)  # in seconds
cpu_warn_threshold_percent = warn_exception_config.get('cpu_monitor', {}).get('warn_threshold_percent', 70)
cpu_max_memory_percent = warn_exception_config.get('cpu_monitor', {}).get('max_cpu_memory_percent', 85)

gpu_memory_check_interval = warn_exception_config.get('gpu_monitor', {}).get('gpu_memory_check_interval', 60)  # in seconds
gpu_warn_threshold_percent = warn_exception_config.get('gpu_monitor', {}).get('warn_threshold_percent', 70)
gpu_max_memory_percent = warn_exception_config.get('gpu_monitor', {}).get('max_gpu_memory_percent', 85)