import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import cv2
import numpy as np
from collections import Counter
from torchvision import transforms
from typing import List, Union, Dict
from tqdm import tqdm

from src.schema.data_schema import OneSample
from torch.utils.data import DataLoader
from src.middleware.logger import data_process_logger
from src.exception.data_exception_handling import MemoryOverflowException
from src.middleware.monitor import memory_monitor


class VQADataset(Dataset):
    def __init__(self, data_list: List[Union[OneSample, Dict]], img_dir: str, tokenizer, answer2id: Dict, transform=None, mode='train', enable_memory_monitor=True):
        """
        Args:
            data_list: List các mẫu dữ liệu (OneSample hoặc dict từ file json)
            img_dir: Đường dẫn thư mục chứa ảnh (dùng khi data_list là dict chứa filename)
            tokenizer: Tokenizer
            answer2id: Dict map câu trả lời sang số (VD: {'yes': 0})
            transform: Hàm xử lý ảnh (Resize, Normalize)
            mode: 'train' hoặc 'val' (nếu val thì không cần tính target chính xác)
            enable_memory_monitor: Whether to monitor memory usage during data loading
        """
        self.data = data_list
        self.img_dir = img_dir
        self.tokenizer = tokenizer
        self.answer2id = answer2id
        self.transform = transform
        self.mode = mode
        self.enable_memory_monitor = enable_memory_monitor
        
        global memory_monitor
        
        # Initialize memory monitor if enabled
        if self.enable_memory_monitor:
            self.memory_monitor = memory_monitor
        else:
            self.memory_monitor = None

        # Định nghĩa transform mặc định nếu người dùng không truyền vào
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)), # Resize chuẩn cho ResNet/ViT
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], # Chuẩn ImageNet
                                     std=[0.229, 0.224, 0.225])
            ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Check memory periodically during data loading
        if self.memory_monitor is not None and idx % 100 == 0:
            try:
                self.memory_monitor.check_memory_usage(f"Loading batch item {idx}")
            except MemoryOverflowException as e:
                data_process_logger.critical(f"Memory overflow at batch item {idx}: {e}")
                raise
        
        # --- 1. XỬ LÝ ẢNH ---
        image = None
        
        if isinstance(item, OneSample):
            # OneSample now stores image_path instead of loaded image for memory efficiency
            # Load image on-the-fly from path
            img_path = item.image_path
            try:
                # Load image using PIL (RGB format)
                image = Image.open(img_path).convert('RGB')
            except Exception as e:
                data_process_logger.warning(f"Failed to load image {img_path}: {e}")
                # Create black placeholder image
                image = Image.new('RGB', (224, 224), (0, 0, 0))
        else:
            # Nếu là dict (từ json cũ)
            img_filename = item.get('image')
            
            # Xử lý đường dẫn ảnh
            if os.path.isabs(img_filename):
                img_path = img_filename
            else:
                img_path = os.path.join(self.img_dir, img_filename)
            
            try:
                # Dùng load_image từ data_utils cho thống nhất, nhưng load_image trả về numpy BGR
                # Hoặc dùng PIL trực tiếp như cũ để tiện
                # Ở đây ta dùng PIL trực tiếp để giữ logic cũ cho trường hợp này, 
                # hoặc dùng load_image rồi convert. Để an toàn và đơn giản cho luồng cũ:
                image = Image.open(img_path).convert('RGB')
            except:
                # Tạo ảnh đen nếu không tìm thấy file (tránh crash code training)
                image = Image.new('RGB', (224, 224), (0, 0, 0))
            
        if self.transform:
            image = self.transform(image) # Output: Tensor [3, 224, 224]

        # --- 2. XỬ LÝ CÂU HỎI ---
        if isinstance(item, OneSample):
            question = item.question
        else:
            question = item['question']
            
        # Tokenizer trả về dict {'input_ids': ..., 'attention_mask': ...}
        tokenized = self.tokenizer(question) 

        # --- 3. XỬ LÝ CÂU TRẢ LỜI (LABEL) ---
        label_tensor = torch.tensor(0) # Mặc định
        
        if self.mode == 'train':
            if isinstance(item, OneSample):
                answers = item.answers
            else:
                answers = item['answers'] # List 5 câu trả lời
            
            # Chiến thuật: Lấy câu trả lời xuất hiện nhiều nhất (Majority Vote)
            if answers:
                most_common_answer = Counter(answers).most_common(1)[0][0]
                
                # Map sang ID
                if most_common_answer in self.answer2id:
                    label_id = self.answer2id[most_common_answer]
                else:
                    label_id = self.answer2id.get('<unk>', 0) # Nếu không có trong top answers
            else:
                 label_id = self.answer2id.get('<unk>', 0)

            # Dùng LongTensor cho CrossEntropyLoss
            label_tensor = torch.tensor(label_id, dtype=torch.long) 
            
        # Get ground truth answer text for evaluation display
        ground_truth_answer = ""
        if self.mode == 'train' or self.mode == 'val' or self.mode == 'test':
            if isinstance(item, OneSample):
                answers = item.answers
            else:
                answers = item.get('answers', [])
            if answers:
                ground_truth_answer = Counter(answers).most_common(1)[0][0]
            
        return {
            'image': image,
            'input_ids': tokenized['input_ids'],
            'attention_mask': tokenized['attention_mask'],
            'label': label_tensor,
            'question': question,
            'ground_truth': ground_truth_answer
        }

# --- Helper Function: Xây dựng từ điển câu trả lời ---
def build_answer_vocab(data_list: List[Union[OneSample, Dict]], min_freq=5):
    """
    Chỉ giữ lại các câu trả lời xuất hiện nhiều hơn min_freq lần.
    Các câu hiếm sẽ quy về <unk>.
    """
    print("Đang xây dựng từ điển Answer (Label Map)...")
    all_answers = []
    for item in tqdm(data_list, desc="Building answer vocabulary"):
        if isinstance(item, OneSample):
            answers = item.answers
        else:
            answers = item['answers']
            
        if answers:
            # Lấy câu phổ biến nhất của mẫu đó để đếm
            major_ans = Counter(answers).most_common(1)[0][0]
            all_answers.append(major_ans)
        
    # Đếm tần suất
    counter = Counter(all_answers)
    
    answer2id = {'<unk>': 0}
    idx = 1
    for ans, count in counter.items():
        if count >= min_freq:
            answer2id[ans] = idx
            idx += 1
            
    print(f"Số lượng Class (Output): {len(answer2id)}")
    return answer2id


def create_data_loaders(dataset, batch_size=32, num_workers=4, mode='train'):
    """
    Create DataLoaders for training, validation, and testing.
    
    Args:
        batch_size: Batch size for data loading
        num_workers: Number of workers for parallel data loading
        mode: Mode of the dataset ('train', 'val', 'test')
    Returns:
        DataLoader for the specified mode
    """
    dataloader_instance = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(mode == 'train'),
        num_workers=num_workers,
        pin_memory=True
    )
    
    data_process_logger.info(f"DataLoader created with batch_size={batch_size}, num_workers={num_workers}, mode={mode}")
    return dataloader_instance
