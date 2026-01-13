import torch
from transformers import AutoTokenizer


class PretrainedTokenizer:
    def __init__(self, model_name='bert-base-uncased', max_len=30):
        """
        Wrapper bọc quanh HuggingFace Tokenizer để dễ sử dụng cho VQA.
        """
        print(f"Loading Pre-trained Tokenizer: {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_len = max_len
        
        # Lấy kích thước vocab để khai báo cho Embedding layer của Model
        self.vocab_size = self.tokenizer.vocab_size

    def __call__(self, text):
        """
        Input: "What is this?"
        Output: Dict {'input_ids': ..., 'attention_mask': ...}
        """
        encoding = self.tokenizer(
            text,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors="pt"
        )
        # Squeeze để bỏ dimension batch ảo (vì DataLoader sẽ tự stack lại sau)
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0)
        }

    def decode(self, token_ids):
        """Chuyển ngược từ ID sang text (để debug)"""
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)
