import torch
from transformers import AutoTokenizer
import json
from collections import Counter
from nltk.tokenize import word_tokenize


class FromScratchTokenizer:
    def __init__(self, max_len=30, min_freq=2):
        """
        Tự xây dựng từ điển từ dataset.
        min_freq: Từ nào xuất hiện ít hơn số này sẽ bị coi là <unk>
        """
        self.word2idx = {}
        self.idx2word = {}
        self.max_len = max_len
        self.min_freq = min_freq
        
        # Must have special tokens
        self.pad_token = "<pad>"
        self.start_token = "<sos>"
        self.end_token = "<eos>"
        self.unk_token = "<unk>"
        
        # Define special tokens
        self.special_tokens = [self.pad_token, self.start_token, self.end_token, self.unk_token]
        for idx, token in enumerate(self.special_tokens):
            self.word2idx[token] = idx
            self.idx2word[idx] = token
            
        self.vocab_size = len(self.word2idx)

    def build_vocab(self, text_list):
        """
        text_list: List chứa toàn bộ câu hỏi trong dataset (['What is this?', 'Is it a cat?'])
        """
        print("Building vocabulary from scratch...")
        counter = Counter()
        for sentence in text_list:
            # Tokenize đơn giản bằng split (hoặc dùng thư viện nltk nếu muốn tốt hơn)
            tokens = word_tokenize(sentence)
            counter.update(tokens)

        # Chỉ thêm từ xuất hiện đủ nhiều
        for word, count in counter.items():
            if count >= self.min_freq:
                idx = len(self.word2idx)
                self.word2idx[word] = idx
                self.idx2word[idx] = word
        
        self.vocab_size = len(self.word2idx)
        print(f"Vocab built! Size: {self.vocab_size} words.")

    def encode(self, text):
        """Chuyển text thành Tensor ID"""
        tokens = word_tokenize(text)
        
        # Thêm <sos> và <eos>
        token_ids = [self.word2idx[self.start_token]]
        
        for token in tokens:
            idx = self.word2idx.get(token, self.word2idx[self.unk_token])
            token_ids.append(idx)
            
        token_ids.append(self.word2idx[self.end_token])
        
        # Padding hoặc Truncating thủ công
        if len(token_ids) < self.max_len:
            # Padding
            padding_len = self.max_len - len(token_ids)
            token_ids += [self.word2idx[self.pad_token]] * padding_len
        else:
            # Truncating (Cắt bớt)
            token_ids = token_ids[:self.max_len]
            # Đảm bảo luôn có token kết thúc (tùy chọn)
            token_ids[-1] = self.word2idx[self.end_token] 

        return torch.tensor(token_ids, dtype=torch.long)

    def save_vocab(self, path):
        with open(path, 'w') as f:
            json.dump(self.word2idx, f)
            
    def load_vocab(self, path):
        with open(path, 'r') as f:
            self.word2idx = json.load(f)
        self.idx2word = {v: k for k, v in self.word2idx.items()}
        self.vocab_size = len(self.word2idx)