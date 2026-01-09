# Generative VQA Model

Vietnamese Visual Question Answering với kiến trúc Encoder-Decoder, sinh câu trả lời thực tế thay vì phân loại.

## 📋 Tổng quan

### So sánh với Classification VQA

| Đặc điểm | Classification VQA | Generative VQA |
|----------|-------------------|----------------|
| **Output** | Class ID (0-459) | Chuỗi tokens |
| **Vocabulary** | Giới hạn (min_freq) | Không giới hạn (64K tokens) |
| **Câu trả lời mới** | ❌ Không thể | ✅ Có thể |
| **Xử lý OOV** | `<unk>` | Sinh từng token |
| **Độ linh hoạt** | Thấp | Cao |
| **Training** | CrossEntropy (class) | LM Loss (tokens) |

### Kiến trúc

```
┌─────────────────────────────────────────┐
│              INPUT                      │
│   🖼️ Image + ❓ Question                 │
└───────────────┬─────────────────────────┘
                │
┌───────────────▼─────────────────────────┐
│         VISUAL ENCODER                  │
│    CLIP ViT → Visual Features           │
│    [CLS] + patches → (197, 768)         │
└───────────────┬─────────────────────────┘
                │
┌───────────────▼─────────────────────────┐
│       QUESTION ENCODER                  │
│    PhoBERT → Question Features          │
│    tokens → (seq_len, 768)              │
└───────────────┬─────────────────────────┘
                │
┌───────────────▼─────────────────────────┐
│      CROSS-MODAL FUSION                 │
│    V + Q → Fused Features               │
│    Cross-Attention + FFN                │
└───────────────┬─────────────────────────┘
                │
┌───────────────▼─────────────────────────┐
│      TRANSFORMER DECODER                │
│    Auto-regressive generation           │
│    6 layers × 8 heads                   │
└───────────────┬─────────────────────────┘
                │
┌───────────────▼─────────────────────────┐
│            OUTPUT                       │
│    📝 Generated Answer (tokens)         │
└─────────────────────────────────────────┘
```

## 🚀 Quick Start

### Training

```bash
# Basic training
python -m src.core.generative_vqa_pipeline --mode train

# With custom parameters
python -m src.core.generative_vqa_pipeline \
    --mode train \
    --images-dir data/raw/images \
    --text-file data/raw/texts/evaluate_60k_data_balanced_preprocessed.csv \
    --batch-size 16 \
    --epochs 20 \
    --learning-rate 5e-5 \
    --freeze-visual \
    --freeze-text
```

### Evaluation

```bash
python -m src.core.generative_vqa_pipeline \
    --mode evaluate \
    --resume checkpoints/generative/best_generative_model.pt
```

### Demo

```bash
python -m src.core.generative_vqa_pipeline \
    --mode demo \
    --resume checkpoints/generative/best_generative_model.pt
```

## 📁 File Structure

```
src/
├── core/
│   ├── generative_vqa_pipeline.py    # Main CLI pipeline
│   └── generative_training_pipeline.py  # Training logic
├── data/
│   └── generative_dataset.py         # Seq2Seq dataset
└── modeling/
    └── meta_arch/
        └── generative_vqa_model.py   # Model architecture
```

## ⚙️ Configuration

### Model Config

```python
from src.modeling.meta_arch.generative_vqa_model import get_default_generative_vqa_config

config = get_default_generative_vqa_config(
    visual_backbone='openai/clip-vit-base-patch32',
    text_encoder='vinai/phobert-base',
    vocab_size=64001,  # PhoBERT vocab
    freeze_visual_encoder=True,  # Freeze CLIP ViT
    freeze_question_encoder=True,  # Freeze PhoBERT
    num_decoder_layers=6,
    num_attention_heads=8,
    hidden_size=768,
    max_answer_length=64
)
```

### Training Config

```python
from src.core.generative_training_pipeline import GenerativeTrainingConfig

training_config = GenerativeTrainingConfig(
    num_epochs=20,
    learning_rate=5e-5,
    weight_decay=0.01,
    warmup_ratio=0.1,
    use_amp=True,  # Mixed precision
    gradient_accumulation_steps=2,
    early_stopping=True,
    patience=5,
    metric_for_best='bleu'
)
```

## 📊 Metrics

Generative VQA sử dụng các metric NLG:

| Metric | Mô tả |
|--------|-------|
| **BLEU-4** | N-gram precision với brevity penalty |
| **METEOR** | F1 dựa trên synonyms và stemming |
| **ROUGE-L** | Longest Common Subsequence |
| **CIDEr** | TF-IDF weighted n-gram similarity |
| **Exact Match** | Khớp chính xác sau normalize |
| **Perplexity** | exp(loss) - độ "ngạc nhiên" |

## 🔧 Programmatic Usage

### Create Model

```python
from src.modeling.meta_arch import (
    GenerativeVQAModel,
    get_default_generative_vqa_config,
    create_generative_vqa_model
)

# Get config
config = get_default_generative_vqa_config(
    vocab_size=64001,
    freeze_visual_encoder=True,
    freeze_question_encoder=True
)

# Create model
model = create_generative_vqa_model(config)
model.to('cuda')

# Count parameters
total = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total: {total:,}, Trainable: {trainable:,}")
```

### Create Dataset

```python
from src.data.generative_dataset import (
    GenerativeVQADataset,
    generative_vqa_collate_fn
)
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import pandas as pd

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained('vinai/phobert-base')

# Load data
df = pd.read_csv('data/raw/texts/evaluate_60k_data_balanced_preprocessed.csv')

# Create dataset
dataset = GenerativeVQADataset(
    df=df,
    images_dir='data/raw/images',
    tokenizer=tokenizer,
    max_question_length=64,
    max_answer_length=32
)

# Create dataloader
loader = DataLoader(
    dataset,
    batch_size=16,
    collate_fn=generative_vqa_collate_fn,
    shuffle=True
)
```

### Training Loop

```python
from torch.optim import AdamW

optimizer = AdamW(
    [p for p in model.parameters() if p.requires_grad],
    lr=5e-5
)

model.train()
for batch in loader:
    batch = {k: v.to('cuda') if isinstance(v, torch.Tensor) else v 
             for k, v in batch.items()}
    
    outputs = model(
        pixel_values=batch['image'],
        input_ids=batch['input_ids'],
        attention_mask=batch['attention_mask'],
        decoder_input_ids=batch['decoder_input_ids'],
        decoder_attention_mask=batch['decoder_attention_mask'],
        labels=batch['labels']
    )
    
    loss = outputs.loss
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

### Generation

```python
model.eval()
with torch.no_grad():
    generated_ids = model.generate(
        pixel_values=image.unsqueeze(0).to('cuda'),
        input_ids=question_ids.unsqueeze(0).to('cuda'),
        attention_mask=question_mask.unsqueeze(0).to('cuda'),
        max_length=64,
        num_beams=4,  # Beam search
        do_sample=False
    )
    
    answer = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    print(f"Generated: {answer}")
```

## 📈 Training Tips

1. **Freeze Encoders**: Bắt đầu với frozen CLIP và PhoBERT để decoder học trước
2. **Learning Rate**: Dùng 5e-5 với warmup 10%
3. **Batch Size**: 16-32 với gradient accumulation nếu GPU nhỏ
4. **Mixed Precision**: Luôn bật AMP để tiết kiệm memory
5. **Early Stopping**: Dùng BLEU làm metric
6. **Generation**: Beam search (num_beams=4) thường tốt hơn greedy

## 🐛 Common Issues

### Out of Memory
```bash
# Giảm batch size
--batch-size 8 --gradient-accumulation 4
```

### Empty Generation
- Model cần train lâu hơn
- Tăng temperature khi sampling
- Sử dụng beam search

### Slow Training
- Freeze encoders
- Enable mixed precision (--use-amp)
- Tăng num_workers

## 📚 References

- [CLIP: Learning Transferable Visual Models](https://arxiv.org/abs/2103.00020)
- [PhoBERT: Pre-trained Vietnamese Language Models](https://arxiv.org/abs/2003.00744)
- [VQA: Visual Question Answering](https://arxiv.org/abs/1505.00468)
