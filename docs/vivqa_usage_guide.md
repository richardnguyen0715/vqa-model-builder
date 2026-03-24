# VIVQA Dataset Usage Guide

This document provides comprehensive instructions for working with the Vietnamese VQA (VIVQA) dataset in this project.

## Overview

The VIVQA dataset is a Vietnamese visual question answering dataset with:
- **Train set**: Multiple samples with questions, answers, and COCO image IDs
- **Test set**: Evaluation samples with the same structure
- **Images**: COCO dataset images referenced by image IDs

Dataset statistics:
- Questions and answers in Vietnamese
- Question types: Color-based visual questions (type 2)
- Images from MSCOCO dataset
- Single answers per question (not multiple choices)

## File Structure

```
project/
├── src/data/
│   ├── download_coco_images.py      # Script to download COCO images
│   ├── vivqa_dataset.py             # VIVQA dataset class
│
├── src/core/
│   ├── vivqa_evaluation_pipeline.py # Evaluation pipeline
│   ├── vivqa_eval_cli.py            # CLI entry point
│
├── configs/
│   ├── vivqa_data_configs.yaml      # Data configuration
│   ├── vivqa_evaluation_configs.yaml # Evaluation configuration
│
└── data/vivqa/
    ├── train.csv       # Training data
    ├── test.csv        # Test data
    └── images/         # Downloaded COCO images
```

## Step 1: Download COCO Images

### Option A: Using Python Script

```python
from src.data.download_coco_images import download_vivqa_images

# Download all images (train + test)
stats = download_vivqa_images(
    csv_dir="data/vivqa",
    output_dir="data/vivqa/images",
    download_train=True,
    download_test=True
)

# Print statistics
print(f"Downloaded: {stats['successful_downloads']}")
print(f"Failed: {stats['failed_downloads']}")
print(f"Total in directory: {stats['total_in_directory']}")
```

### Option B: Using Command Line

```bash
# Download all images
python -m src.data.download_coco_images

# Download only training images
python -m src.data.download_coco_images --skip-test

# Download only test images
python -m src.data.download_coco_images --skip-train

# Custom output directory
python -m src.data.download_coco_images --output-dir /path/to/images
```

### Download Notes

- Images are downloaded from MSCOCO server using standard COCO format: `COCO_train2014_000000{img_id:12d}.jpg`
- The downloader automatically retries failed downloads (default 3 retries)
- Already downloaded images are skipped automatically
- Download progress is shown with progress bar
- Detailed logging of successes and failures

## Step 2: Load VIVQA Dataset

### Basic Dataset Loading

```python
from transformers import AutoTokenizer
from src.data.vivqa_dataset import VivqaDataset, create_vivqa_dataloader

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained('vinai/phobert-base')

# Create dataset
dataset = VivqaDataset(
    csv_path='data/vivqa/test.csv',
    images_dir='data/vivqa/images',
    tokenizer=tokenizer,
    mode='inference',
    max_question_length=64,
    max_answer_length=64
)

print(f"Dataset size: {len(dataset)}")

# Get single sample
sample = dataset[0]
print(f"Question: {sample['question']}")
print(f"Answer: {sample['answer']}")
print(f"Image shape: {sample['image'].shape}")
```

### Create DataLoader

```python
# Create dataloader for batch processing
dataloader = create_vivqa_dataloader(
    dataset=dataset,
    batch_size=32,
    num_workers=4,
    shuffle=False,
    pin_memory=True
)

# Iterate through batches
for batch in dataloader:
    images = batch['image']              # [batch_size, 3, 224, 224]
    input_ids = batch['input_ids']       # [batch_size, seq_len]
    attention_mask = batch['attention_mask']  # [batch_size, seq_len]
    
    # ... process batch
    break
```

## Step 3: Run Model Evaluation

### Using CLI (Recommended)

```bash
# Basic evaluation
python -m src.core.vivqa_eval_cli \
    --checkpoint checkpoints/model.pt

# With custom batch size
python -m src.core.vivqa_eval_cli \
    --checkpoint checkpoints/model.pt \
    --batch-size 64

# With beam search generation
python -m src.core.vivqa_eval_cli \
    --checkpoint checkpoints/model.pt \
    --num-beams 3

# With sampling
python -m src.core.vivqa_eval_cli \
    --checkpoint checkpoints/model.pt \
    --do-sample \
    --temperature 0.7 \
    --top-p 0.9

# Custom output directory
python -m src.core.vivqa_eval_cli \
    --checkpoint checkpoints/model.pt \
    --output-dir results/my_eval/

# CPU evaluation
python -m src.core.vivqa_eval_cli \
    --checkpoint checkpoints/model.pt \
    --device cpu
```

### Using Python API

```python
import torch
from transformers import AutoTokenizer
from src.modeling.meta_arch.generative_vqa_model import GenerativeVQAModel
from src.core.vivqa_evaluation_pipeline import VivqaEvaluationPipeline, EvaluationConfig

# Load model
checkpoint = torch.load('checkpoints/model.pt')
model = GenerativeVQAModel()
model.load_state_dict(checkpoint['model_state_dict'])

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained('vinai/phobert-base')

# Create config
config = EvaluationConfig(
    csv_path='data/vivqa/test.csv',
    images_dir='data/vivqa/images',
    batch_size=32,
    checkpoint_path='checkpoints/model.pt',
    max_generate_length=64,
    output_dir='outputs/evaluation'
)

# Run evaluation
pipeline = VivqaEvaluationPipeline(
    model=model,
    tokenizer=tokenizer,
    config=config
)

results = pipeline.evaluate()

# Access results
print(f"Accuracy: {results.metrics['accuracy']:.4f}")
print(f"BLEU: {results.metrics['bleu']:.4f}")
print(f"ROUGE-L: {results.metrics['rouge_l']:.4f}")
print(f"METEOR: {results.metrics['meteor']:.4f}")
print(f"CIDEr: {results.metrics['cider']:.4f}")
print(f"F1: {results.metrics['f1']:.4f}")
```

## Evaluation Metrics

The evaluation pipeline computes the following metrics:

### 1. **Accuracy (Exact Match)**
- Percentage of predictions matching reference answers exactly
- Normalized comparison (lowercased, punctuation removed)

### 2. **Precision / Recall / F1**
- Token-level precision: Fraction of predicted tokens present in reference
- Token-level recall: Fraction of reference tokens present in prediction
- F1 Score: Harmonic mean of precision and recall

### 3. **BLEU Score**
- Bilingual Evaluation Understudy
- Measures n-gram overlap between prediction and reference
- Score range: 0-1 (higher is better)

### 4. **ROUGE-L (Recall-Oriented Understudy for Gisting Evaluation)**
- Longest common subsequence F-measure
- Measures longest matching sequence between prediction and reference
- Score range: 0-1 (higher is better)

### 5. **METEOR (Metric for Evaluation of Translation with Explicit ORdering)**
- Based on explicit word matches (exact, stem, synonym)
- Handles synonyms and inflections better than BLEU
- Score range: 0-1 (higher is better)

### 6. **CIDEr (Consensus-based Image Description Evaluation)**
- Uses TF-IDF weighting of n-grams
- Emphasizes rare, informative n-grams
- Score range: 0-10 (higher is better)

## Output Files

After evaluation, the following files are generated in `output_dir`:

### 1. **metrics_{timestamp}.json**
```json
{
  "timestamp": "2024-03-24T10:30:45.123456",
  "total_samples": 2993,
  "evaluation_time_seconds": 245.67,
  "metrics": {
    "accuracy": 0.8234,
    "precision": 0.7921,
    "recall": 0.8104,
    "f1": 0.8011,
    "bleu": 0.6543,
    "rouge_l": 0.7234,
    "meteor": 0.6789,
    "cider": 5.4321
  },
  "config": {
    "csv_path": "data/vivqa/test.csv",
    "batch_size": 32,
    "max_generate_length": 64
  }
}
```

### 2. **predictions_{timestamp}.json**
```json
[
  {
    "question": "màu của áo là gì",
    "prediction": "màu xanh dương",
    "reference": "màu xanh dương",
    "img_id": 44135
  },
  {
    "question": "màu của chiếc bình là gì",
    "prediction": "màu xanh lá",
    "reference": "màu xanh lá",
    "img_id": 68857
  },
  ...
]
```

## Configuration Files

### vivqa_data_configs.yaml
Configures dataset loading and image handling:
- Image dimensions and preprocessing
- Tokenization parameters
- Data loader settings

### vivqa_evaluation_configs.yaml
Configures evaluation pipeline:
- CSV paths and image directory
- Generation parameters (beam search, sampling)
- Metrics to compute
- Output configuration

## Advanced Usage

### Custom Data Preprocessing

```python
from torchvision import transforms
from src.data.vivqa_dataset import VivqaDataset

# Custom image transforms
custom_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

dataset = VivqaDataset(
    csv_path='data/vivqa/test.csv',
    images_dir='data/vivqa/images',
    tokenizer=tokenizer,
    transform=custom_transform
)
```

### Different Tokenizers

```python
from transformers import AutoTokenizer

# PhoBERT (recommended for Vietnamese)
tokenizer = AutoTokenizer.from_pretrained('vinai/phobert-base')

# Or other Vietnamese-compatible tokenizers
tokenizer = AutoTokenizer.from_pretrained('vinai/phobert-large')

dataset = VivqaDataset(
    csv_path='data/vivqa/test.csv',
    images_dir='data/vivqa/images',
    tokenizer=tokenizer
)
```

### Partial Dataset Evaluation

```python
from torch.utils.data import Subset

# Evaluate only first 100 samples
subset_dataset = Subset(dataset, range(100))

# Create dataloader from subset
from src.data.vivqa_dataset import create_vivqa_dataloader
dataloader = create_vivqa_dataloader(
    dataset=subset_dataset,
    batch_size=32
)

# Run evaluation as normal
results = pipeline.evaluate()
```

## Troubleshooting

### Issue: "Image not found" warnings

**Cause**: COCO images not downloaded yet.

**Solution**: 
```bash
python -m src.data.download_coco_images
```

### Issue: Out of Memory (OOM) errors

**Solutions**:
- Reduce batch size: `--batch-size 16`
- Use CPU: `--device cpu`
- Reduce max generation length: `--max-length 32`

### Issue: Tokenizer not found

**Cause**: PhoBERT model not downloaded.

**Solution**: Download manually or specify alternative tokenizer in code.

```bash
# Download PhoBERT
python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('vinai/phobert-base')"
```

### Issue: Model checkpoint format mismatch

**Cause**: Checkpoint saved in incompatible format.

**Solution**: Ensure checkpoint contains one of:
- `model_state_dict` key with state dict
- Direct state dict dictionary
- Model instance

## Performance Tips

1. **Use GPU**: Evaluation is significantly faster on GPU
   ```bash
   python -m src.core.vivqa_eval_cli --checkpoint model.pt --device cuda
   ```

2. **Increase batch size** (if GPU memory allows):
   ```bash
   python -m src.core.vivqa_eval_cli --checkpoint model.pt --batch-size 64
   ```

3. **Use multiple workers** for data loading:
   ```bash
   # Controlled via config (default 4 workers)
   ```

4. **Disable GPU for small datasets**:
   ```bash
   python -m src.core.vivqa_eval_cli --checkpoint model.pt --device cpu
   ```

## Citation

If using the VIVQA dataset, please cite:
```
[Include VIVQA paper citation here]
```

## References

- VIVQA Dataset: [Link to dataset paper/repository]
- MSCOCO Dataset: https://cocodataset.org/
- PhoBERT: https://github.com/VinAIResearch/PhoBERT
- Transformers: https://huggingface.co/transformers/

## Support

For issues or questions, please:
1. Check the Troubleshooting section above
2. Review configuration files in `configs/`
3. Check logs in `logs/` directory
4. Review detailed function docstrings in source files
