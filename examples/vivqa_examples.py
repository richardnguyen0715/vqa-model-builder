"""
VIVQA Quick Start Examples
Quick reference guide with ready-to-run examples for VIVQA dataset.
"""

# Example 1: Download COCO Images
# ================================
# python -m src.data.download_coco_images


# Example 2: Load VIVQA Dataset
# ==============================

from transformers import AutoTokenizer
from src.data.vivqa_dataset import VivqaDataset

# Load PhoBERT tokenizer
tokenizer = AutoTokenizer.from_pretrained('vinai/phobert-base')

# Create dataset from test set
dataset = VivqaDataset(
    csv_path='data/vivqa/test.csv',
    images_dir='data/vivqa/images',
    tokenizer=tokenizer,
    mode='inference'
)

print(f"Loaded {len(dataset)} samples")

# Get first sample
sample = dataset[0]
print(f"Question: {sample['question']}")
print(f"Answer: {sample['answer']}")
print(f"Image ID: {sample['img_id']}")
print(f"Image shape: {sample['image'].shape}")


# Example 3: Iterate Through Dataset
# ====================================

from src.data.vivqa_dataset import create_vivqa_dataloader

# Create dataloader
dataloader = create_vivqa_dataloader(
    dataset=dataset,
    batch_size=32,
    num_workers=4,
    shuffle=False
)

# Iterate through batches
for batch_idx, batch in enumerate(dataloader):
    print(f"\nBatch {batch_idx}:")
    print(f"  Images shape: {batch['image'].shape}")
    print(f"  Questions: {len(batch['question'])} items")
    print(f"  Answers: {batch['answer']}")
    print(f"  Image IDs: {batch['img_id']}")
    
    # Only show first batch
    if batch_idx == 0:
        break


# Example 4: Evaluate Model
# ==========================
# Command line:
# python -m src.core.vivqa_eval_cli --checkpoint checkpoints/model.pt

# Or in Python:

import torch
from transformers import AutoTokenizer
from src.core.vivqa_evaluation_pipeline import VivqaEvaluationPipeline, EvaluationConfig

# Prepare model and tokenizer
# (Assumes checkpoint contains model state)
checkpoint = torch.load('checkpoints/model.pt')

# Create config
config = EvaluationConfig(
    csv_path='data/vivqa/test.csv',
    images_dir='data/vivqa/images',
    batch_size=32,
    checkpoint_path='checkpoints/model.pt',
    max_generate_length=64,
    output_dir='outputs/evaluation'
)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained('vinai/phobert-base')

# Note: Actual model loading requires importing the specific model class
# from src.modeling.meta_arch.generative_vqa_model


# Example 5: Custom Transforms
# ============================

from torchvision import transforms
from src.data.vivqa_dataset import VivqaDataset

# Custom preprocessing
custom_transform = transforms.Compose([
    transforms.Resize((224, 224)),      # Resize to 224x224
    transforms.ToTensor(),               # Convert to tensor
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],     # ImageNet mean
        std=[0.229, 0.224, 0.225]       # ImageNet std
    )
])

# Create dataset with custom transforms
dataset = VivqaDataset(
    csv_path='data/vivqa/test.csv',
    images_dir='data/vivqa/images',
    tokenizer=tokenizer,
    transform=custom_transform
)


# Example 6: Process All Datasets (Train + Test)
# ===============================================

for split in ['train', 'test']:
    csv_path = f'data/vivqa/{split}.csv'
    
    dataset = VivqaDataset(
        csv_path=csv_path,
        images_dir='data/vivqa/images',
        tokenizer=tokenizer,
        mode='inference'
    )
    
    print(f"\n{split.upper()} SET:")
    print(f"  Total samples: {len(dataset)}")
    
    sample = dataset[0]
    print(f"  Sample question: {sample['question']}")
    print(f"  Sample answer: {sample['answer']}")


# Example 7: Check Dataset Statistics
# ====================================

import pandas as pd

for split in ['train', 'test']:
    csv_path = f'data/vivqa/{split}.csv'
    df = pd.read_csv(csv_path, index_col=0)
    
    print(f"\n{split.upper()} SET STATISTICS:")
    print(f"  Total samples: {len(df)}")
    print(f"  Columns: {df.columns.tolist()}")
    print(f"  Question types: {df['type'].unique()}")
    print(f"  Sample question: {df['question'].iloc[0]}")
    print(f"  Sample answer: {df['answer'].iloc[0]}")


# Example 8: Filter Dataset by Type
# ==================================

import pandas as pd
from torch.utils.data import Subset

# Load all data
df = pd.read_csv('data/vivqa/test.csv', index_col=0)

# Get indices for type 2 questions (color)
type_2_indices = df[df['type'] == 2].index.tolist()

# Create subset
dataset = VivqaDataset(
    csv_path='data/vivqa/test.csv',
    images_dir='data/vivqa/images',
    tokenizer=tokenizer
)

type_2_subset = Subset(dataset, range(len(type_2_indices)))
print(f"Color questions (type 2): {len(type_2_indices)}")


# Example 9: Batch Processing Without DataLoader
# ===============================================

from torch.utils.data import DataLoader

dataset = VivqaDataset(
    csv_path='data/vivqa/test.csv',
    images_dir='data/vivqa/images',
    tokenizer=tokenizer
)

# Create dataloader
dataloader = create_vivqa_dataloader(
    dataset=dataset,
    batch_size=32,
    shuffle=False
)

# Process first batch
for batch in dataloader:
    # Process batch
    images = batch['image']
    questions_input = batch['input_ids']
    questions_mask = batch['attention_mask']
    
    print(f"Batch shapes:")
    print(f"  Images: {images.shape}")
    print(f"  Input IDs: {questions_input.shape}")
    print(f"  Attention mask: {questions_mask.shape}")
    
    break


# Example 10: Evaluation Commands
# ================================

"""
# Basic evaluation
python -m src.core.vivqa_eval_cli --checkpoint checkpoints/model.pt

# With specific batch size
python -m src.core.vivqa_eval_cli --checkpoint checkpoints/model.pt --batch-size 64

# With beam search (3 beams)
python -m src.core.vivqa_eval_cli --checkpoint checkpoints/model.pt --num-beams 3

# With sampling (temperature sampling)
python -m src.core.vivqa_eval_cli --checkpoint checkpoints/model.pt --do-sample --temperature 0.7

# With nucleus sampling (top-p)
python -m src.core.vivqa_eval_cli --checkpoint checkpoints/model.pt --do-sample --top-p 0.95

# Save to custom directory
python -m src.core.vivqa_eval_cli --checkpoint checkpoints/model.pt --output-dir results/my_eval

# CPU evaluation
python -m src.core.vivqa_eval_cli --checkpoint checkpoints/model.pt --device cpu

# Custom CSV path
python -m src.core.vivqa_eval_cli --checkpoint checkpoints/model.pt --csv-path data/vivqa/train.csv

# Combine multiple options
python -m src.core.vivqa_eval_cli \\
    --checkpoint checkpoints/model.pt \\
    --batch-size 64 \\
    --num-beams 3 \\
    --output-dir results/eval_v1 \\
    --device cuda
"""


# Example 11: Access Evaluation Results
# ======================================

import json
from pathlib import Path

# Read metrics JSON
metrics_file = Path('outputs/evaluation/metrics_20240324_103045.json')
if metrics_file.exists():
    with open(metrics_file, 'r') as f:
        metrics = json.load(f)
    
    print("Evaluation Metrics:")
    for metric_name, value in metrics['metrics'].items():
        print(f"  {metric_name}: {value:.4f}")

# Read predictions JSON
predictions_file = Path('outputs/evaluation/predictions_20240324_103045.json')
if predictions_file.exists():
    with open(predictions_file, 'r') as f:
        predictions = json.load(f)
    
    print(f"\nFirst 5 Predictions:")
    for i, pred in enumerate(predictions[:5]):
        print(f"\n{i+1}. Question: {pred['question']}")
        print(f"   Prediction: {pred['prediction']}")
        print(f"   Reference: {pred['reference']}")


# Example 12: Compare Multiple Checkpoints
# =========================================

"""
# Script to evaluate multiple models

checkpoints = [
    'checkpoints/model_v1.pt',
    'checkpoints/model_v2.pt',
    'checkpoints/model_best.pt'
]

for checkpoint in checkpoints:
    print(f"\\nEvaluating {checkpoint}...")
    # python -m src.core.vivqa_eval_cli --checkpoint {checkpoint}
"""


# Example 13: Data Exploration
# ============================

import pandas as pd
import matplotlib.pyplot as plt

# Load train and test data
train_df = pd.read_csv('data/vivqa/train.csv', index_col=0)
test_df = pd.read_csv('data/vivqa/test.csv', index_col=0)

print("VIVQA Dataset Overview:")
print(f"  Train samples: {len(train_df)}")
print(f"  Test samples: {len(test_df)}")
print(f"  Total: {len(train_df) + len(test_df)}")

print(f"\nTrain set sample:")
print(train_df[['question', 'answer', 'img_id', 'type']].head())

print(f"\nQuestion types distribution:")
print(train_df['type'].value_counts())

# Calculate answer statistics
train_df['answer_length'] = train_df['answer'].str.len()
train_df['question_length'] = train_df['question'].str.len()

print(f"\nAnswer length statistics:")
print(train_df['answer_length'].describe())

print(f"\nQuestion length statistics:")
print(train_df['question_length'].describe())


print("\n" + "="*60)
print("For detailed documentation, see docs/vivqa_usage_guide.md")
print("="*60)
