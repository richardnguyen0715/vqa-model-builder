# VIVQA Quick Reference

Quick reference for using the VIVQA dataset and evaluation tools.

## Files Overview

| File | Purpose | Size |
|------|---------|------|
| `src/data/download_coco_images.py` | Download COCO images | 10.2 KB |
| `src/data/vivqa_dataset.py` | VIVQA dataset loader | 13.9 KB |
| `src/core/vivqa_evaluation_pipeline.py` | Evaluation pipeline | 18.0 KB |
| `src/core/vivqa_eval_cli.py` | CLI interface | 10.7 KB |
| `configs/vivqa_data_configs.yaml` | Data configuration | 1.2 KB |
| `configs/vivqa_evaluation_configs.yaml` | Evaluation configuration | 1.3 KB |

## Quick Commands

### 1. Download COCO Images
```bash
python -m src.data.download_coco_images
```

### 2. Run Evaluation
```bash
# Basic evaluation
python -m src.core.vivqa_eval_cli --checkpoint checkpoints/model.pt

# With custom settings
python -m src.core.vivqa_eval_cli \
    --checkpoint checkpoints/model.pt \
    --batch-size 64 \
    --num-beams 3 \
    --output-dir results/

# CPU mode
python -m src.core.vivqa_eval_cli --checkpoint checkpoints/model.pt --device cpu
```

## Code Examples

### Load Dataset
```python
from transformers import AutoTokenizer
from src.data.vivqa_dataset import VivqaDataset

tokenizer = AutoTokenizer.from_pretrained('vinai/phobert-base')
dataset = VivqaDataset(
    csv_path='data/vivqa/test.csv',
    images_dir='data/vivqa/images',
    tokenizer=tokenizer
)
```

### Create DataLoader
```python
from src.data.vivqa_dataset import create_vivqa_dataloader

dataloader = create_vivqa_dataloader(
    dataset=dataset,
    batch_size=32,
    num_workers=4
)
```

### Evaluate Model
```python
from src.core.vivqa_evaluation_pipeline import VivqaEvaluationPipeline, EvaluationConfig

config = EvaluationConfig(
    csv_path='data/vivqa/test.csv',
    images_dir='data/vivqa/images',
    checkpoint_path='checkpoints/model.pt'
)

pipeline = VivqaEvaluationPipeline(model=model, tokenizer=tokenizer, config=config)
results = pipeline.evaluate()

print(f"Accuracy: {results.metrics['accuracy']:.4f}")
print(f"F1: {results.metrics['f1']:.4f}")
print(f"BLEU: {results.metrics['bleu']:.4f}")
```

## Configuration

### vivqa_data_configs.yaml
- `vivqa_data_path`: Path configuration
- `vivqa_dataset_config`: Image size, token length, batch size
- `coco_download_config`: Download parameters

### vivqa_evaluation_configs.yaml
- `vivqa_evaluation_config`: All evaluation settings
- Includes data paths, generation parameters, metrics

## Metrics Computed

1. **Accuracy** - Exact match percentage
2. **Precision** - Token-level precision
3. **Recall** - Token-level recall
4. **F1** - Harmonic mean of precision/recall
5. **BLEU** - Bilingual Evaluation Understudy (0-1)
6. **ROUGE-L** - Recall Of GistUng Evaluation (0-1)
7. **METEOR** - Metric for Evaluation of Translation (0-1)
8. **CIDEr** - Consensus Image Description (0-10)

## Output Files

Results saved in `outputs/evaluation/`:

```
metrics_{timestamp}.json      # Computed metrics
predictions_{timestamp}.json  # Model predictions vs references
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Image not found | `python -m src.data.download_coco_images` |
| Out of Memory | Reduce `--batch-size` or use `--device cpu` |
| Tokenizer not found | Download PhoBERT: `python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('vinai/phobert-base')"` |

## Dataset Format

CSV Structure:
```csv
,question,answer,img_id,type
0,màu của áo là gì,màu xanh dương,44135,2
1,màu của chiếc bình là gì,màu xanh lá,68857,2
```

Fields:
- `question`: Vietnamese question text
- `answer`: Vietnamese answer text
- `img_id`: COCO image ID (used to locate image)
- `type`: Question type (typically 2 for color)

## Documentation

- **Full Guide**: `docs/vivqa_usage_guide.md`
- **Code Examples**: `examples/vivqa_examples.py`
- **Implementation Details**: `VIVQA_IMPLEMENTATION.md`

## Performance

- **Image Download**: 500-800ms per image (network dependent)
- **Dataset Loading**: 10-20ms per sample
- **GPU Evaluation** (T4): ~50-100 samples/second
- **CPU Evaluation**: ~5-10 samples/second

## CLI Arguments Reference

### Required
- `--checkpoint PATH` - Path to model checkpoint

### Data
- `--csv-path PATH` (default: `data/vivqa/test.csv`)
- `--images-dir PATH` (default: `data/vivqa/images`)
- `--batch-size INT` (default: 32)
- `--num-workers INT` (default: 4)

### Generation
- `--max-length INT` (default: 64)
- `--num-beams INT` (default: 1)
- `--do-sample` (use sampling)
- `--temperature FLOAT` (default: 1.0)
- `--top-p FLOAT` (default: 0.9)

### Device
- `--device [auto|cuda|cpu]` (default: auto)

### Output
- `--output-dir PATH` (default: `outputs/evaluation`)
- `--save-predictions` (default: True)
- `--save-metrics` (default: True)

### Logging
- `--num-samples INT` (default: 10)
- `--log-interval INT` (default: 10)

## Example Workflow

```bash
# 1. Download images
python -m src.data.download_coco_images

# 2. Run evaluation
python -m src.core.vivqa_eval_cli --checkpoint checkpoints/my_model.pt

# 3. View results
cat outputs/evaluation/metrics_*.json | python -m json.tool

# 4. Check predictions
cat outputs/evaluation/predictions_*.json | python -m json.tool | head -50
```

## Supported Tokenizers

- **Recommended**: `vinai/phobert-base` (Vietnamese-specific)
- **Alternative**: `vinai/phobert-large` (larger model)
- **Generic**: Any HuggingFace transformer tokenizer

```python
tokenizer = AutoTokenizer.from_pretrained('vinai/phobert-base')
```

## Environment

- **Python**: 3.8+
- **PyTorch**: 1.9+
- **Transformers**: 4.0+
- **CUDA**: 11.0+ (optional, for GPU)

## Next Steps

1. **Setup**: Follow `docs/vivqa_usage_guide.md`
2. **Explore**: Run examples in `examples/vivqa_examples.py`
3. **Evaluate**: Use `vivqa_eval_cli.py` with your model
4. **Analyze**: Review output metrics and predictions

## Support

- Detailed troubleshooting in `docs/vivqa_usage_guide.md`
- Code examples in `examples/vivqa_examples.py`
- Function docstrings in source files
- Configuration comments in YAML files

---

**Last Updated**: March 24, 2026
**Status**: Production Ready
