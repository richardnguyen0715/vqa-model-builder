# Documentation Index

Welcome to the AutoViVQA Model Builder documentation!

---

## 📖 Quick Navigation

| Guide | Description | Audience |
|-------|-------------|----------|
| [Getting Started](getting_started.md) | Install and run your first training | Beginners |
| [Pipeline Usage](pipeline_usage.md) | Complete pipeline documentation | All users |
| [Configuration Guide](configuration_guide.md) | All configuration options | Intermediate |
| [API Reference](api_reference.md) | Python API documentation | Developers |

---

## 📚 Documentation Structure

### 1. Getting Started
- [**Getting Started Guide**](getting_started.md) - Quick installation and first run
  - Prerequisites
  - Installation steps
  - First training run
  - Troubleshooting

### 2. Pipeline Documentation
- [**Pipeline Usage**](pipeline_usage.md) - Complete pipeline guide
  - Data Pipeline
  - Model Pipeline
  - Training Pipeline
  - CLI reference
  - YAML configuration

### 3. Configuration
- [**Configuration Guide**](configuration_guide.md) - All settings explained
  - Data configuration
  - Model configuration
  - Training configuration
  - Resource management
  - Environment variables

### 4. API Reference
- [**API Reference**](api_reference.md) - Python API documentation
  - VQAPipeline class
  - DataPipeline class
  - ModelPipeline class
  - TrainingPipeline class
  - Configuration classes
  - Utility functions

### 5. Architecture Documentation
- [**VQA Architecture**](vqa_architecture.md) - System design
  - Overall architecture
  - Component interactions
  - Data flow

- [**Fusion Approaches**](fusion_approaches.md) - Multimodal fusion
  - Concatenation fusion
  - Element-wise fusion
  - Cross-attention fusion
  - Q-Former integration

- [**MOE Approaches**](moe_approaches.md) - Mixture of Experts
  - Expert routing
  - Load balancing
  - Implementation details

- [**Knowledge Base**](knowledge_base_approaches.md) - RAG system
  - FAISS integration
  - ChromaDB support
  - Retrieval strategies

### 6. Component Documentation
- [**Image Representation**](image_representation_approaches.md) - Visual encoders
  - ViT (Vision Transformer)
  - CLIP visual encoder
  - ResNet backbones
  - Feature extraction

- [**Text Representation**](text_representation_approaches.md) - Text encoders
  - PhoBERT for Vietnamese
  - BERT multilingual
  - Tokenization strategies

- [**Data Preparation**](prepare_data.md) - Dataset handling
  - Dataset format
  - Preprocessing
  - Augmentation strategies

---

## 🎯 Learning Paths

### Path 1: First-Time User
1. [Getting Started](getting_started.md) → Install and verify
2. [Pipeline Usage](pipeline_usage.md) → Run first training
3. [Configuration Guide](configuration_guide.md) → Customize settings

### Path 2: Understanding the System
1. [VQA Architecture](vqa_architecture.md) → Overview
2. [Image Representation](image_representation_approaches.md) → Visual encoders
3. [Text Representation](text_representation_approaches.md) → Text encoders
4. [Fusion Approaches](fusion_approaches.md) → Multimodal fusion

### Path 3: Advanced Features
1. [MOE Approaches](moe_approaches.md) → Expert networks
2. [Knowledge Base](knowledge_base_approaches.md) → RAG system
3. [API Reference](api_reference.md) → Custom integrations

---

## 🔧 Quick Commands

### Training
```bash
# Quick start
bash src/cli/quick_start.sh

# CLI training
python -m src.core.vqa_pipeline --mode train --epochs 20

# Config file
python -m src.core.vqa_pipeline --config configs/pipeline_config.yaml
```

### Evaluation
```bash
python -m src.core.vqa_pipeline --mode evaluate --resume checkpoints/best_model.pt
```

### Inference
```bash
python -m src.core.vqa_pipeline --mode inference --resume checkpoints/best_model.pt
```

---

## 📊 Common Configurations

### For 8GB GPU
```yaml
data:
  batch_size: 8
training:
  use_amp: true
```

### For 24GB+ GPU
```yaml
data:
  batch_size: 64
training:
  use_amp: true
model:
  use_moe: true
```

### For CPU Only
```yaml
data:
  batch_size: 4
  pin_memory: false
training:
  use_amp: false
```

---

## ❓ FAQ

### How do I reduce memory usage?
- Reduce `batch_size`
- Enable `use_amp: true`
- Set `freeze_visual_encoder: true`

### How do I speed up training?
- Increase `num_workers`
- Enable `use_amp: true`
- Use GPU instead of CPU

### How do I resume training?
```bash
python -m src.core.vqa_pipeline --resume checkpoints/checkpoint_epoch_5.pt
```

### How do I use my own data?
See [Data Preparation](prepare_data.md) guide.

---

## 📞 Support

- **GitHub Issues**: Report bugs or request features
- **Email**: richardnguyen0715@gmail.com
- **Documentation**: This guide

---

## 📝 Document Updates

| Date | Document | Change |
|------|----------|--------|
| 2025-01 | All | Initial documentation |

---

[← Back to README](../README.md)
