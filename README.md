# 🇻🇳 AutoViVQA Model Builder

<div align="center">

```
╔══════════════════════════════════════════════════════════════════════════════════════════╗
║                                                                                          ║
║  ██╗   ██╗ ██████╗  █████╗     ██████╗ ██╗██████╗ ███████╗██╗     ██╗███╗   ██╗███████╗  ║
║  ██║   ██║██╔═══██╗██╔══██╗    ██╔══██╗██║██╔══██╗██╔════╝██║     ██║████╗  ██║██╔════╝  ║
║  ██║   ██║██║   ██║███████║    ██████╔╝██║██████╔╝█████╗  ██║     ██║██╔██╗ ██║█████╗    ║
║  ╚██╗ ██╔╝██║▄▄ ██║██╔══██║    ██╔═══╝ ██║██╔═══╝ ██╔══╝  ██║     ██║██║╚██╗██║██╔══╝    ║
║   ╚████╔╝ ╚██████╔╝██║  ██║    ██║     ██║██║     ███████╗███████╗██║██║ ╚████║███████╗  ║
║    ╚═══╝   ╚══▀▀═╝ ╚═╝  ╚═╝    ╚═╝     ╚═╝╚═╝     ╚══════╝╚══════╝╚═╝╚═╝  ╚═══╝╚══════╝  ║
║                                                                                          ║
║                            Vietnamese Visual Question Answering                          ║
║                                  AutoViVQA Model Builder                                 ║
║                                                                                          ║
╚══════════════════════════════════════════════════════════════════════════════════════════╝
```

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.9+-ee4c2c.svg)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**Hệ thống Visual Question Answering tiếng Việt toàn diện với kiến trúc tiên tiến**

[Tính năng](#-tính-năng) •
[Cài đặt](#-cài-đặt) •
[Bắt đầu nhanh](#-bắt-đầu-nhanh) •
[Pipeline Guide](#-pipeline-guide) •
[Tài liệu](#-tài-liệu-chi-tiết)

</div>

---

## 📋 Mục lục

- [Tính năng](#-tính-năng)
- [Kiến trúc tổng quan](#-kiến-trúc-tổng-quan)
- [Cài đặt](#-cài-đặt)
- [Bắt đầu nhanh](#-bắt-đầu-nhanh)
- [Pipeline Guide](#-pipeline-guide)
  - [1. Chuẩn bị dữ liệu (Data Processing)](#1-chuẩn-bị-dữ-liệu-data-processing)
  - [2. Huấn luyện (Training)](#2-huấn-luyện-training)
  - [3. Đánh giá (Evaluation)](#3-đánh-giá-evaluation)
  - [4. Suy luận (Inference)](#4-suy-luận-inference)
  - [5. Generative VQA Pipeline](#5-generative-vqa-pipeline)
  - [6. Ablation Study](#6-ablation-study)
    - [6.9 Chạy lại experiment cụ thể (`--rerun`)](#69-chạy-lại-experiment-cụ-thể---rerun)
    - [6.10 Graceful Interruption (Ctrl+C)](#610-graceful-interruption-ctrlc)
    - [6.11 Per-experiment Log Files](#611-per-experiment-log-files)
    - [6.12 Per-epoch Results](#612-per-epoch-results-kết-quả-từng-epoch)
    - [6.13 Incremental Reporting & Model-type-aware Metrics](#613-incremental-reporting--model-type-aware-metrics)
  - [7. Quản lý tài nguyên (Resource Management)](#7-quản-lý-tài-nguyên-resource-management)
- [Cấu hình (Configuration)](#-cấu-hình-configuration)
- [Cấu trúc dự án](#-cấu-trúc-dự-án)
- [Tổng hợp lệnh thường dùng](#-tổng-hợp-lệnh-thường-dùng)
- [Tài liệu chi tiết](#-tài-liệu-chi-tiết)

---

## 🌟 Tính năng

| Thành phần | Chi tiết |
|:----------:|:---------|
| 🖼️ **Visual Encoders** | CLIP ViT-B/32, ResNet, Swin Transformer, DINOv2 |
| 📝 **Text Encoders** | PhoBERT (tối ưu tiếng Việt), BERT, RoBERTa, BARTpho |
| 🔗 **Multimodal Fusion** | Cross-Attention, Bilinear, MCAN, Concat, MuTAN |
| 🧠 **Mixture of Experts** | VQA MOE Layer, 4 loại router, 6+ loại expert chuyên biệt |
| 📚 **Knowledge Base / RAG** | FAISS vector store, dense/sparse/hybrid retriever |
| 📊 **Metrics** | VQA Soft Accuracy, BLEU-4, METEOR, ROUGE-L, CIDEr, F1, WUPS |
| 🔬 **Ablation Study** | Expert-level, Router-level, tự động báo cáo Markdown/CSV/LaTeX |
| ⚡ **Training** | AMP FP16, gradient accumulation, early stopping, cosine warmup |
| 🛡️ **Resource Management** | GPU/CPU/RAM monitoring, emergency backup, auto shutdown |
| 🎯 **Dual Paradigm** | Classification VQA + Generative VQA (encoder-decoder) |
| 🔄 **Graceful Interruption** | Ctrl+C → emergency checkpoint → tự động resume |
| 📝 **Per-experiment Logging** | Log riêng biệt cho từng experiment, per-epoch CSV/JSON |
| 📈 **Incremental Reporting** | Báo cáo tự động sau mỗi experiment hoàn thành |

---

## 🏗 Kiến trúc tổng quan

```
┌─────────────────────────────────────────────────────────────────────┐
│                        VQA Pipeline Orchestrator                    │
│    (Classification: vqa_pipeline.py / Generative: generative_vqa)  │
├──────────┬──────────────┬──────────────────┬────────────────────────┤
│ Data     │ Model        │ Training         │ Evaluation             │
│ Pipeline │ Pipeline     │ Pipeline         │ Pipeline               │
├──────────┼──────────────┼──────────────────┼────────────────────────┤
│ CSV+Img  │ VisualEnc    │ AdamW + Cosine   │ VQA Accuracy           │
│ PhoBERT  │ TextEnc      │ AMP FP16         │ BLEU / METEOR          │
│ Tokenize │ Fusion       │ Early Stopping   │ ROUGE-L / CIDEr        │
│ Augment  │ MOE Layer    │ Grad Accumulate  │ Precision/Recall/F1    │
│ Split    │ Knowledge    │ Checkpointing    │ Per-type Analysis      │
│ Loader   │ AnswerHead   │ Resource Mgmt    │ Confusion Matrix       │
└──────────┴──────────────┴──────────────────┴────────────────────────┘
                                │
                    ┌───────────┴───────────┐
                    │    Ablation Study      │
                    │  Expert / Router / MOE │
                    │  Analysis & Reporting  │
                    └───────────────────────┘
```

---

## 📦 Cài đặt

### Yêu cầu hệ thống

- Python >= 3.11
- CUDA >= 12.0 (khuyến nghị, hỗ trợ CPU nhưng chậm)
- RAM >= 16 GB
- VRAM >= 6 GB (RTX 3060 trở lên)

### Cài đặt với Conda (khuyến nghị)

```bash
# 1. Clone repo
git clone https://github.com/your-username/AutovivqaModelBuilder.git
cd AutovivqaModelBuilder

# 2. Tạo môi trường
conda create -n tgng.modelbuilder python=3.11 -y
conda activate tgng.modelbuilder

# 3. Cài đặt PyTorch (CUDA 12.x)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 4. Cài đặt project
pip install -e .

# Hoặc với Poetry
poetry install
```

### Cài đặt với venv

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -e .
```

### Kiểm tra cài đặt

```bash
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python -c "from src.core.vqa_pipeline import VQAPipeline; print('Import OK')"
```

---

## 🚀 Bắt đầu nhanh

### One-Command (Dùng shell script)

```bash
# Quick start: download data + train
bash src/cli/quick_start.sh
```

### Chạy từ YAML config

```bash
# Dùng file config mặc định
python -m src.core.vqa_pipeline --config configs/pipeline_config.yaml
```

### Chạy trực tiếp với tham số

```bash
python -m src.core.vqa_pipeline \
    --mode train \
    --images-dir data/raw/images \
    --text-file data/raw/texts/evaluate_60k_data_balanced_preprocessed.csv \
    --visual-backbone vit \
    --text-encoder phobert \
    --epochs 20 \
    --batch-size 32 \
    --learning-rate 2e-5 \
    --use-moe \
    --output-dir outputs \
    --seed 42
```

---

## 📖 Pipeline Guide

### 1. Chuẩn bị dữ liệu (Data Processing)

#### 1.1 Tải dữ liệu từ Kaggle

```bash
# Tải dataset AutoViVQA từ Kaggle
python -m src.data.download_data

# Hoặc dùng shell script
bash src/cli/download_data.sh
```

> **Yêu cầu:** Tạo file `~/.kaggle/kaggle.json` với API key từ [kaggle.com/settings](https://kaggle.com/settings).

#### 1.2 Cấu trúc dữ liệu đầu vào

```
data/
└── raw/
    ├── images/          # Ảnh (JPG/PNG)
    │   ├── image_001.jpg
    │   ├── image_002.jpg
    │   └── ...
    └── texts/
        └── evaluate_60k_data_balanced_preprocessed.csv
```

**Định dạng CSV** (3 cột bắt buộc):

| Cột | Mô tả | Ví dụ |
|-----|--------|-------|
| `image_link` | Tên file ảnh | `image_001.jpg` |
| `question` | Câu hỏi tiếng Việt | `Có bao nhiêu người trong ảnh?` |
| `answers` | Danh sách đáp án (JSON/pipe-separated) | `["hai", "2", "hai người"]` |

#### 1.3 Data Pipeline tự động

Data pipeline xử lý 9 bước tự động khi chạy training:

1. Load raw data (CSV + images)
2. Validate data (kiểm tra file ảnh tồn tại)
3. Compute statistics (phân tích phân phối)
4. Split data (train 80% / val 10% / test 10%)
5. Build answer vocabulary (min_freq = 5)
6. Initialize tokenizer (PhoBERT)
7. Build image transforms (augmentation)
8. Create DataLoaders
9. Validate DataLoaders

**Tùy chỉnh data processing:**

```bash
python -m src.core.vqa_pipeline \
    --mode train \
    --images-dir /path/to/images \
    --text-file /path/to/data.csv \
    --batch-size 16
```

Hoặc chỉnh trong `configs/pipeline_config.yaml`:

```yaml
data:
  images_dir: "data/raw/images"
  text_file: "data/raw/texts/evaluate_60k_data_balanced_preprocessed.csv"
  train_ratio: 0.8
  val_ratio: 0.1
  test_ratio: 0.1
  batch_size: 32
  eval_batch_size: 64
  num_workers: 4
  image_size: [224, 224]
  augmentation_strength: "medium"   # light | medium | strong
  tokenizer_name: "vinai/phobert-base"
  max_seq_length: 64
  min_answer_freq: 5
```

---

### 2. Huấn luyện (Training)

#### 2.1 Classification VQA (Phân loại đáp án)

**Cách 1: YAML Config (khuyến nghị)**

```bash
python -m src.core.vqa_pipeline --config configs/pipeline_config.yaml --mode train
```

**Cách 2: CLI Arguments**

```bash
python -m src.core.vqa_pipeline \
    --mode train \
    --epochs 20 \
    --batch-size 32 \
    --learning-rate 2e-5 \
    --visual-backbone vit \
    --text-encoder phobert \
    --use-moe \
    --output-dir outputs \
    --seed 42
```

**Cách 3: Shell Script (với colored output)**

```bash
bash src/cli/run_pipeline.sh \
    --mode train \
    --epochs 20 \
    --batch-size 32 \
    --visual-backbone vit \
    --text-encoder phobert \
    --use-moe
```

**Cách 4: Python API**

```python
from src.core.vqa_pipeline import VQAPipeline, VQAPipelineConfig

config = VQAPipelineConfig.from_yaml("configs/pipeline_config.yaml")
config.mode = "train"

pipeline = VQAPipeline(config)
output = pipeline.run()

print(f"Best VQA Accuracy: {output.training_output.best_metric:.4f}")
print(f"Checkpoint: {output.training_output.best_model_path}")
```

#### 2.2 Resume Training (tiếp tục huấn luyện)

```bash
# Resume từ checkpoint
python -m src.core.vqa_pipeline \
    --mode train \
    --resume checkpoints/checkpoint_epoch_5.pt \
    --epochs 20
```

#### 2.3 Training Options đầy đủ

| Flag | Mặc định | Mô tả |
|------|----------|-------|
| `--mode` | `train` | `train` / `evaluate` / `inference` |
| `--config` | — | Đường dẫn YAML config |
| `--images-dir` | `data/raw/images` | Thư mục ảnh |
| `--text-file` | `data/raw/texts/...csv` | File CSV dữ liệu |
| `--visual-backbone` | `vit` | `vit`, `resnet`, `swin`, `clip`, `dino` |
| `--text-encoder` | `phobert` | `phobert`, `bert`, `roberta`, `bartpho` |
| `--use-moe` | `False` | Bật Mixture of Experts |
| `--use-knowledge` | `False` | Bật Knowledge Base / RAG |
| `--epochs` | `20` | Số epoch |
| `--batch-size` | `32` | Batch size |
| `--learning-rate` | `2e-5` | Learning rate |
| `--output-dir` | `outputs` | Thư mục output |
| `--seed` | `42` | Random seed |
| `--resume` | — | Đường dẫn checkpoint để resume |

#### 2.4 Training Configs nâng cao

Chỉnh trong `configs/training_configs.yaml`:

```yaml
optimizer:
  name: "adamw"
  learning_rate: 2e-5
  weight_decay: 0.01
  betas: [0.9, 0.999]

scheduler:
  name: "cosine"
  warmup_ratio: 0.1
  min_lr: 1e-7

loss:
  label_smoothing: 0.1
  focal_loss_gamma: 2.0
  moe_aux_weight: 0.01

training:
  strategy: "gradual_unfreeze"  # full | freeze_visual | freeze_text | linear_probe | gradual_unfreeze
  mixed_precision: "fp16"       # off | fp16 | bf16
  gradient_accumulation_steps: 1
  max_grad_norm: 1.0

early_stopping:
  enabled: true
  patience: 5
  metric: "accuracy"
  min_delta: 0.001
```

#### 2.5 Checkpoint format

Checkpoint lưu tại `checkpoints/` gồm:

```python
{
    "epoch": int,
    "global_step": int,
    "model_state_dict": ...,
    "optimizer_state_dict": ...,
    "scheduler_state_dict": ...,
    "best_metric": float,
    "metrics": dict,
    "config": TrainingPipelineConfig,
    "model_config": VQAModelConfig,
    "vocabulary": dict,      # answer → id
    "num_answers": int
}
```

---

### 3. Đánh giá (Evaluation)

#### 3.1 Chạy evaluation

```bash
# Evaluation trên validation/test set
python -m src.core.vqa_pipeline \
    --mode evaluate \
    --resume checkpoints/best_model.pt
```

#### 3.2 Từ YAML config

```bash
# Trong pipeline_config.yaml, set mode: evaluate
python -m src.core.vqa_pipeline --config configs/pipeline_config.yaml --mode evaluate
```

#### 3.3 Metrics được tính

| Nhóm | Metric | Mô tả |
|------|--------|-------|
| **Classification** | VQA Soft Accuracy | `min(count/3, 1)` theo VQA v2 protocol |
| | Exact Match | So khớp chuỗi chính xác |
| | Top-K Accuracy | Top-5 accuracy |
| **NLG** | BLEU-4 | Bilingual Evaluation Understudy |
| | METEOR | Metric for Evaluation of Translation |
| | ROUGE-L | Longest Common Subsequence |
| | CIDEr | Consensus-based Image Description Evaluation |
| **Token-level** | Precision | Word-level precision |
| | Recall | Word-level recall |
| | F1 Score | Harmonic mean P/R |
| **Semantic** | WUPS | Wu-Palmer Similarity |

#### 3.4 Python API

```python
from src.pipeline.evaluator.vqa_evaluator import VQAEvaluator

evaluator = VQAEvaluator(model, val_loader, device=device, id2answer=id2answer)
result = evaluator.evaluate()

print(f"VQA Accuracy: {result.overall_metrics['vqa_accuracy']:.4f}")
print(f"BLEU-4: {result.overall_metrics['bleu']:.4f}")
```

---

### 4. Suy luận (Inference)

#### 4.1 Chạy inference

```bash
# Inference mode
python -m src.core.vqa_pipeline \
    --mode inference \
    --resume checkpoints/best_model.pt
```

#### 4.2 Python API

```python
from src.modeling.inference.vqa_predictor import VQAPredictor

predictor = VQAPredictor(
    model=model,
    tokenizer=tokenizer,
    id2answer=id2answer,
    device=device
)

# Single prediction
result = predictor.predict(image_path="path/to/image.jpg", question="Đây là gì?")
print(f"Answer: {result.answer} (confidence: {result.confidence:.4f})")
print(f"Top answers: {result.top_answers}")

# Batch prediction
results = predictor.predict_batch(images, questions)
```

#### 4.3 Inference config

Chỉnh trong `configs/inference_configs.yaml`:

```yaml
model:
  checkpoint_path: "checkpoints/best_model.pt"
  device: "auto"          # auto | cuda | cpu
  use_fp16: false

inference:
  batch_size: 32
  top_k: 5
  confidence_threshold: 0.0

output:
  format: "json"
  include_probabilities: true
  include_top_k: true
```

---

### 5. Generative VQA Pipeline

Pipeline cho mô hình sinh đáp án (encoder-decoder), khác với classification pipeline.

#### 5.1 Chạy Generative VQA

**Cách 1: YAML Config**

```bash
python -m src.core.generative_vqa_pipeline --config configs/generative_configs.yaml --mode train
```

**Cách 2: CLI Arguments**

```bash
python -m src.core.generative_vqa_pipeline \
    --mode train \
    --images-dir data/raw/images \
    --text-file data/raw/texts/evaluate_60k_data_balanced_preprocessed.csv \
    --visual-backbone openai/clip-vit-base-patch32 \
    --text-encoder vinai/phobert-base \
    --batch-size 8 \
    --num-epochs 20 \
    --learning-rate 5e-5 \
    --use-moe \
    --moe-type standard \
    --moe-position fusion \
    --num-experts 8 \
    --enable-resource-management
```

#### 5.2 Generative VQA Modes

| Mode | Lệnh | Mô tả |
|------|-------|-------|
| `train` | `--mode train` | Huấn luyện encoder-decoder |
| `evaluate` | `--mode evaluate` | Đánh giá NLG metrics |
| `inference` | `--mode inference` | Sinh đáp án cho ảnh + câu hỏi |
| `demo` | `--mode demo` | Demo tương tác |

#### 5.3 MOE Options cho Generative

```bash
python -m src.core.generative_vqa_pipeline \
    --mode train \
    --use-moe \
    --moe-type vqa \
    --moe-position both \
    --num-experts 8 \
    --num-vision-experts 2 \
    --num-text-experts 2 \
    --num-multimodal-experts 2 \
    --num-specialized-experts 2 \
    --expert-top-k 2
```

#### 5.4 Generation Options

```bash
python -m src.core.generative_vqa_pipeline \
    --mode inference \
    --max-generate-length 64 \
    --num-beams 5 \
    --do-sample \
    --temperature 0.8 \
    --top-k 50 \
    --top-p 0.95 \
    --checkpoint checkpoints/generative/best_generative_model.pt
```

#### 5.5 Tham số đầy đủ Generative Pipeline

| Flag | Mặc định | Mô tả |
|------|----------|-------|
| `--mode` | `train` | `train` / `evaluate` / `inference` / `demo` |
| `--config` | — | YAML config path |
| `--batch-size` | `16` | Batch size (giảm nếu thiếu VRAM) |
| `--num-epochs` | `20` | Số epoch |
| `--learning-rate` | `5e-5` | Learning rate |
| `--hidden-size` | `768` | Hidden dimension |
| `--num-decoder-layers` | `6` | Số layer decoder |
| `--num-attention-heads` | `8` | Số attention head |
| `--use-moe` | `False` | Bật MOE |
| `--moe-type` | `standard` | Loại MOE layer |
| `--moe-position` | `fusion` | Vị trí MOE trong model |
| `--use-knowledge` | `False` | Bật Knowledge Base / RAG |
| `--max-generate-length` | `64` | Token tối đa sinh ra |
| `--num-beams` | `1` | Beam search width (1 = greedy) |
| `--do-sample` | `False` | Bật sampling |
| `--temperature` | `1.0` | Sampling temperature |
| `--enable-resource-management` | `False` | Bật giám sát tài nguyên |

---

### 6. Ablation Study

Pipeline ablation tự động cho nghiên cứu MOE — phân tích vai trò từng loại expert, router, và cấu hình tối ưu.

#### 6.1 Dry Run (xem danh sách thí nghiệm)

```bash
python -m src.ablation.run_ablation \
    --config configs/ablation_config.yaml \
    --dry-run
```

Output mẫu:

```
======================================================================
  ABLATION STUDY DRY RUN: moe_ablation_study
  Total experiments: 27
======================================================================
  [  1] full__noisy_topk_k2            — Baseline (all experts)
  [  2] no_moe__noisy_topk_k2          — No MOE
  [  3] leave_one_out_no_multimodal     — LOO: -multimodal
  ...
  [ 27] subset_text_vision__noisy_topk  — Subset: text+vision

  Breakdown by mode:
    full: 8 | leave_one_out: 4 | no_moe: 1 | single_expert: 4 | subset: 10
======================================================================
```

#### 6.2 Chạy Ablation Study đầy đủ

```bash
# Từ checkpoint đã train
python -m src.ablation.run_ablation \
    --config configs/ablation_config.yaml \
    --checkpoint checkpoints/best_model.pt

# Với overrides
python -m src.ablation.run_ablation \
    --config configs/ablation_config.yaml \
    --checkpoint checkpoints/best_model.pt \
    --epochs 5 \
    --batch-size 16 \
    --output-dir outputs/ablation_quick

# Resume (tự bỏ qua experiment đã chạy)
python -m src.ablation.run_ablation \
    --config configs/ablation_config.yaml \
    --resume

# Chạy lại từ đầu
python -m src.ablation.run_ablation \
    --config configs/ablation_config.yaml \
    --no-resume

# Chạy lại một số experiment cụ thể từ đầu (xóa kết quả cũ), resume các experiment khác
python -m src.ablation.run_ablation \
    --config configs/ablation_config.yaml \
    --experiments 10,11,12 \
    --rerun 10,11
# → Experiment 10, 11: xóa kết quả cũ và chạy lại từ đầu
# → Experiment 12: bỏ qua nếu đã hoàn thành (resume)
```

#### 6.3 Chọn chạy experiment riêng lẻ

Thay vì chạy toàn bộ, có thể chọn chạy một số experiment cụ thể:

```bash
# Bước 1: Xem danh sách experiment (có đánh số)
python -m src.ablation.run_ablation --dry-run

# Bước 2a: Chạy experiment theo số — hỗ trợ số đơn, dãy, range
python -m src.ablation.run_ablation \
    --config configs/ablation_config.yaml \
    --experiments 1,3,5-7
# → Chạy experiment #1, #3, #5, #6, #7

# Bước 2b: Hoặc dùng chế độ tương tác — hiển thị danh sách, gõ số để chọn
python -m src.ablation.run_ablation \
    --config configs/ablation_config.yaml \
    --interactive
```

**Cú pháp `--experiments`:**

| Cú pháp | Ý nghĩa |
|---------|--------|
| `3` | Chạy experiment #3 |
| `1,3,5` | Chạy experiment #1, #3, #5 |
| `5-7` | Chạy experiment #5, #6, #7 |
| `1,3,5-7,10` | Kết hợp: #1, #3, #5, #6, #7, #10 |

**Chế độ `--interactive`:**

```
======================================================================
  EXPERIMENT LIST: moe_ablation_study
  Total experiments: 27
======================================================================

  ── FULL ──────────────────────────────────────────────
  [  1] full__noisy_topk_k2
        Baseline (all experts) | NoisyTopK k=2
  ── NO_MOE ────────────────────────────────────────────
  [  2] no_moe__noisy_topk_k2
        No MOE | NoisyTopK k=2
  ...

──────────────────────────────────────────────────────────────────────
  Enter experiment numbers to run.
  Format: 1,3,5-7  |  'all' = run all  |  'q' = quit
──────────────────────────────────────────────────────────────────────

  Your selection: 1,3,5-7

  Selected 5 experiment(s):
    [  1] full__noisy_topk_k2
    [  3] leave_one_out_no_multimodal__noisy_topk_k2
    [  5] single_expert_vision__noisy_topk_k2
    [  6] single_expert_text__noisy_topk_k2
    [  7] single_expert_multimodal__noisy_topk_k2

  Proceed? [Y/n]: y
```

#### 6.4 Ablation CLI Options

| Flag | Mặc định | Mô tả |
|------|----------|-------|
| `--config` | `configs/ablation_config.yaml` | YAML config |
| `--checkpoint` | — | Base model checkpoint |
| `--epochs` | (từ config) | Override số epoch per experiment |
| `--batch-size` | (từ config) | Override batch size |
| `--lr` | (từ config) | Override learning rate |
| `--seed` | (từ config) | Override seed |
| `--output-dir` | (từ config) | Override output directory |
| `--dry-run` | `False` | Liệt kê experiments, không chạy |
| `--experiments` | — | Chọn experiment theo số: `1,3,5-7` |
| `--rerun` | — | Chạy lại experiment cụ thể từ đầu (xóa kết quả cũ): `10,11` |
| `--interactive` | `False` | Chế độ tương tác: hiển thị danh sách, gõ số để chọn |
| `--resume` | `True` | Bỏ qua experiments đã hoàn thành |
| `--no-resume` | `False` | Chạy lại tất cả từ đầu |
| `--device` | `auto` | `cuda` / `cpu` / `cuda:0` |

#### 6.5 Các loại thí nghiệm Ablation

| Loại | Mô tả | Số experiments |
|------|--------|:--------------:|
| **Full Baseline** | Tất cả experts + default router | 1 |
| **No-MOE** | Tắt MOE hoàn toàn | 1 |
| **Single Expert** | Chỉ giữ 1 loại expert (vision / text / multimodal / specialized) | 4 |
| **Leave-One-Out** | Bỏ 1 loại expert, giữ 3 còn lại | 4 |
| **Subset** | Tổ hợp 2-3 loại expert | 10 |
| **Router Ablation** | Thay đổi router type & top_k trên full baseline | 7+ |

#### 6.6 Ablation Config YAML

```yaml
ablation:
  name: "moe_ablation_study"

  # Search space
  search_space:
    expert_types: [vision, text, multimodal, specialized]
    run_single_expert: true
    run_leave_one_out: true
    run_subsets: true
    max_subset_size: 3
    min_subset_size: 2
    include_no_moe_baseline: true
    include_full_baseline: true
    router_types: [topk, noisy_topk, soft, expert_choice]
    top_k_values: [1, 2, 4]
    load_balance_weights: [0.0, 0.01, 0.1]
    cross_expert_router: false   # true = cross-product (nhiều experiments hơn)

  # Training mỗi experiment
  num_epochs: 10
  learning_rate: 2e-5
  batch_size: 32
  early_stopping: true
  patience: 5

  # Output
  output_dir: "outputs/ablation"
  checkpoint_dir: "checkpoints/ablation"
  resume: true
```

#### 6.7 Output của Ablation Study

```
outputs/ablation/
├── moe_ablation_study_summary.json       # Tóm tắt + key findings
├── moe_ablation_study_report.md          # Báo cáo Markdown chi tiết
├── moe_ablation_study_table.tex          # Bảng LaTeX (cho paper)
├── moe_ablation_study_analysis.json      # Phân tích đầy đủ
├── moe_ablation_study_raw_results.json   # Kết quả thô
├── ablation_results.csv                  # Bảng metrics CSV
├── expert_contributions.csv              # Đóng góp từng expert
├── experiment_manifest.json              # Ma trận thí nghiệm
├── progress.json                         # Tracking tiến độ
├── experiment_results/                   # Kết quả từng experiment
│   ├── full__noisy_topk_k2.json
│   ├── no_moe__noisy_topk_k2.json
│   ├── single_expert_vision__noisy_topk_k2.json
│   └── ...
└── epoch_results/                        # Kết quả chi tiết từng epoch
    ├── full__noisy_topk_k2/
    │   ├── train_history.csv             # Loss mỗi epoch
    │   ├── val_history.csv               # Metrics validation mỗi epoch
    │   └── epoch_summary.json            # Metadata + toàn bộ history
    ├── no_moe__noisy_topk_k2/
    │   ├── train_history.csv
    │   ├── val_history.csv
    │   └── epoch_summary.json
    └── ...

logs/ablation/                            # Log riêng biệt từng experiment
├── full__noisy_topk_k2.log
├── no_moe__noisy_topk_k2.log
├── single_expert_vision__noisy_topk_k2.log
└── ...
```

**Báo cáo Markdown** bao gồm:
- Key Findings tự động phát hiện
- Bảng xếp hạng experiment theo primary metric (BLEU cho generative, VQA Accuracy cho classification)
- Phân tích đóng góp expert (essential / redundant)
- Ma trận synergy giữa các cặp expert
- So sánh router types
- Bảng metrics đầy đủ (tự động chọn đúng metrics theo model type)
- Delta từ baseline
- Khuyến nghị cấu hình tối ưu

#### 6.8 Ablation Python API

```python
from src.ablation import AblationRunner, AblationConfig

config = AblationConfig.from_yaml("configs/ablation_config.yaml")

# Xem experiment matrix
experiments = config.generate_experiment_matrix()
for i, exp in enumerate(experiments, 1):
    print(f"[{i}] {exp.experiment_id}: {exp.expert_config.description}")

# Chạy toàn bộ study
runner = AblationRunner(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    config=config,
    device=device,
    vocabulary=vocabulary,
    id2answer=id2answer,
    tokenizer=tokenizer,       # Bắt buộc cho generative model
)
summary = runner.run()
print(summary["key_findings"])

# Hoặc chạy một số experiment cụ thể (1-based)
summary = runner.run(selected_indices=[1, 3, 5, 6, 7])

# Chạy lại experiment cụ thể từ đầu
summary = runner.run(
    selected_indices=[10, 11, 12],
    rerun_indices=[10, 11],  # Chỉ rerun 10, 11; resume 12
)
```

#### 6.9 Chạy lại experiment cụ thể (`--rerun`)

Khi cần chạy lại một số experiment mà không ảnh hưởng đến kết quả của các experiment khác, sử dụng `--rerun`:

```bash
# Chạy lại experiment 10 và 11 từ đầu, resume experiment 12
python -m src.ablation.run_ablation \
    --experiments 10,11,12 \
    --rerun 10,11

# Chỉ rerun experiment 5 (không cần --experiments)
python -m src.ablation.run_ablation --rerun 5
```

**Cơ chế hoạt động:**
- `--rerun` xóa kết quả cũ của các experiment được chỉ định trước khi chạy lại
- Các experiment không nằm trong `--rerun` sẽ được resume bình thường (bỏ qua nếu đã hoàn thành)
- Có thể kết hợp `--experiments` và `--rerun` để kiểm soát chính xác experiment nào resume, experiment nào chạy lại

#### 6.10 Graceful Interruption (Ctrl+C)

Khi nhấn **Ctrl+C** trong quá trình chạy ablation study, hệ thống sẽ dừng an toàn:

```
╔══════════════════════════════════════════════════════════╗
║  ⚠ INTERRUPTED — saving emergency checkpoint...         ║
║  Emergency checkpoint saved: checkpoints/emergency/...   ║
║  Completed: 15/27                                        ║
║  Resume with: python -m src.ablation.run_ablation        ║
║               --config configs/ablation_config.yaml      ║
║               --resume                                   ║
╚══════════════════════════════════════════════════════════╝
```

**Quy trình xử lý:**

1. **Trong experiment**: Lưu emergency checkpoint (model state + optimizer + epoch) → kết thúc experiment hiện tại
2. **Trong runner**: Lưu progress + partial results → sinh báo cáo trung gian → thoát sạch
3. **Resume**: Chạy lại với `--resume` sẽ bỏ qua các experiment đã hoàn thành và tiếp tục từ experiment bị gián đoạn

**Lưu ý:** Emergency checkpoint được lưu tại `checkpoints/emergency_backups/`.

#### 6.11 Per-experiment Log Files

Mỗi experiment tự động tạo file log riêng biệt tại `logs/ablation/`:

```
logs/ablation/
├── full__noisy_topk_k2.log               # Log chi tiết experiment 1
├── no_moe__noisy_topk_k2.log             # Log chi tiết experiment 2
├── single_expert_vision__noisy_topk_k2.log
└── ...
```

Mỗi file log chứa:
- Thông tin cấu hình experiment (expert mask, router type)
- Chi tiết từng epoch (train loss, validation metrics)
- Thời gian bắt đầu/kết thúc
- Warning và error (nếu có)

Log format:
```
2026-02-24 10:15:32 | INFO | [full__noisy_topk_k2] Starting experiment...
2026-02-24 10:15:32 | INFO | Expert config: all experts enabled
2026-02-24 10:15:35 | INFO | Epoch 1/10 — train_loss: 2.3456
2026-02-24 10:15:40 | INFO | Epoch 1/10 — val_loss: 2.1234, bleu: 0.1523, meteor: 0.2341
...
```

#### 6.12 Per-epoch Results (Kết quả từng epoch)

Sau mỗi experiment, kết quả chi tiết từng epoch được lưu tự động tại `outputs/ablation/epoch_results/<experiment_id>/`:

**`train_history.csv`** — Loss training mỗi epoch:

```csv
epoch,train_loss
1,2.3456
2,1.8765
3,1.5432
...
```

**`val_history.csv`** — Validation metrics mỗi epoch:

```csv
epoch,val_loss,bleu,meteor,rouge_l,cider,exact_match,precision,recall,f1
1,2.1234,0.1523,0.2341,0.1876,0.3456,0.0512,0.2100,0.1800,0.1900
2,1.7654,0.2145,0.3012,0.2543,0.4567,0.0823,0.2800,0.2400,0.2600
...
```

**`epoch_summary.json`** — Metadata + toàn bộ history:

```json
{
  "experiment_id": "full__noisy_topk_k2",
  "total_epochs": 10,
  "best_epoch": 8,
  "best_metric": 0.3245,
  "primary_metric": "bleu",
  "timestamp": "2026-02-24T10:30:00",
  "train_history": [
    {"epoch": 1, "train_loss": 2.3456},
    {"epoch": 2, "train_loss": 1.8765}
  ],
  "val_history": [
    {"epoch": 1, "val_loss": 2.1234, "bleu": 0.1523, "meteor": 0.2341},
    {"epoch": 2, "val_loss": 1.7654, "bleu": 0.2145, "meteor": 0.3012}
  ]
}
```

> **Lưu ý:** Kết quả per-epoch cũng được lưu khi experiment bị gián đoạn bởi Ctrl+C (lưu các epoch đã hoàn thành).

#### 6.13 Incremental Reporting & Model-type-aware Metrics

**Báo cáo tăng dần:** Sau mỗi experiment hoàn thành, hệ thống tự động sinh báo cáo trung gian (Markdown + CSV) với kết quả hiện có. Điều này cho phép theo dõi tiến trình ngay trong khi ablation study đang chạy.

**Metrics tự động theo model type:** Evaluator và reporter tự động chọn bộ metrics phù hợp dựa trên `model_type` trong config:

| Model Type | Metrics sử dụng |
|:----------:|:-----------------|
| **generative** | BLEU, METEOR, ROUGE-L, CIDEr, Exact Match, Precision, Recall, F1, Loss, Perplexity |
| **classification** | VQA Accuracy, Accuracy, Exact Match, Precision, Recall, F1, Loss |

- Bảng LaTeX tự động chọn metrics phù hợp (generative: BLEU/METEOR/ROUGE-L/CIDEr; classification: VQA Accuracy/Accuracy)
- CSV export chỉ chứa metrics liên quan đến model type
- Markdown report xếp hạng theo `primary_metric` (mặc định: `bleu` cho generative, `vqa_accuracy` cho classification)

Cấu hình trong `configs/ablation_config.yaml`:

```yaml
ablation:
  model_type: "generative"    # "generative" hoặc "classification"
  primary_metric: "bleu"      # Metric chính để xếp hạng
```

---

### 7. Quản lý tài nguyên (Resource Management)

#### 7.1 Bật Resource Management

```bash
# Trong Generative pipeline
python -m src.core.generative_vqa_pipeline \
    --mode train \
    --enable-resource-management

# Hoặc trong Python
from src.resource_management import get_resource_manager

rm = get_resource_manager()
rm.start()  # Bắt đầu monitoring

# ... training code ...

rm.stop()
```

#### 7.2 Tính năng

- **Real-time monitoring**: CPU, RAM, GPU (VRAM), Disk mỗi 5 giây
- **Threshold alerts**: Warning (70%), Critical (90%)
- **Emergency backup**: Tự động save checkpoint khi resource critical
- **Auto shutdown**: Tắt training khi đạt ngưỡng critical
- **Reports**: JSON reports tự động mỗi 30 phút

#### 7.3 Cấu hình Resource

Xem `configs/resource_configs.yaml`:

```yaml
thresholds:
  cpu:
    warning: 70
    critical: 90
  memory:
    warning: 70
    critical: 90
  gpu:
    warning: 70
    critical: 90

critical_action: "backup_and_shutdown"

backup:
  emergency_dir: "checkpoints/emergency_backups"
  max_backups: 5
  include_optimizer: true

reports:
  auto_save_interval: 1800   # 30 phút
  output_dir: "logs/resource_reports"
```

---

## ⚙ Cấu hình (Configuration)

### File cấu hình

| File | Mô tả |
|------|--------|
| `configs/pipeline_config.yaml` | **Master config** — data + model + training |
| `configs/data_configs.yaml` | Data paths, preprocessing, normalization |
| `configs/model_configs.yaml` | Model architecture parameters |
| `configs/training_configs.yaml` | Optimizer, scheduler, loss, strategy |
| `configs/inference_configs.yaml` | Inference mode, decoding, output format |
| `configs/checkpoint_configs.yaml` | Save/load policy, best model tracking |
| `configs/generative_configs.yaml` | Generative VQA specific configs |
| `configs/ablation_config.yaml` | Ablation study search space |
| `configs/resource_configs.yaml` | GPU/CPU/RAM thresholds, backup policy |
| `configs/logging_configs.yaml` | Python logging handlers, formatters |
| `configs/warning_configs.yaml` | Resource warning thresholds |

### Thứ tự ưu tiên

```
CLI Arguments  >  YAML Config  >  Default Values
```

### Ví dụ kết hợp

```bash
# Dùng YAML config nhưng override epochs và batch-size qua CLI
python -m src.core.vqa_pipeline \
    --config configs/pipeline_config.yaml \
    --epochs 10 \
    --batch-size 16
```

---

## 📁 Cấu trúc dự án

```
AutovivqaModelBuilder/
├── configs/                        # Tất cả YAML configs
│   ├── pipeline_config.yaml        # Master pipeline config
│   ├── ablation_config.yaml        # Ablation study config
│   ├── data_configs.yaml           # Data processing
│   ├── model_configs.yaml          # Model architecture
│   ├── training_configs.yaml       # Training hyperparameters
│   ├── inference_configs.yaml      # Inference settings
│   ├── generative_configs.yaml     # Generative VQA config
│   ├── checkpoint_configs.yaml     # Checkpoint policy
│   ├── resource_configs.yaml       # Resource management
│   ├── logging_configs.yaml        # Logging setup
│   └── warning_configs.yaml        # Warning thresholds
│
├── src/
│   ├── core/                       # Core pipeline orchestrators
│   │   ├── vqa_pipeline.py         # Main VQA pipeline (classification)
│   │   ├── generative_vqa_pipeline.py  # Generative VQA pipeline
│   │   ├── data_pipeline.py        # 9-step data pipeline
│   │   ├── training_pipeline.py    # Training loop + AMP + early stopping
│   │   ├── generative_training_pipeline.py  # Generative training
│   │   ├── model_pipeline.py       # Model construction
│   │   └── pipeline_logger.py      # Colored pipeline logging
│   │
│   ├── modeling/                   # Model architectures
│   │   ├── meta_arch/              # VQA model + config
│   │   │   ├── vqa_model.py        # VietnameseVQAModel
│   │   │   ├── vqa_config.py       # VQAModelConfig
│   │   │   └── generative_vqa_model.py  # GenerativeVQAModel
│   │   ├── moe/                    # Mixture of Experts
│   │   │   ├── moe_layer.py        # MOELayer, VQAMOELayer, SparseMOE
│   │   │   ├── router.py           # TopK, NoisyTopK, Soft, ExpertChoice
│   │   │   ├── expert_types.py     # Vision, Text, Multimodal, GLU experts
│   │   │   ├── specialized_experts.py  # Segmentation, Detection, OCR, Scene
│   │   │   └── moe_config.py       # MOE configuration
│   │   ├── fusion/                 # Multimodal fusion strategies
│   │   ├── knowledge_base/         # RAG, FAISS, retriever, Vietnamese NLP
│   │   ├── inference/              # VQAPredictor, InferenceConfig
│   │   └── tokenizer/              # Tokenizer implementations
│   │
│   ├── ablation/                   # Ablation study pipeline
│   │   ├── run_ablation.py         # CLI entry point
│   │   ├── ablation_config.py      # Search space & experiment matrix
│   │   ├── ablation_runner.py      # Orchestrator (resume, progress)
│   │   ├── ablation_trainer.py     # Per-experiment training + MOE modifier
│   │   ├── ablation_evaluator.py   # Cross-experiment metric aggregation
│   │   ├── ablation_analyzer.py    # Expert importance, synergy, recommendations
│   │   └── ablation_reporter.py    # Markdown, CSV, LaTeX report generation
│   │
│   ├── solvers/                    # Optimizers, losses, metrics
│   │   ├── metrics/vqa_metrics.py  # VQA Accuracy, BLEU, METEOR, ROUGE, CIDEr
│   │   ├── losses/vqa_losses.py    # CE, Focal, Label Smoothing, MOE balance
│   │   └── optimizers/             # AdamW, LAMB, Lookahead, LR schedulers
│   │
│   ├── pipeline/                   # Sub-pipeline components
│   │   ├── trainer/                # VQATrainer (resource-aware)
│   │   └── evaluator/              # VQAEvaluator (per-type analysis)
│   │
│   ├── data/                       # Data handling
│   │   ├── dataset.py              # VQADataset + collate_fn
│   │   ├── generative_dataset.py   # GenerativeVQADataset
│   │   ├── data_actions.py         # Raw data loading
│   │   ├── augmentation.py         # Image augmentation
│   │   └── download_data.py        # Kaggle download
│   │
│   ├── resource_management/        # GPU/CPU/RAM monitoring
│   │   ├── resource_manager.py     # Main ResourceManager facade
│   │   ├── resource_monitor.py     # Real-time monitoring
│   │   ├── backup_handler.py       # Emergency checkpoint backup
│   │   └── progress_tracker.py     # Training progress tracker
│   │
│   ├── cli/                        # Shell script entry points
│   │   ├── run_pipeline.sh         # Main training script
│   │   ├── quick_start.sh          # Quick start (download + train)
│   │   ├── run_clean.sh            # Clean run (suppressed warnings)
│   │   ├── run_with_config.sh      # Config-based run
│   │   └── download_data.sh        # Data download script
│   │
│   ├── schema/                     # Data schemas (OneSample, etc.)
│   ├── exception/                  # Custom exceptions
│   ├── middleware/                  # Warning middleware
│   └── utils/                      # Utility functions
│
├── data/raw/                       # Raw data (images + CSV)
├── checkpoints/                    # Model checkpoints
├── outputs/                        # Pipeline outputs
├── logs/                           # Training & resource logs
├── examples/                       # Example scripts
├── docs/                           # Documentation
├── notebook/                       # Jupyter notebooks
├── tests/                          # Tests
│
├── pyproject.toml                  # Project metadata & dependencies
├── LICENSE                         # MIT License
└── README.md                       # This file
```

---

## 🔧 Tổng hợp lệnh thường dùng

```bash
# ═══════════════════════════════════════════════
#  DATA
# ═══════════════════════════════════════════════
python -m src.data.download_data                             # Tải data từ Kaggle

# ═══════════════════════════════════════════════
#  CLASSIFICATION VQA
# ═══════════════════════════════════════════════
python -m src.core.vqa_pipeline --mode train                 # Train mặc định
python -m src.core.vqa_pipeline --config configs/pipeline_config.yaml  # Train từ config
python -m src.core.vqa_pipeline --mode train --use-moe       # Train với MOE
python -m src.core.vqa_pipeline --mode train --use-knowledge # Train với RAG
python -m src.core.vqa_pipeline --mode evaluate --resume checkpoints/best_model.pt  # Evaluate
python -m src.core.vqa_pipeline --mode inference --resume checkpoints/best_model.pt # Inference

# ═══════════════════════════════════════════════
#  GENERATIVE VQA
# ═══════════════════════════════════════════════
python -m src.core.generative_vqa_pipeline --mode train      # Train generative
python -m src.core.generative_vqa_pipeline --mode train --use-moe --moe-type vqa   # Với MOE
python -m src.core.generative_vqa_pipeline --mode evaluate   # Evaluate NLG metrics
python -m src.core.generative_vqa_pipeline --mode inference --num-beams 5  # Beam search
python -m src.core.generative_vqa_pipeline --mode demo       # Demo tương tác
python -m src.core.generative_vqa_pipeline --config configs/generative_configs.yaml  # Từ config

# ═══════════════════════════════════════════════
#  ABLATION STUDY
# ═══════════════════════════════════════════════
python -m src.ablation.run_ablation --dry-run                # Xem danh sách experiments
python -m src.ablation.run_ablation --checkpoint checkpoints/best_model.pt  # Chạy đầy đủ
python -m src.ablation.run_ablation --experiments 1,3,5-7    # Chạy experiment cụ thể
python -m src.ablation.run_ablation --interactive            # Chọn tương tác
python -m src.ablation.run_ablation --epochs 5 --batch-size 16  # Quick ablation
python -m src.ablation.run_ablation --no-resume              # Chạy lại từ đầu
python -m src.ablation.run_ablation --rerun 10,11            # Chạy lại experiment 10, 11 từ đầu
python -m src.ablation.run_ablation --experiments 10-12 --rerun 10  # Mixed resume/rerun

# ═══════════════════════════════════════════════
#  SHELL SCRIPTS
# ═══════════════════════════════════════════════
bash src/cli/quick_start.sh                                  # Quick start
bash src/cli/run_pipeline.sh --mode train --epochs 20        # Shell script
bash src/cli/run_with_config.sh                              # Từ config
bash src/cli/run_clean.sh                                    # Clean run
```

---

## 📚 Tài liệu chi tiết

| Tài liệu | Mô tả |
|-----------|--------|
| [docs/getting_started.md](docs/getting_started.md) | Hướng dẫn cài đặt và bắt đầu |
| [docs/pipeline_usage.md](docs/pipeline_usage.md) | Chi tiết pipeline components |
| [docs/configuration_guide.md](docs/configuration_guide.md) | Tất cả configuration options |
| [docs/prepare_data.md](docs/prepare_data.md) | Chuẩn bị dữ liệu |
| [docs/vqa_architecture.md](docs/vqa_architecture.md) | Kiến trúc hệ thống |
| [docs/fusion_approaches.md](docs/fusion_approaches.md) | Chiến lược fusion |
| [docs/moe_approaches.md](docs/moe_approaches.md) | Thiết kế MOE |
| [docs/generative_vqa.md](docs/generative_vqa.md) | Generative VQA |
| [docs/knowledge_base_approaches.md](docs/knowledge_base_approaches.md) | RAG implementation |
| [docs/image_representation_approaches.md](docs/image_representation_approaches.md) | Visual encoders |
| [docs/text_representation_approaches.md](docs/text_representation_approaches.md) | Text encoders |
| [docs/api_reference.md](docs/api_reference.md) | Python API reference |

---

## 📄 License

MIT License — xem [LICENSE](LICENSE) để biết thêm chi tiết.

---

<div align="center">

**AutoViVQA Model Builder** — Vietnamese Visual Question Answering

</div>
