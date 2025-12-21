"""
Complete VQA Pipeline Example
Demonstrates end-to-end Vietnamese VQA model training, inference, and evaluation.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np


# =============================================================================
# 1. CONFIGURATION SETUP
# =============================================================================

def setup_model_config():
    """Setup Vietnamese VQA model configuration."""
    from src.modeling.meta_arch import (
        VQAModelConfig,
        VisualEncoderConfig,
        TextEncoderConfig,
        FusionConfig,
        MOEConfig,
        KnowledgeConfig,
        AnswerHeadConfig
    )
    
    config = VQAModelConfig(
        # Visual encoder: ViT-Base
        visual_encoder=VisualEncoderConfig(
            backbone_type='vit',
            model_name='openai/clip-vit-base-patch32',
            pretrained=True,
            output_dim=768,
            freeze_backbone=False
        ),
        
        # Text encoder: PhoBERT for Vietnamese
        text_encoder=TextEncoderConfig(
            encoder_type='phobert',
            model_name='vinai/phobert-base',
            pretrained=True,
            output_dim=768,
            max_length=64
        ),
        
        # Fusion: Cross-attention
        fusion=FusionConfig(
            fusion_type='cross_attention',
            hidden_dim=768,
            num_heads=8,
            num_layers=2,
            dropout=0.1
        ),
        
        # MOE: Enable mixture of experts
        moe=MOEConfig(
            use_moe=True,
            num_experts=8,
            top_k=2,
            load_balance_weight=0.01
        ),
        
        # Knowledge: RAG disabled for basic example
        knowledge=KnowledgeConfig(
            use_knowledge=False
        ),
        
        # Answer head
        answer_head=AnswerHeadConfig(
            num_answers=3000,  # Vietnamese VQA vocabulary size
            hidden_dims=[768, 512],
            dropout=0.1
        )
    )
    
    return config


def setup_training_config():
    """Setup training configuration."""
    from src.pipeline.trainer import (
        TrainingConfig,
        OptimizerConfig,
        SchedulerConfig,
        LossConfig,
        DataConfig,
        LoggingConfig,
        CheckpointConfig,
        EarlyStoppingConfig,
        TrainingStrategy,
        MixedPrecisionMode
    )
    
    config = TrainingConfig(
        num_epochs=20,
        gradient_accumulation_steps=2,
        max_grad_norm=1.0,
        strategy=TrainingStrategy.GRADUAL_UNFREEZE,
        mixed_precision=MixedPrecisionMode.FP16,
        seed=42,
        
        optimizer=OptimizerConfig(
            name="adamw",
            lr=2e-5,
            weight_decay=0.01,
            use_lookahead=True
        ),
        
        scheduler=SchedulerConfig(
            name="cosine",
            warmup_ratio=0.1
        ),
        
        loss=LossConfig(
            classification_loss="label_smoothing",
            label_smoothing=0.1,
            moe_aux_weight=0.01
        ),
        
        data=DataConfig(
            batch_size=16,
            eval_batch_size=32,
            num_workers=4
        ),
        
        logging=LoggingConfig(
            log_dir="logs",
            log_interval=50,
            use_tensorboard=True
        ),
        
        checkpoint=CheckpointConfig(
            save_dir="checkpoints",
            save_best=True,
            metric_for_best="accuracy"
        ),
        
        early_stopping=EarlyStoppingConfig(
            enabled=True,
            patience=5
        )
    )
    
    return config


# =============================================================================
# 2. DUMMY DATASET FOR DEMONSTRATION
# =============================================================================

class DummyVQADataset(Dataset):
    """
    Dummy VQA dataset for demonstration.
    Replace with actual Vietnamese VQA dataset.
    """
    
    def __init__(
        self,
        num_samples: int = 1000,
        image_size: int = 224,
        max_length: int = 64,
        num_answers: int = 3000
    ):
        self.num_samples = num_samples
        self.image_size = image_size
        self.max_length = max_length
        self.num_answers = num_answers
        
        # Vietnamese question templates
        self.question_templates = [
            "Có bao nhiêu {object} trong hình?",
            "Màu của {object} là gì?",
            "{object} đang ở đâu?",
            "Đây có phải là {object} không?",
            "Ai đang {action}?",
            "{object} có màu gì?",
            "Có mấy người trong hình?"
        ]
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Random image tensor (replace with actual image loading)
        image = torch.randn(3, self.image_size, self.image_size)
        
        # Random question (replace with actual tokenization)
        input_ids = torch.randint(0, 30000, (self.max_length,))
        attention_mask = torch.ones(self.max_length)
        
        # Random answer label
        label = torch.randint(0, self.num_answers, (1,)).item()
        
        # Question text (for evaluation)
        question = np.random.choice(self.question_templates)
        
        return {
            "image": image,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": label,
            "questions": question
        }


# =============================================================================
# 3. TRAINING
# =============================================================================

def train_model():
    """Train Vietnamese VQA model."""
    from src.modeling.meta_arch import create_vqa_model
    from src.pipeline.trainer import create_trainer, set_seed
    
    print("=" * 60)
    print("VIETNAMESE VQA MODEL TRAINING")
    print("=" * 60)
    
    # Setup configurations
    model_config = setup_model_config()
    train_config = setup_training_config()
    
    # Set random seed
    set_seed(train_config.seed)
    
    # Create model
    print("\n[1] Creating model...")
    model = create_vqa_model(model_config)
    print(f"    Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create datasets
    print("\n[2] Loading datasets...")
    train_dataset = DummyVQADataset(num_samples=1000)
    val_dataset = DummyVQADataset(num_samples=200)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config.data.batch_size,
        shuffle=True,
        num_workers=train_config.data.num_workers
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_config.data.eval_batch_size,
        shuffle=False
    )
    
    print(f"    Train samples: {len(train_dataset)}")
    print(f"    Val samples: {len(val_dataset)}")
    
    # Create trainer
    print("\n[3] Initializing trainer...")
    trainer = create_trainer(
        model=model,
        config=train_config,
        train_dataloader=train_loader,
        val_dataloader=val_loader
    )
    
    # Train
    print("\n[4] Starting training...")
    results = trainer.train()
    
    print("\n[5] Training complete!")
    print(f"    Best accuracy: {results['best_metric']:.4f}")
    print(f"    Best epoch: {results['best_epoch']}")
    
    return model, results


# =============================================================================
# 4. INFERENCE
# =============================================================================

def run_inference(model=None, checkpoint_path=None):
    """Run inference on sample images."""
    from src.modeling.inference import (
        VQAPredictor,
        VQAInferenceConfig,
        InferenceConfig,
        AnswerConfig,
        DecodingStrategy
    )
    
    print("\n" + "=" * 60)
    print("VIETNAMESE VQA INFERENCE")
    print("=" * 60)
    
    # Create dummy model if not provided
    if model is None:
        from src.modeling.meta_arch import create_vqa_model
        model_config = setup_model_config()
        model = create_vqa_model(model_config)
        print("\n[!] Using untrained model for demonstration")
    
    # Setup inference config
    inference_config = VQAInferenceConfig(
        inference=InferenceConfig(
            device="auto",
            use_fp16=True
        ),
        answer=AnswerConfig(
            decoding_strategy=DecodingStrategy.GREEDY,
            return_top_n=5,
            confidence_threshold=0.0
        )
    )
    
    # Create dummy answer vocabulary
    answer_vocabulary = [f"câu_trả_lời_{i}" for i in range(3000)]
    
    # Create predictor
    print("\n[1] Creating predictor...")
    predictor = VQAPredictor(
        model=model,
        config=inference_config,
        answer_vocabulary=answer_vocabulary
    )
    
    # Create dummy image
    print("\n[2] Running single prediction...")
    dummy_image = torch.randn(3, 224, 224)
    question = "Có bao nhiêu người trong hình?"
    
    result = predictor.predict(
        image=dummy_image,
        question=question,
        return_attention=False
    )
    
    print(f"    Question: {result.question}")
    print(f"    Answer: {result.answer}")
    print(f"    Confidence: {result.confidence:.4f}")
    print(f"    Top-5 answers:")
    for ans, score in result.top_answers:
        print(f"      - {ans}: {score:.4f}")
    
    # Batch inference
    print("\n[3] Running batch prediction...")
    images = [torch.randn(3, 224, 224) for _ in range(10)]
    questions = [
        "Có bao nhiêu người?",
        "Màu của xe là gì?",
        "Đây là ở đâu?",
        "Con vật này là gì?",
        "Thời tiết như thế nào?",
        "Có mấy cái cây?",
        "Người này đang làm gì?",
        "Đồ vật này màu gì?",
        "Có phải ban ngày không?",
        "Ai đang đứng?"
    ]
    
    batch_result = predictor.predict_batch(images, questions)
    
    print(f"    Batch size: {batch_result.batch_size}")
    print(f"    Inference time: {batch_result.inference_time:.2f}s")
    
    return predictor


# =============================================================================
# 5. EVALUATION
# =============================================================================

def evaluate_model(model=None):
    """Evaluate model on test set."""
    from src.pipeline.evaluator import (
        VQAEvaluator,
        EvaluationConfig,
        EvaluationMode,
        MetricConfig,
        AnalysisConfig
    )
    
    print("\n" + "=" * 60)
    print("VIETNAMESE VQA EVALUATION")
    print("=" * 60)
    
    # Create dummy model if not provided
    if model is None:
        from src.modeling.meta_arch import create_vqa_model
        model_config = setup_model_config()
        model = create_vqa_model(model_config)
        print("\n[!] Using untrained model for demonstration")
    
    # Setup evaluation config
    eval_config = EvaluationConfig(
        mode=EvaluationMode.DETAILED,
        batch_size=32,
        use_fp16=True,
        metrics=MetricConfig(
            compute_accuracy=True,
            compute_wups=True,
            compute_f1=True,
            compute_per_type=True,
            top_k_values=[1, 3, 5]
        ),
        analysis=AnalysisConfig(
            save_predictions=True,
            save_errors=True,
            num_error_examples=50
        ),
        output_dir="evaluation_results"
    )
    
    # Create answer vocabulary
    answer_vocabulary = [f"câu_trả_lời_{i}" for i in range(3000)]
    
    # Create evaluator
    print("\n[1] Creating evaluator...")
    evaluator = VQAEvaluator(
        config=eval_config,
        answer_vocabulary=answer_vocabulary
    )
    
    # Create test dataset and loader
    print("\n[2] Loading test data...")
    test_dataset = DummyVQADataset(num_samples=500)
    test_loader = DataLoader(
        test_dataset,
        batch_size=eval_config.batch_size,
        shuffle=False
    )
    
    # Run evaluation
    print("\n[3] Running evaluation...")
    result = evaluator.evaluate(
        model=model,
        dataloader=test_loader,
        return_predictions=True
    )
    
    # Print summary
    evaluator.print_summary(result)
    
    # Save results
    print("\n[4] Saving results...")
    evaluator.save_results(result)
    
    return result


# =============================================================================
# 6. COMPLETE PIPELINE
# =============================================================================

def run_complete_pipeline():
    """Run complete VQA pipeline: train -> inference -> evaluate."""
    
    print("\n" + "=" * 80)
    print(" COMPLETE VIETNAMESE VQA PIPELINE")
    print("=" * 80)
    
    # Step 1: Train
    model, train_results = train_model()
    
    # Step 2: Inference demo
    predictor = run_inference(model)
    
    # Step 3: Evaluate
    eval_results = evaluate_model(model)
    
    print("\n" + "=" * 80)
    print(" PIPELINE COMPLETE")
    print("=" * 80)
    print(f"\nTraining best accuracy: {train_results['best_metric']:.4f}")
    print(f"Evaluation accuracy: {eval_results.overall_metrics.get('accuracy', 0):.4f}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Vietnamese VQA Pipeline")
    parser.add_argument(
        "--mode",
        choices=["train", "inference", "evaluate", "full"],
        default="full",
        help="Pipeline mode to run"
    )
    
    args = parser.parse_args()
    
    if args.mode == "train":
        train_model()
    elif args.mode == "inference":
        run_inference()
    elif args.mode == "evaluate":
        evaluate_model()
    else:
        run_complete_pipeline()
