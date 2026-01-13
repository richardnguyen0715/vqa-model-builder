"""
Examples of using Text Representation models with pretrained weights.
"""

import torch
from src.modeling.heads import create_text_representation


def example_bert():
    """Example: BERT text embedding."""
    print("\n=== BERT Text Embedding ===")
    
    model = create_text_representation(
        model_type='bert',
        model_name='bert-base-uncased',
        pooling_strategy='cls',
        finetune=True
    )
    
    # Get tokenizer
    tokenizer = model.get_tokenizer()
    
    # Tokenize questions
    questions = ["What color is the car?", "How many dogs are there?"]
    encoded = tokenizer(
        questions,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors='pt'
    )
    
    # Forward
    model.eval()
    with torch.no_grad():
        features = model(encoded['input_ids'], encoded['attention_mask'])
    
    print(f"Questions: {questions}")
    print(f"Input shape: {encoded['input_ids'].shape}")
    print(f"Output shape: {features.shape}")
    print(f"Feature dimension: {model.get_feature_dim()}")


def example_roberta():
    """Example: RoBERTa text embedding."""
    print("\n=== RoBERTa Text Embedding ===")
    
    model = create_text_representation(
        model_type='roberta',
        model_name='roberta-base',
        pooling_strategy='mean',
        finetune=True
    )
    
    tokenizer = model.get_tokenizer()
    
    questions = ["Is this a cat or dog?"]
    encoded = tokenizer(
        questions,
        padding=True,
        truncation=True,
        return_tensors='pt'
    )
    
    model.eval()
    with torch.no_grad():
        features = model(encoded['input_ids'], encoded['attention_mask'])
    
    print(f"Questions: {questions}")
    print(f"Output shape: {features.shape}")
    print(f"Feature dimension: {model.get_feature_dim()}")


def example_deberta():
    """Example: DeBERTa V3 text embedding."""
    print("\n=== DeBERTa V3 Text Embedding ===")
    
    model = create_text_representation(
        model_type='deberta-v3',
        model_name='microsoft/deberta-v3-base',
        pooling_strategy='cls',
        finetune=False  # Freeze for feature extraction
    )
    
    tokenizer = model.get_tokenizer()
    
    questions = ["Where is the person standing?"]
    encoded = tokenizer(
        questions,
        padding=True,
        truncation=True,
        return_tensors='pt'
    )
    
    model.eval()
    with torch.no_grad():
        features = model(
            encoded['input_ids'],
            encoded['attention_mask'],
            encoded.get('token_type_ids', None)
        )
    
    print(f"Questions: {questions}")
    print(f"Output shape: {features.shape}")
    print(f"Feature dimension: {model.get_feature_dim()}")


def example_generic_distilbert():
    """Example: Generic wrapper with DistilBERT."""
    print("\n=== Generic (DistilBERT) Text Embedding ===")
    
    model = create_text_representation(
        model_type='generic',
        model_name='distilbert-base-uncased',
        pooling_strategy='mean',
        finetune=True
    )
    
    tokenizer = model.get_tokenizer()
    
    questions = ["What time is it?", "How old is he?"]
    encoded = tokenizer(
        questions,
        padding=True,
        truncation=True,
        return_tensors='pt'
    )
    
    model.eval()
    with torch.no_grad():
        features = model(encoded['input_ids'], encoded['attention_mask'])
    
    print(f"Questions: {questions}")
    print(f"Output shape: {features.shape}")
    print(f"Feature dimension: {model.get_feature_dim()}")


def example_pooling_strategies():
    """Compare different pooling strategies."""
    print("\n=== Pooling Strategies Comparison ===")
    
    pooling_strategies = ['cls', 'mean', 'max', 'all']
    questions = ["What is this?"]
    
    for strategy in pooling_strategies:
        print(f"\n--- Pooling: {strategy} ---")
        
        model = create_text_representation(
            model_type='bert',
            model_name='bert-base-uncased',
            pooling_strategy=strategy,
            finetune=False
        )
        
        tokenizer = model.get_tokenizer()
        encoded = tokenizer(
            questions,
            padding=True,
            truncation=True,
            return_tensors='pt'
        )
        
        model.eval()
        with torch.no_grad():
            features = model(encoded['input_ids'], encoded['attention_mask'])
        
        print(f"  Output shape: {features.shape}")


def example_comparison():
    """Compare all text representation models."""
    print("\n=== Comparison of All Text Models ===")
    
    models_config = [
        ('bert', 'bert-base-uncased'),
        ('roberta', 'roberta-base'),
        ('deberta-v3', 'microsoft/deberta-v3-base'),
        ('generic', 'distilbert-base-uncased'),
    ]
    
    questions = ["What color is the sky?"]
    
    print(f"\nQuestion: {questions[0]}")
    print("\nModel Outputs:")
    
    for model_type, model_name in models_config:
        try:
            model = create_text_representation(
                model_type=model_type,
                model_name=model_name,
                pooling_strategy='cls',
                finetune=False
            )
            
            tokenizer = model.get_tokenizer()
            encoded = tokenizer(
                questions,
                padding=True,
                truncation=True,
                return_tensors='pt'
            )
            
            model.eval()
            with torch.no_grad():
                features = model(encoded['input_ids'], encoded['attention_mask'])
            
            print(f"  {model_name:40s}: {tuple(features.shape)}")
            
        except Exception as e:
            print(f"  {model_name:40s}: Error - {e}")


def example_with_projection():
    """Example with projection layer."""
    print("\n=== With Projection Layer ===")
    
    # Create model with projection to custom dimension
    model = create_text_representation(
        model_type='bert',
        model_name='bert-base-uncased',
        output_dim=512,  # Custom dimension
        pooling_strategy='cls',
        add_projection=True  # Enable projection
    )
    
    tokenizer = model.get_tokenizer()
    questions = ["Test question"]
    encoded = tokenizer(questions, return_tensors='pt')
    
    model.eval()
    with torch.no_grad():
        features = model(encoded['input_ids'], encoded['attention_mask'])
    
    print(f"BERT hidden size: 768")
    print(f"Projected output: {features.shape}")
    print(f"Feature dimension: {model.get_feature_dim()}")


if __name__ == "__main__":
    print("=" * 60)
    print("Text Representation Models - Examples")
    print("=" * 60)
    
    # Run examples
    try:
        example_bert()
    except Exception as e:
        print(f"Error in BERT example: {e}")
    
    try:
        example_roberta()
    except Exception as e:
        print(f"Error in RoBERTa example: {e}")
    
    try:
        example_deberta()
    except Exception as e:
        print(f"Error in DeBERTa example: {e}")
    
    try:
        example_generic_distilbert()
    except Exception as e:
        print(f"Error in DistilBERT example: {e}")
    
    try:
        example_pooling_strategies()
    except Exception as e:
        print(f"Error in pooling strategies: {e}")
    
    try:
        example_comparison()
    except Exception as e:
        print(f"Error in comparison: {e}")
    
    try:
        example_with_projection()
    except Exception as e:
        print(f"Error in projection example: {e}")
    
    print("\n" + "=" * 60)
    print("Examples completed!")
    print("=" * 60)
