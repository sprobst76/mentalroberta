#!/usr/bin/env python3
"""
Quick Test Script for MentalRoBERTa-Caps
Demonstrates the model without requiring Streamlit

Usage: python -m mentalroberta.tools.quick_test
"""

import torch
import torch.nn.functional as F
import re

print("=" * 60)
print("  MentalRoBERTa-Caps Quick Test")
print("  Based on Wagay et al. (2025)")
print("=" * 60)
print()

# Import model
print("📦 Loading model architecture...")
from mentalroberta.model import MentalRoBERTaCaps

# Labels
LABELS = ['depression', 'anxiety', 'bipolar', 'SuicideWatch', 'offmychest']

# Test texts
TEST_TEXTS = [
    "I feel so empty inside. Nothing brings me joy anymore. I just lay in bed all day.",
    "My heart is racing and I can't stop worrying about everything. What if something bad happens?",
    "Last week I was on top of the world, now I can barely get out of bed. These mood swings are exhausting.",
    "I just need to get this off my chest. Work has been stressful and nobody understands.",
    "I don't want to be here anymore. Everything feels hopeless."
]

def preprocess(text):
    """Simple text preprocessing"""
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def test_architecture():
    """Test model architecture"""
    print("🔧 Testing model architecture...")
    
    model = MentalRoBERTaCaps(num_classes=5, num_layers=6)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    print(f"   Memory footprint: ~{total_params * 4 / 1e9:.2f} GB")
    
    # Test forward pass
    batch_size = 2
    seq_len = 64
    dummy_input = torch.randint(0, 50265, (batch_size, seq_len))
    dummy_mask = torch.ones(batch_size, seq_len)
    
    print(f"\n   Test input shape: {dummy_input.shape}")
    
    model.eval()
    with torch.no_grad():
        logits, caps = model(dummy_input, dummy_mask)
    
    print(f"   Output logits shape: {logits.shape}")
    print(f"   Capsule outputs shape: {caps.shape}")
    print("   ✅ Architecture test passed!")
    
    return model

def test_with_tokenizer():
    """Test with actual tokenizer and texts"""
    print("\n🔤 Loading MentalRoBERTa tokenizer...")
    
    try:
        from transformers import AutoTokenizer, AutoModel
        tokenizer = AutoTokenizer.from_pretrained("mental/mental-roberta-base")
        print("   ✅ Tokenizer loaded successfully!")
    except Exception as e:
        print(f"   ⚠️ Could not load tokenizer: {e}")
        print("   Run: pip install transformers")
        return None
    
    # Create model
    model = MentalRoBERTaCaps(num_classes=5, num_layers=6)
    model.eval()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    print(f"   Model moved to: {device}")
    
    return model, tokenizer, device

def predict(text, model, tokenizer, device):
    """Make prediction"""
    clean_text = preprocess(text)
    
    inputs = tokenizer(
        clean_text,
        return_tensors='pt',
        max_length=512,
        truncation=True,
        padding=True
    )
    
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)
    
    with torch.no_grad():
        logits, capsule_outputs = model(input_ids, attention_mask)
        probs = F.softmax(logits, dim=-1)[0]
    
    return probs.cpu().numpy()

def main():
    # Test 1: Architecture
    print("\n" + "=" * 60)
    print("  TEST 1: Model Architecture")
    print("=" * 60)
    test_architecture()
    
    # Test 2: Full inference
    print("\n" + "=" * 60)
    print("  TEST 2: Full Inference Pipeline")
    print("=" * 60)
    
    result = test_with_tokenizer()
    
    if result is not None:
        model, tokenizer, device = result
        
        print("\n📊 Running predictions on test texts:\n")
        
        for i, text in enumerate(TEST_TEXTS, 1):
            probs = predict(text, model, tokenizer, device)
            top_idx = probs.argmax()
            top_label = LABELS[top_idx]
            top_prob = probs[top_idx] * 100
            
            print(f"Text {i}: \"{text[:60]}...\"")
            print(f"   → Prediction: {top_label.upper()} ({top_prob:.1f}%)")
            print(f"   → All probs: {', '.join([f'{l}:{p*100:.1f}%' for l, p in zip(LABELS, probs)])}")
            print()
    
    print("=" * 60)
    print("  ✅ All tests completed!")
    print("=" * 60)
    print("\n⚠️  Note: This is an untrained model - predictions are random!")
    print("    For real predictions, you need to train the model on mental health data.")
    print("\n📖 Paper: https://pmc.ncbi.nlm.nih.gov/articles/PMC12284574/")

if __name__ == "__main__":
    main()
