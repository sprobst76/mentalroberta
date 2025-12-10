"""
Training Script for MentalRoBERTa-Caps
Based on Wagay et al. (2025)

Usage:
    python train.py --data synthetic_data.json --epochs 10 --batch_size 16
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
import json
import argparse
from pathlib import Path
from tqdm import tqdm
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import warnings
warnings.filterwarnings('ignore', category=UserWarning)  # Unterdrücke sklearn Warnings
from collections import Counter
import re

from model import MentalRoBERTaCaps


# ============================================================
# Dataset
# ============================================================

class MentalHealthDataset(Dataset):
    """Dataset for mental health text classification"""
    
    LABEL_MAP = {
        'depression': 0,
        'anxiety': 1,
        'bipolar': 2,
        'suicidewatch': 3,
        'offmychest': 4
    }
    
    def __init__(self, data, tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Normalize labels
        for item in self.data:
            item['label'] = item['label'].lower().replace(' ', '')
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        text = self.preprocess(item['text'])
        label = self.LABEL_MAP.get(item['label'], 0)
        
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.long)
        }
    
    @staticmethod
    def preprocess(text):
        """Preprocess text as described in paper"""
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'<[^>]+>', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text


# ============================================================
# Training Functions
# ============================================================

def compute_class_weights(dataset):
    """Compute class weights for imbalanced data"""
    labels = [item['label'].lower().replace(' ', '') for item in dataset.data]
    label_counts = Counter(labels)
    
    total = len(labels)
    num_classes = len(MentalHealthDataset.LABEL_MAP)
    
    weights = []
    for label in sorted(MentalHealthDataset.LABEL_MAP.keys(), 
                       key=lambda x: MentalHealthDataset.LABEL_MAP[x]):
        count = label_counts.get(label, 1)
        weight = total / (num_classes * count)
        weights.append(weight)
    
    return torch.tensor(weights, dtype=torch.float32)


def train_epoch(model, dataloader, optimizer, scheduler, criterion, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    progress_bar = tqdm(dataloader, desc="Training")
    
    for batch in progress_bar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        
        optimizer.zero_grad()
        
        logits, capsule_outputs = model(input_ids, attention_mask)
        
        # Standard cross-entropy loss
        ce_loss = criterion(logits, labels)
        
        # Margin loss for capsules (optional, as in original capsule paper)
        caps_lengths = model.get_capsule_lengths(capsule_outputs)
        
        # Create one-hot labels
        one_hot = F.one_hot(labels, num_classes=logits.size(1)).float()
        
        # Margin loss: L = T_c * max(0, m+ - ||v_c||)^2 + lambda * (1 - T_c) * max(0, ||v_c|| - m-)^2
        m_plus = 0.9
        m_minus = 0.1
        lambda_val = 0.5
        
        present_error = F.relu(m_plus - caps_lengths) ** 2
        absent_error = F.relu(caps_lengths - m_minus) ** 2
        margin_loss = (one_hot * present_error + lambda_val * (1 - one_hot) * absent_error).sum(dim=1).mean()
        
        # Combined loss
        loss = ce_loss + 0.5 * margin_loss
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        
        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    avg_loss = total_loss / len(dataloader)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    
    return avg_loss, f1


def evaluate(model, dataloader, criterion, device):
    """Evaluate model"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            logits, _ = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            
            total_loss += loss.item()
            
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    
    # Metrics
    label_names = sorted(MentalHealthDataset.LABEL_MAP.keys(), 
                        key=lambda x: MentalHealthDataset.LABEL_MAP[x])
    
    report = classification_report(all_labels, all_preds, 
                                   target_names=label_names, 
                                   output_dict=True,
                                   zero_division=0)  # Unterdrücke Warnings
    
    return avg_loss, report, all_preds, all_labels


# ============================================================
# Main Training Loop
# ============================================================

def main(args):
    print("=" * 60)
    print("  MentalRoBERTa-Caps Training")
    print("=" * 60)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n📱 Device: {device}")
    
    # Load data
    print(f"\n📂 Loading data from: {args.data}")
    with open(args.data, 'r', encoding='utf-8') as f:
        all_data = json.load(f)
    
    print(f"   Total samples: {len(all_data)}")
    
    # Split data
    np.random.seed(42)
    np.random.shuffle(all_data)
    
    train_size = int(0.8 * len(all_data))
    val_size = int(0.1 * len(all_data))
    
    train_data = all_data[:train_size]
    val_data = all_data[train_size:train_size + val_size]
    test_data = all_data[train_size + val_size:]
    
    print(f"   Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
    
    # Tokenizer
    print(f"\n🔤 Loading tokenizer for: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    # Datasets
    train_dataset = MentalHealthDataset(train_data, tokenizer, args.max_length)
    val_dataset = MentalHealthDataset(val_data, tokenizer, args.max_length)
    test_dataset = MentalHealthDataset(test_data, tokenizer, args.max_length)
    
    # Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
    
    # Model
    print(f"\n🧠 Initializing model with: {args.model_name}")
    model = MentalRoBERTaCaps(
        num_classes=5,
        num_layers=args.num_layers,
        num_primary_caps=args.num_primary_caps,
        primary_cap_dim=args.primary_cap_dim,
        class_cap_dim=args.class_cap_dim,
        num_routing_iterations=args.routing_iterations,
        model_name=args.model_name
    )
    model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    
    # Class weights for imbalanced data
    class_weights = compute_class_weights(train_dataset).to(device)
    print(f"   Class weights: {class_weights.tolist()}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    
    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(0.1 * total_steps)
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    # Training
    print("\n" + "=" * 60)
    print("  Starting Training")
    print("=" * 60)
    
    best_val_f1 = 0
    best_epoch = 0
    
    for epoch in range(args.epochs):
        print(f"\n📈 Epoch {epoch + 1}/{args.epochs}")
        
        # Train
        train_loss, train_f1 = train_epoch(
            model, train_loader, optimizer, scheduler, criterion, device
        )
        print(f"   Train Loss: {train_loss:.4f}, Train F1: {train_f1:.4f}")
        
        # Validate
        val_loss, val_report, _, _ = evaluate(model, val_loader, criterion, device)
        val_f1 = val_report['macro avg']['f1-score']
        val_recall = val_report['macro avg']['recall']
        
        print(f"   Val Loss: {val_loss:.4f}, Val F1: {val_f1:.4f}, Val Recall: {val_recall:.4f}")
        
        # Save best model
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch + 1
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_f1': val_f1,
                'val_report': val_report
            }, args.output / 'best_model.pt')
            print(f"   💾 Saved best model (F1: {val_f1:.4f})")
    
    # Final evaluation on test set
    print("\n" + "=" * 60)
    print("  Final Evaluation on Test Set")
    print("=" * 60)
    
    # Load best model
    checkpoint = torch.load(args.output / 'best_model.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_loss, test_report, test_preds, test_labels = evaluate(
        model, test_loader, criterion, device
    )
    
    print(f"\n📊 Test Results (Best model from epoch {best_epoch}):")
    print(f"   Loss: {test_loss:.4f}")
    print(f"   Macro F1: {test_report['macro avg']['f1-score']:.4f}")
    print(f"   Macro Recall: {test_report['macro avg']['recall']:.4f}")
    
    print("\n📋 Per-class results:")
    label_names = sorted(MentalHealthDataset.LABEL_MAP.keys(), 
                        key=lambda x: MentalHealthDataset.LABEL_MAP[x])
    for label in label_names:
        if label in test_report:
            print(f"   {label:15s}: P={test_report[label]['precision']:.3f}, "
                  f"R={test_report[label]['recall']:.3f}, F1={test_report[label]['f1-score']:.3f}")
    
    # Confusion matrix
    print("\n🔢 Confusion Matrix:")
    cm = confusion_matrix(test_labels, test_preds)
    print(f"   {'':15s} " + " ".join([f"{l[:4]:>6s}" for l in label_names]))
    for i, row in enumerate(cm):
        print(f"   {label_names[i]:15s} " + " ".join([f"{v:6d}" for v in row]))
    
    # Save final report
    with open(args.output / 'test_report.json', 'w') as f:
        json.dump(test_report, f, indent=2)
    
    print(f"\n✅ Training complete! Model saved to: {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MentalRoBERTa-Caps")
    
    # Data
    parser.add_argument('--data', type=str, required=True, help='Path to training data JSON')
    parser.add_argument('--output', type=Path, default=Path('checkpoints'), help='Output directory')
    parser.add_argument('--max_length', type=int, default=256, help='Max sequence length')
    
    # Model
    parser.add_argument('--model_name', type=str, default='deepset/gbert-base',
                        help='HuggingFace model name (default: deepset/gbert-base for German)')
    parser.add_argument('--num_layers', type=int, default=6, help='Number of encoder layers')
    parser.add_argument('--num_primary_caps', type=int, default=8, help='Number of primary capsules')
    parser.add_argument('--primary_cap_dim', type=int, default=16, help='Primary capsule dimension')
    parser.add_argument('--class_cap_dim', type=int, default=16, help='Class capsule dimension')
    parser.add_argument('--routing_iterations', type=int, default=3, help='Routing iterations')
    
    # Training
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=2e-5, help='Learning rate')
    
    args = parser.parse_args()
    
    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)
    
    main(args)
