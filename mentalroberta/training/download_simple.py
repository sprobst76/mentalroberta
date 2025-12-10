#!/usr/bin/env python3
"""
Schneller Dataset-Download von Huggingface (ohne Kaggle API)
============================================================

Dieses Script lädt direkt von Huggingface herunter - keine API Keys nötig!

Verwendung:
    python download_simple.py --samples 2000 --output german_hf.json
    python download_simple.py --samples 5000 --output german_hf.json --balance
"""

import argparse
import json
import random
from pathlib import Path

print("🔄 Lade Dependencies...")

try:
    import torch
    from transformers import MarianMTModel, MarianTokenizer
    from datasets import load_dataset
    from tqdm import tqdm
except ImportError as e:
    print(f"❌ Fehlende Dependencies: {e}")
    print("   Installiere mit:")
    print("   pip install torch transformers datasets tqdm")
    exit(1)

# Unsere Ziel-Labels
TARGET_LABELS = ['depression', 'anxiety', 'bipolar', 'suicidewatch', 'offmychest']

# Label-Mapping
LABEL_MAP = {
    0: 'anxiety',      # Anxiety
    1: 'depression',   # Depression  
    2: 'suicidewatch', # Suicidal
    3: 'anxiety',      # Stress → Anxiety
    4: 'bipolar',      # Bipolar
    5: 'bipolar',      # Personality disorder → Bipolar
    6: 'offmychest',   # Normal → OffMyChest
}


class Translator:
    """Einfacher Deutsch-Übersetzer"""
    
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"🔄 Lade Übersetzer auf {self.device}...")
        
        model_name = "Helsinki-NLP/opus-mt-en-de"
        self.tokenizer = MarianTokenizer.from_pretrained(model_name)
        self.model = MarianMTModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        print("✅ Übersetzer bereit!")
    
    def translate(self, texts, batch_size=8):
        """Übersetze Liste von Texten"""
        results = []
        
        for i in tqdm(range(0, len(texts), batch_size), desc="Übersetze"):
            batch = texts[i:i+batch_size]
            
            # Kürze lange Texte
            batch = [t[:1500] if len(t) > 1500 else t for t in batch]
            
            inputs = self.tokenizer(
                batch,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            ).to(self.device)
            
            with torch.no_grad():
                translated = self.model.generate(**inputs, max_length=512)
            
            results.extend(self.tokenizer.batch_decode(translated, skip_special_tokens=True))
        
        return results


def main():
    parser = argparse.ArgumentParser(description="Einfacher Dataset-Download von Huggingface")
    parser.add_argument('--samples', type=int, default=2000, help='Anzahl Samples')
    parser.add_argument('--output', type=str, default='german_hf_dataset.json', help='Ausgabe-Datei')
    parser.add_argument('--balance', action='store_true', help='Klassen balancieren')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch-Größe')
    args = parser.parse_args()
    
    print("=" * 60)
    print("  Huggingface Mental Health Dataset → Deutsch")
    print("=" * 60)
    
    # 1. Dataset laden
    print("\n📥 Lade Dataset von Huggingface...")
    
    # Verwende ein öffentlich verfügbares Dataset
    try:
        # Versuche zuerst das Kaggle-ähnliche Dataset
        ds = load_dataset("mrSoul7766/mental-health", split="train")
        text_col = "text"
        label_col = "label"
        print(f"   ✅ mrSoul7766/mental-health geladen: {len(ds)} Einträge")
    except Exception:
        try:
            # Alternative: Reddit-basiertes Dataset
            ds = load_dataset("solomonk/reddit_mental_health_posts", split="train")
            text_col = "selftext" if "selftext" in ds.column_names else "body"
            label_col = "subreddit"
            print(f"   ✅ reddit_mental_health_posts geladen: {len(ds)} Einträge")
        except Exception as e:
            print(f"❌ Konnte kein Dataset laden: {e}")
            print("\nAlternative: Lade manuell von Kaggle:")
            print("   kaggle datasets download -d suchintikasarkar/sentiment-analysis-for-mental-health")
            return
    
    # 2. Samples auswählen
    print(f"\n🎯 Wähle {args.samples} Samples aus...")
    
    # Konvertiere zu Liste
    all_data = []
    for item in ds:
        text = item.get(text_col, "")
        label = item.get(label_col, 0)
        
        if not text or len(text) < 20:
            continue
        
        # Mappe Label
        if isinstance(label, int):
            mapped_label = LABEL_MAP.get(label, 'offmychest')
        else:
            label_str = str(label).lower()
            if 'depress' in label_str:
                mapped_label = 'depression'
            elif 'anxi' in label_str or 'stress' in label_str:
                mapped_label = 'anxiety'
            elif 'bipolar' in label_str:
                mapped_label = 'bipolar'
            elif 'suicid' in label_str:
                mapped_label = 'suicidewatch'
            else:
                mapped_label = 'offmychest'
        
        all_data.append({'text': text, 'label': mapped_label})
    
    print(f"   Verfügbar: {len(all_data)} gültige Einträge")
    
    # Stratified sampling
    if args.balance:
        samples_per_label = args.samples // len(TARGET_LABELS)
        selected = []
        for label in TARGET_LABELS:
            label_data = [d for d in all_data if d['label'] == label]
            random.shuffle(label_data)
            selected.extend(label_data[:samples_per_label])
            print(f"      {label}: {min(len(label_data), samples_per_label)} ausgewählt")
    else:
        random.shuffle(all_data)
        selected = all_data[:args.samples]
    
    print(f"   → {len(selected)} Samples ausgewählt")
    
    # 3. Übersetzen
    translator = Translator()
    
    texts_en = [d['text'] for d in selected]
    texts_de = translator.translate(texts_en, batch_size=args.batch_size)
    
    # 4. Zusammenführen
    result = []
    for item, text_de in zip(selected, texts_de):
        result.append({
            'text': text_de,
            'label': item['label']
        })
    
    # 5. Speichern
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    # Statistik
    label_counts = {}
    for item in result:
        label_counts[item['label']] = label_counts.get(item['label'], 0) + 1
    
    print(f"\n💾 Gespeichert: {args.output}")
    print(f"   Gesamt: {len(result)} Einträge")
    print(f"   Verteilung:")
    for label, count in sorted(label_counts.items()):
        print(f"      {label}: {count}")
    
    print("\n" + "=" * 60)
    print("  ✅ Fertig!")
    print("=" * 60)
    print(f"\nNächster Schritt:")
    print(f"   python train.py --data {args.output} --epochs 30 --batch_size 8")


if __name__ == "__main__":
    main()
