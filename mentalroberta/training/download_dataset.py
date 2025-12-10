#!/usr/bin/env python3
"""
Mental Health Dataset Downloader & Translator
==============================================

Dieses Script:
1. Lädt das Kaggle Mental Health Dataset herunter
2. Übersetzt die englischen Texte nach Deutsch
3. Mappt die Labels auf unsere 5 Kategorien
4. Kombiniert mit bestehenden deutschen Daten
5. Erstellt Train/Val/Test Splits

Verwendung:
    # Nur Kaggle-Daten herunterladen und übersetzen
    python download_dataset.py --source kaggle --output data_translated.json
    
    # Mit bestehenden Daten kombinieren
    python download_dataset.py --source kaggle --output data_combined.json --combine german_data.json
    
    # Huggingface SWMH Dataset verwenden
    python download_dataset.py --source huggingface --output data_swmh.json
    
    # Nur übersetzen (wenn CSV bereits vorhanden)
    python download_dataset.py --source local --input Combined_Data.csv --output data_translated.json

Voraussetzungen:
    pip install transformers torch pandas tqdm kaggle datasets

Für Kaggle:
    1. Account auf kaggle.com erstellen
    2. API Key herunterladen: kaggle.com/settings -> "Create New Token"
    3. kaggle.json nach ~/.kaggle/ (Linux/Mac) oder C:\\Users\\<user>\\.kaggle\\ (Windows) kopieren
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Dict, Optional
import random

# Prüfe Dependencies
def check_dependencies():
    """Prüfe ob alle Dependencies installiert sind"""
    missing = []
    
    try:
        import torch
    except ImportError:
        missing.append("torch")
    
    try:
        import transformers
    except ImportError:
        missing.append("transformers")
    
    try:
        import pandas
    except ImportError:
        missing.append("pandas")
    
    try:
        from tqdm import tqdm
    except ImportError:
        missing.append("tqdm")
    
    if missing:
        print(f"❌ Fehlende Dependencies: {', '.join(missing)}")
        print(f"   Installiere mit: pip install {' '.join(missing)}")
        sys.exit(1)

check_dependencies()

import torch
import pandas as pd
from tqdm import tqdm
from transformers import MarianMTModel, MarianTokenizer


# =============================================================================
# Konfiguration
# =============================================================================

# Label-Mapping: Kaggle/SWMH Labels → Unsere 5 Kategorien
LABEL_MAPPING = {
    # Kaggle Dataset Labels
    'depression': 'depression',
    'anxiety': 'anxiety', 
    'bipolar': 'bipolar',
    'suicidal': 'suicidewatch',
    'stress': 'anxiety',  # Stress → Anxiety
    'normal': 'offmychest',  # Normal → OffMyChest
    'personality disorder': 'bipolar',  # Personality → Bipolar (ähnlich)
    
    # SWMH Dataset Labels (falls verwendet)
    'adhd': 'offmychest',
    'ptsd': 'anxiety',
    'autism': 'offmychest',
    'schizophrenia': 'bipolar',
    'eating': 'depression',
    'selfharm': 'suicidewatch',
    'lonely': 'depression',
    'health anxiety': 'anxiety',
    'social anxiety': 'anxiety',
    'ocd': 'anxiety',
    'bpd': 'bipolar',
    
    # Bereits korrekte Labels
    'suicidewatch': 'suicidewatch',
    'offmychest': 'offmychest',
}

# Unsere Ziel-Labels
TARGET_LABELS = ['depression', 'anxiety', 'bipolar', 'suicidewatch', 'offmychest']


# =============================================================================
# Übersetzung
# =============================================================================

class GermanTranslator:
    """Übersetzt englische Texte nach Deutsch mit MarianMT"""
    
    def __init__(self, device: str = None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🔄 Lade Übersetzer (Helsinki-NLP/opus-mt-en-de) auf {self.device}...")
        
        self.model_name = "Helsinki-NLP/opus-mt-en-de"
        self.tokenizer = MarianTokenizer.from_pretrained(self.model_name)
        self.model = MarianMTModel.from_pretrained(self.model_name).to(self.device)
        self.model.eval()
        
        print("✅ Übersetzer geladen!")
    
    def translate(self, text: str, max_length: int = 512) -> str:
        """Übersetze einen einzelnen Text"""
        if not text or not isinstance(text, str):
            return ""
        
        # Kürze sehr lange Texte
        if len(text) > 2000:
            text = text[:2000] + "..."
        
        try:
            inputs = self.tokenizer(
                text, 
                return_tensors="pt", 
                truncation=True, 
                max_length=max_length
            ).to(self.device)
            
            with torch.no_grad():
                translated = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    num_beams=4,
                    early_stopping=True
                )
            
            result = self.tokenizer.decode(translated[0], skip_special_tokens=True)
            return result
            
        except Exception as e:
            print(f"⚠️ Übersetzungsfehler: {e}")
            return text  # Rückgabe des Originals bei Fehler
    
    def translate_batch(self, texts: List[str], batch_size: int = 8) -> List[str]:
        """Übersetze mehrere Texte in Batches"""
        results = []
        
        for i in tqdm(range(0, len(texts), batch_size), desc="Übersetze"):
            batch = texts[i:i+batch_size]
            
            # Filtere leere Texte
            valid_batch = [t if t and isinstance(t, str) else "" for t in batch]
            
            try:
                inputs = self.tokenizer(
                    valid_batch,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512,
                    padding=True
                ).to(self.device)
                
                with torch.no_grad():
                    translated = self.model.generate(
                        **inputs,
                        max_length=512,
                        num_beams=4,
                        early_stopping=True
                    )
                
                batch_results = self.tokenizer.batch_decode(translated, skip_special_tokens=True)
                results.extend(batch_results)
                
            except Exception as e:
                print(f"⚠️ Batch-Fehler: {e}")
                # Bei Fehler einzeln übersetzen
                for text in valid_batch:
                    results.append(self.translate(text))
        
        return results


# =============================================================================
# Datenquellen
# =============================================================================

def download_kaggle_dataset(output_dir: str = ".") -> Path:
    """Lade Kaggle Mental Health Dataset herunter"""
    print("\n📥 Lade Kaggle Dataset herunter...")
    
    dataset_name = "suchintikasarkar/sentiment-analysis-for-mental-health"
    output_path = Path(output_dir)
    csv_path = output_path / "Combined_Data.csv"
    
    # Prüfe ob bereits vorhanden
    if csv_path.exists():
        print(f"✅ Dataset bereits vorhanden: {csv_path}")
        return csv_path
    
    # Prüfe Kaggle API
    try:
        import kaggle
    except ImportError:
        print("❌ Kaggle API nicht installiert!")
        print("   pip install kaggle")
        print("\n   Dann: kaggle.com/settings → 'Create New Token' → kaggle.json herunterladen")
        print("   Kopiere nach: ~/.kaggle/kaggle.json (Linux/Mac)")
        print("   Oder: C:\\Users\\<user>\\.kaggle\\kaggle.json (Windows)")
        sys.exit(1)
    
    # Download
    try:
        print(f"   Downloading {dataset_name}...")
        subprocess.run([
            "kaggle", "datasets", "download", "-d", dataset_name,
            "-p", str(output_path), "--unzip"
        ], check=True)
        
        if csv_path.exists():
            print(f"✅ Download erfolgreich: {csv_path}")
            return csv_path
        else:
            # Suche nach CSV
            for f in output_path.glob("*.csv"):
                print(f"✅ Gefunden: {f}")
                return f
                
    except subprocess.CalledProcessError as e:
        print(f"❌ Download fehlgeschlagen: {e}")
        print("\n   Manueller Download:")
        print(f"   1. Gehe zu: kaggle.com/datasets/{dataset_name}")
        print(f"   2. Lade 'Combined_Data.csv' herunter")
        print(f"   3. Speichere in: {output_path}")
        sys.exit(1)


def load_kaggle_data(csv_path: Path) -> List[Dict]:
    """Lade und verarbeite Kaggle CSV"""
    print(f"\n📂 Lade Kaggle-Daten: {csv_path}")
    
    df = pd.read_csv(csv_path)
    print(f"   Gefunden: {len(df)} Einträge")
    print(f"   Spalten: {list(df.columns)}")
    
    # Finde Text- und Label-Spalten
    text_col = None
    label_col = None
    
    for col in df.columns:
        col_lower = col.lower()
        if 'text' in col_lower or 'statement' in col_lower or 'post' in col_lower:
            text_col = col
        if 'status' in col_lower or 'label' in col_lower or 'class' in col_lower:
            label_col = col
    
    if not text_col or not label_col:
        print(f"   Verfügbare Spalten: {list(df.columns)}")
        text_col = input("   Text-Spalte eingeben: ").strip()
        label_col = input("   Label-Spalte eingeben: ").strip()
    
    print(f"   Text-Spalte: {text_col}")
    print(f"   Label-Spalte: {label_col}")
    
    # Label-Verteilung anzeigen
    print(f"\n   Original Label-Verteilung:")
    for label, count in df[label_col].value_counts().items():
        print(f"      {label}: {count}")
    
    # Konvertiere zu Liste
    data = []
    skipped = 0
    
    for _, row in df.iterrows():
        text = str(row[text_col]).strip()
        label = str(row[label_col]).strip().lower()
        
        # Skippe leere Texte
        if not text or text == 'nan' or len(text) < 10:
            skipped += 1
            continue
        
        # Mappe Label
        mapped_label = LABEL_MAPPING.get(label)
        if not mapped_label:
            # Versuche partial match
            for key, value in LABEL_MAPPING.items():
                if key in label or label in key:
                    mapped_label = value
                    break
        
        if not mapped_label:
            print(f"   ⚠️ Unbekanntes Label: '{label}' - überspringe")
            skipped += 1
            continue
        
        data.append({
            'text': text,
            'label': mapped_label,
            'original_label': label,
            'source': 'kaggle'
        })
    
    print(f"\n   ✅ Geladen: {len(data)} Einträge ({skipped} übersprungen)")
    
    return data


def load_huggingface_swmh() -> List[Dict]:
    """Lade SWMH Dataset von Huggingface"""
    print("\n📥 Lade SWMH Dataset von Huggingface...")
    
    try:
        from datasets import load_dataset
    except ImportError:
        print("❌ datasets nicht installiert!")
        print("   pip install datasets")
        sys.exit(1)
    
    try:
        dataset = load_dataset("AIMH/SWMH", split="train")
        print(f"   Gefunden: {len(dataset)} Einträge")
        
        data = []
        for item in tqdm(dataset, desc="Verarbeite"):
            text = item.get('text', '')
            label = item.get('label', '')
            
            if not text or len(text) < 10:
                continue
            
            # Label-Namen aus Index
            label_names = ['adhd', 'anxiety', 'bipolar', 'depression', 'eating', 
                          'health anxiety', 'lonely', 'ocd', 'ptsd', 'schizophrenia',
                          'social anxiety', 'suicidewatch']
            
            if isinstance(label, int) and label < len(label_names):
                label_name = label_names[label]
            else:
                label_name = str(label).lower()
            
            mapped_label = LABEL_MAPPING.get(label_name, 'offmychest')
            
            data.append({
                'text': text,
                'label': mapped_label,
                'original_label': label_name,
                'source': 'swmh'
            })
        
        print(f"   ✅ Geladen: {len(data)} Einträge")
        return data
        
    except Exception as e:
        print(f"❌ Fehler beim Laden: {e}")
        print("   Das SWMH Dataset erfordert möglicherweise Zugriffsgenehmigung.")
        print("   Alternative: Verwende --source kaggle")
        return []


# =============================================================================
# Hauptfunktionen
# =============================================================================

def process_dataset(
    data: List[Dict],
    translator: GermanTranslator,
    max_samples: int = None,
    batch_size: int = 8
) -> List[Dict]:
    """Übersetze und verarbeite Dataset"""
    
    if max_samples:
        print(f"\n🔢 Limitiere auf {max_samples} Samples...")
        # Stratified sampling - gleiche Anzahl pro Label
        samples_per_label = max_samples // len(TARGET_LABELS)
        
        sampled_data = []
        for label in TARGET_LABELS:
            label_data = [d for d in data if d['label'] == label]
            random.shuffle(label_data)
            sampled_data.extend(label_data[:samples_per_label])
        
        data = sampled_data
        print(f"   → {len(data)} Samples ausgewählt")
    
    # Extrahiere Texte
    texts = [d['text'] for d in data]
    
    # Übersetze
    print(f"\n🔄 Übersetze {len(texts)} Texte nach Deutsch...")
    translated_texts = translator.translate_batch(texts, batch_size=batch_size)
    
    # Kombiniere
    result = []
    for original, translated in zip(data, translated_texts):
        result.append({
            'text': translated,
            'label': original['label'],
            'text_en': original['text'],
            'source': original.get('source', 'unknown')
        })
    
    return result


def combine_datasets(
    translated_data: List[Dict],
    existing_path: str = None
) -> List[Dict]:
    """Kombiniere übersetzte Daten mit bestehenden deutschen Daten"""
    
    combined = translated_data.copy()
    
    if existing_path and Path(existing_path).exists():
        print(f"\n📎 Kombiniere mit: {existing_path}")
        
        with open(existing_path, 'r', encoding='utf-8') as f:
            existing_data = json.load(f)
        
        print(f"   Bestehende Daten: {len(existing_data)} Einträge")
        
        # Füge source hinzu falls nicht vorhanden
        for item in existing_data:
            if 'source' not in item:
                item['source'] = 'original_german'
        
        combined.extend(existing_data)
        print(f"   → Kombiniert: {len(combined)} Einträge")
    
    return combined


def balance_dataset(data: List[Dict], min_samples: int = None) -> List[Dict]:
    """Balanciere Dataset durch Undersampling"""
    
    # Zähle Labels
    label_counts = {}
    for item in data:
        label = item['label']
        label_counts[label] = label_counts.get(label, 0) + 1
    
    print(f"\n⚖️ Balanciere Dataset...")
    print(f"   Vorher:")
    for label, count in sorted(label_counts.items()):
        print(f"      {label}: {count}")
    
    # Bestimme Zielgröße
    if min_samples:
        target_count = min_samples
    else:
        target_count = min(label_counts.values())
    
    # Undersample
    balanced = []
    for label in TARGET_LABELS:
        label_data = [d for d in data if d['label'] == label]
        random.shuffle(label_data)
        balanced.extend(label_data[:target_count])
    
    print(f"   Nachher: {len(balanced)} Einträge ({target_count} pro Klasse)")
    
    return balanced


def create_splits(
    data: List[Dict],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    output_dir: str = "."
) -> Dict[str, Path]:
    """Erstelle Train/Val/Test Splits"""
    
    print(f"\n📊 Erstelle Splits (Train: {train_ratio:.0%}, Val: {val_ratio:.0%}, Test: {1-train_ratio-val_ratio:.0%})...")
    
    random.shuffle(data)
    
    n = len(data)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    splits = {
        'train': data[:train_end],
        'val': data[train_end:val_end],
        'test': data[val_end:]
    }
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    paths = {}
    for split_name, split_data in splits.items():
        path = output_path / f"mental_health_{split_name}.json"
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(split_data, f, indent=2, ensure_ascii=False)
        paths[split_name] = path
        print(f"   {split_name}: {len(split_data)} → {path}")
    
    return paths


def save_dataset(data: List[Dict], output_path: str):
    """Speichere Dataset als JSON"""
    
    # Entferne extra Felder für Kompatibilität mit train.py
    clean_data = []
    for item in data:
        clean_data.append({
            'text': item['text'],
            'label': item['label']
        })
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(clean_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Gespeichert: {output_path}")
    print(f"   Einträge: {len(clean_data)}")
    
    # Zeige Verteilung
    label_counts = {}
    for item in clean_data:
        label_counts[item['label']] = label_counts.get(item['label'], 0) + 1
    
    print(f"   Verteilung:")
    for label, count in sorted(label_counts.items()):
        print(f"      {label}: {count}")


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Mental Health Dataset Downloader & Translator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Beispiele:
  # Kaggle Dataset herunterladen und übersetzen (Standard)
  python download_dataset.py --source kaggle --output german_kaggle.json
  
  # Mit Limit (schneller für Tests)
  python download_dataset.py --source kaggle --output test.json --max_samples 500
  
  # Mit bestehenden Daten kombinieren
  python download_dataset.py --source kaggle --output combined.json --combine german_data.json
  
  # Huggingface SWMH Dataset
  python download_dataset.py --source huggingface --output german_swmh.json
  
  # Lokale CSV übersetzen
  python download_dataset.py --source local --input meine_daten.csv --output german_custom.json
  
  # Mit Train/Val/Test Splits
  python download_dataset.py --source kaggle --output data.json --create_splits
        """
    )
    
    # Datenquelle
    parser.add_argument('--source', type=str, default='kaggle',
                        choices=['kaggle', 'huggingface', 'local'],
                        help='Datenquelle (default: kaggle)')
    parser.add_argument('--input', type=str,
                        help='Eingabe-Datei für source=local')
    
    # Ausgabe
    parser.add_argument('--output', type=str, default='german_mental_health.json',
                        help='Ausgabe-JSON Datei')
    parser.add_argument('--output_dir', type=str, default='.',
                        help='Ausgabe-Verzeichnis')
    
    # Optionen
    parser.add_argument('--max_samples', type=int,
                        help='Maximale Anzahl Samples (für schnelle Tests)')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch-Größe für Übersetzung')
    parser.add_argument('--combine', type=str,
                        help='Mit bestehender JSON-Datei kombinieren')
    parser.add_argument('--balance', action='store_true',
                        help='Dataset balancieren (Undersampling)')
    parser.add_argument('--create_splits', action='store_true',
                        help='Train/Val/Test Splits erstellen')
    parser.add_argument('--no_translate', action='store_true',
                        help='Nicht übersetzen (für bereits deutsche Daten)')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("  Mental Health Dataset Downloader & Translator")
    print("=" * 60)
    
    # 1. Daten laden
    if args.source == 'kaggle':
        csv_path = download_kaggle_dataset(args.output_dir)
        data = load_kaggle_data(csv_path)
        
    elif args.source == 'huggingface':
        data = load_huggingface_swmh()
        if not data:
            sys.exit(1)
            
    elif args.source == 'local':
        if not args.input:
            print("❌ --input erforderlich für source=local")
            sys.exit(1)
        data = load_kaggle_data(Path(args.input))
    
    # 2. Übersetzen
    if not args.no_translate:
        translator = GermanTranslator()
        data = process_dataset(
            data, 
            translator, 
            max_samples=args.max_samples,
            batch_size=args.batch_size
        )
    
    # 3. Kombinieren
    if args.combine:
        data = combine_datasets(data, args.combine)
    
    # 4. Balancieren
    if args.balance:
        data = balance_dataset(data)
    
    # 5. Speichern
    output_path = Path(args.output_dir) / args.output
    save_dataset(data, output_path)
    
    # 6. Optional: Splits erstellen
    if args.create_splits:
        create_splits(data, output_dir=args.output_dir)
    
    print("\n" + "=" * 60)
    print("  ✅ Fertig!")
    print("=" * 60)
    print(f"\nNächster Schritt - Training starten:")
    print(f"  python train.py --data {output_path} --epochs 30 --batch_size 8")


if __name__ == "__main__":
    main()
