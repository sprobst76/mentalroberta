#!/usr/bin/env python3
"""
Inference-Script für MentalRoBERTa-Caps

Verwendung:
    # Einzelner Text
    python -m mentalroberta.inference --text "Ich fühle mich so leer..."
    
    # Aus Datei
    python -m mentalroberta.inference --input texte.json --output vorhersagen.json
    
    # Interaktiver Modus
    python -m mentalroberta.inference --interactive
"""

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
import json
import argparse
import re
from pathlib import Path
from tqdm import tqdm

from mentalroberta.model import MentalRoBERTaCaps


# Konfiguration
LABELS = ['depression', 'anxiety', 'bipolar', 'suicidewatch', 'offmychest']
LABELS_DE = ['Depression', 'Angst', 'Bipolar', 'Suizidalität', 'Ventil']


class MentalHealthClassifier:
    """Einfache Klasse für Inference mit trainiertem Modell"""
    
    def __init__(self, checkpoint_path, model_name="deepset/gbert-base", device=None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_name = model_name
        
        print(f"🔄 Lade Modell von: {checkpoint_path}")
        print(f"   Gerät: {self.device}")
        
        # Tokenizer laden
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Modell initialisieren
        self.model = MentalRoBERTaCaps(
            num_classes=5,
            num_layers=6,
            model_name=model_name
        )
        
        # Checkpoint laden
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        # Metriken aus Training
        self.val_f1 = checkpoint.get('val_f1', 0)
        self.epoch = checkpoint.get('epoch', 0)
        
        print(f"✅ Modell geladen!")
        print(f"   Validierungs-F1: {self.val_f1:.4f}")
        print(f"   Trainiert für: {self.epoch + 1} Epochen")
    
    @staticmethod
    def preprocess(text):
        """Text vorverarbeiten"""
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'<[^>]+>', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def predict(self, text, return_all_probs=False):
        """
        Vorhersage für einen einzelnen Text
        
        Args:
            text: Eingabetext
            return_all_probs: Wenn True, alle Wahrscheinlichkeiten zurückgeben
            
        Returns:
            dict mit 'label', 'label_de', 'confidence' und optional 'all_probs'
        """
        clean_text = self.preprocess(text)
        
        inputs = self.tokenizer(
            clean_text,
            return_tensors='pt',
            max_length=256,
            truncation=True,
            padding=True
        )
        
        input_ids = inputs['input_ids'].to(self.device)
        attention_mask = inputs['attention_mask'].to(self.device)
        
        with torch.no_grad():
            logits, capsule_outputs = self.model(input_ids, attention_mask)
            probs = F.softmax(logits, dim=-1)[0].cpu().numpy()
        
        pred_idx = probs.argmax()
        
        result = {
            'label': LABELS[pred_idx],
            'label_de': LABELS_DE[pred_idx],
            'confidence': float(probs[pred_idx])
        }
        
        if return_all_probs:
            result['all_probs'] = {
                LABELS_DE[i]: float(probs[i]) 
                for i in range(len(LABELS))
            }
        
        return result
    
    def predict_batch(self, texts, batch_size=16, show_progress=True):
        """
        Vorhersagen für mehrere Texte
        
        Args:
            texts: Liste von Texten
            batch_size: Batch-Größe
            show_progress: Progress-Bar anzeigen
            
        Returns:
            Liste von Vorhersage-Dicts
        """
        results = []
        
        iterator = range(0, len(texts), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Verarbeite")
        
        for i in iterator:
            batch_texts = texts[i:i+batch_size]
            clean_texts = [self.preprocess(t) for t in batch_texts]
            
            inputs = self.tokenizer(
                clean_texts,
                return_tensors='pt',
                max_length=256,
                truncation=True,
                padding=True
            )
            
            input_ids = inputs['input_ids'].to(self.device)
            attention_mask = inputs['attention_mask'].to(self.device)
            
            with torch.no_grad():
                logits, _ = self.model(input_ids, attention_mask)
                probs = F.softmax(logits, dim=-1).cpu().numpy()
            
            for j, text in enumerate(batch_texts):
                pred_idx = probs[j].argmax()
                results.append({
                    'text': text,
                    'label': LABELS[pred_idx],
                    'label_de': LABELS_DE[pred_idx],
                    'confidence': float(probs[j][pred_idx])
                })
        
        return results


def interactive_mode(classifier):
    """Interaktiver Modus für Echtzeit-Vorhersagen"""
    print("\n" + "=" * 60)
    print("  🧠 MentalRoBERTa-Caps - Interaktiver Modus")
    print("=" * 60)
    print("\nGib einen Text ein und drücke Enter für die Vorhersage.")
    print("Tippe 'quit' oder 'exit' zum Beenden.\n")
    
    while True:
        try:
            text = input("📝 Text: ").strip()
            
            if text.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Auf Wiedersehen!")
                break
            
            if not text:
                continue
            
            result = classifier.predict(text, return_all_probs=True)
            
            print(f"\n🎯 Vorhersage: {result['label_de']} ({result['confidence']*100:.1f}%)")
            print("   Alle Wahrscheinlichkeiten:")
            for label, prob in sorted(result['all_probs'].items(), 
                                      key=lambda x: x[1], reverse=True):
                bar = "█" * int(prob * 20)
                print(f"   {label:15s} {bar:20s} {prob*100:5.1f}%")
            print()
            
        except KeyboardInterrupt:
            print("\n\n👋 Auf Wiedersehen!")
            break


def main():
    parser = argparse.ArgumentParser(
        description="MentalRoBERTa-Caps Inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Beispiele:
  python inference.py --text "Ich fühle mich so leer..."
  python inference.py --input texte.json --output vorhersagen.json
  python inference.py --interactive
        """
    )
    
    # Eingabe
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--text', type=str, help='Einzelner Text zur Analyse')
    input_group.add_argument('--input', type=str, help='JSON-Datei mit Texten')
    input_group.add_argument('--interactive', action='store_true', 
                            help='Interaktiver Modus')
    
    # Ausgabe
    parser.add_argument('--output', type=str, help='Ausgabe-JSON (nur mit --input)')
    
    # Modell
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_model.pt',
                        help='Pfad zum trainierten Modell')
    parser.add_argument('--model_name', type=str, default='deepset/gbert-base',
                        help='HuggingFace Basis-Modell')
    
    # Optionen
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch-Größe für Batch-Verarbeitung')
    
    args = parser.parse_args()
    
    # Prüfe ob Checkpoint existiert
    if not Path(args.checkpoint).exists():
        print(f"❌ Fehler: Checkpoint nicht gefunden: {args.checkpoint}")
        print("\nBitte zuerst trainieren mit:")
        print("  python train.py --data german_augmented.json --epochs 15")
        return
    
    # Classifier laden
    classifier = MentalHealthClassifier(
        checkpoint_path=args.checkpoint,
        model_name=args.model_name
    )
    
    # Modus ausführen
    if args.interactive:
        interactive_mode(classifier)
        
    elif args.text:
        # Einzelner Text
        result = classifier.predict(args.text, return_all_probs=True)
        
        print(f"\n📝 Text: {args.text[:100]}{'...' if len(args.text) > 100 else ''}")
        print(f"\n🎯 Vorhersage: {result['label_de']} ({result['confidence']*100:.1f}%)")
        print("\n📊 Alle Wahrscheinlichkeiten:")
        for label, prob in sorted(result['all_probs'].items(), 
                                  key=lambda x: x[1], reverse=True):
            bar = "█" * int(prob * 20)
            print(f"   {label:15s} {bar:20s} {prob*100:5.1f}%")
        
    elif args.input:
        # Batch-Verarbeitung
        print(f"\n📂 Lade Eingabe: {args.input}")
        
        with open(args.input, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Texte extrahieren
        if isinstance(data, list):
            if isinstance(data[0], dict):
                texts = [item.get('text', item.get('content', '')) for item in data]
            else:
                texts = data
        else:
            print("❌ Unbekanntes Datenformat")
            return
        
        print(f"   {len(texts)} Texte gefunden")
        
        # Vorhersagen
        results = classifier.predict_batch(texts, batch_size=args.batch_size)
        
        # Statistiken
        label_counts = {}
        for r in results:
            label_counts[r['label_de']] = label_counts.get(r['label_de'], 0) + 1
        
        print("\n📊 Verteilung:")
        for label, count in sorted(label_counts.items(), key=lambda x: x[1], reverse=True):
            pct = count / len(results) * 100
            bar = "█" * int(pct / 2)
            print(f"   {label:15s} {bar:25s} {count:4d} ({pct:5.1f}%)")
        
        # Speichern
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"\n💾 Ergebnisse gespeichert: {args.output}")
        else:
            # Ausgabe auf Konsole
            print("\n📝 Ergebnisse:")
            for r in results[:10]:  # Erste 10
                print(f"   {r['label_de']:12s} ({r['confidence']*100:5.1f}%) | {r['text'][:50]}...")
            
            if len(results) > 10:
                print(f"   ... und {len(results) - 10} weitere")


if __name__ == "__main__":
    main()
