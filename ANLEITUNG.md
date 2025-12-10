# 🧠 MentalRoBERTa-Caps - Vollständige Anleitung

## Inhaltsverzeichnis

1. [Installation](#1-installation)
2. [Projektstruktur](#2-projektstruktur)
3. [Daten vorbereiten](#3-daten-vorbereiten)
4. [Training](#4-training)
5. [Trainiertes Modell laden](#5-trainiertes-modell-laden)
6. [Inference auf neuen Texten](#6-inference-auf-neuen-texten)
7. [Demo-App starten](#7-demo-app-starten)
8. [Batch-Verarbeitung](#8-batch-verarbeitung)
9. [Modell evaluieren](#9-modell-evaluieren)
10. [Modell exportieren](#10-modell-exportieren)
11. [Tipps & Troubleshooting](#11-tipps--troubleshooting)

---

## 1. Installation

### 1.1 Virtuelle Umgebung erstellen (empfohlen)

```bash
# Mit conda
conda create -n mental python=3.10
conda activate mental

# ODER mit venv
python -m venv mental_env
# Windows:
mental_env\Scripts\activate
# Linux/Mac:
source mental_env/bin/activate
```

### 1.2 Dependencies installieren

```bash
pip install torch torchvision torchaudio
pip install transformers
pip install streamlit plotly pandas numpy
pip install scikit-learn tqdm
```

### 1.3 GPU-Unterstützung (optional aber empfohlen)

```bash
# Prüfe ob CUDA verfügbar ist
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

---

## 2. Projektstruktur

```
MentalRoBERTa/
├── model.py                       # Modell-Architektur
├── inference.py                   # Inference-Script
├── training/                      # Training & Daten-Tools
│   ├── train.py                   # Training-Script
│   ├── augment_german.py          # Daten-Augmentierung
│   └── download_dataset.py        # Download-Helfer
├── apps/
│   └── demo_app.py                # Streamlit Demo
├── tools/
│   └── quick_test.py              # Schneller Architektur-Test
│
├── data/                          # Beispieldaten (JSON/CSV)
│   ├── german_data.json           # Basis-Trainingsdaten (175 Samples)
│   └── german_augmented.json      # Augmentierte Daten (1050 Samples)
│
├── checkpoints/                   # Wird beim Training erstellt
│   ├── best_model.pt              # Bestes Modell
│   └── test_report.json           # Evaluations-Report
│
└── outputs/                       # Für Vorhersagen
    └── predictions.json
```

---

## 3. Daten vorbereiten

### 3.1 Datenformat

Die Trainingsdaten müssen als JSON-Liste vorliegen:

```json
[
  {
    "text": "Ich fühle mich seit Wochen so leer...",
    "label": "depression"
  },
  {
    "text": "Mein Herz rast und ich bekomme keine Luft...",
    "label": "anxiety"
  }
]
```

**Verfügbare Labels:**
- `depression` - Depressive Symptome
- `anxiety` - Angstsymptome
- `bipolar` - Bipolare Symptome
- `suicidewatch` - Suizidale Gedanken
- `offmychest` - Allgemeines Ventil/Venting

### 3.2 Daten augmentieren (mehr Trainingsdaten erzeugen)

```bash
# Standard: 5x mehr Daten, balanciert
python augment_german.py --input german_data.json \
                         --output german_augmented.json \
                         --factor 5 \
                         --balance

# Großer Datensatz: 10x mehr Daten
python augment_german.py --input german_data.json \
                         --output german_large.json \
                         --factor 10 \
                         --balance

# Mit Beispiel-Ausgabe
python augment_german.py --input german_data.json \
                         --output german_augmented.json \
                         --factor 5 \
                         --balance \
                         --show_examples
```

### 3.3 Eigene Daten hinzufügen

Öffne `german_data.json` und füge neue Einträge hinzu:

```json
{
  "text": "Dein neuer Text hier...",
  "label": "depression"
}
```

---

## 4. Training

### 4.1 Basis-Training starten

```bash
python -m mentalroberta.training.train --data data/german_augmented.json --epochs 15 --batch_size 16
```

### 4.2 Alle Training-Parameter

```bash
python -m mentalroberta.training.train \
    --data data/german_augmented.json \
    --output checkpoints \
    --model_name deepset/gbert-base \
    --epochs 15 \
    --batch_size 16 \
    --learning_rate 2e-5 \
    --max_length 256 \
    --num_layers 6 \
    --num_primary_caps 8 \
    --primary_cap_dim 16 \
    --class_cap_dim 16 \
    --routing_iterations 3
```

**Parameter erklärt:**

| Parameter | Default | Beschreibung |
|-----------|---------|--------------|
| `--data` | - | Pfad zur JSON-Datei mit Trainingsdaten |
| `--output` | `checkpoints` | Ausgabe-Ordner für Modell |
| `--model_name` | `deepset/gbert-base` | HuggingFace Basis-Modell |
| `--epochs` | 10 | Anzahl Trainings-Epochen |
| `--batch_size` | 16 | Batch-Größe (kleiner bei wenig RAM) |
| `--learning_rate` | 2e-5 | Lernrate |
| `--max_length` | 256 | Max. Token-Länge |
| `--num_layers` | 6 | Anzahl Encoder-Layer (von 12) |

### 4.3 Training-Ausgabe verstehen

```
📈 Epoch 1/15
Training: 100%|██████████| 53/53 [01:23<00:00]
   Train Loss: 1.5234, Train F1: 0.2145
   Val Loss: 1.4521, Val F1: 0.2834
   💾 Saved best model (F1: 0.2834)

📈 Epoch 15/15
   Train Loss: 0.3421, Train F1: 0.8234
   Val Loss: 0.4123, Val F1: 0.7856
```

**Wichtige Metriken:**
- **Loss**: Sollte sinken (niedriger = besser)
- **F1-Score**: Sollte steigen (0-1, höher = besser)
- Das beste Modell wird automatisch gespeichert

---

## 5. Trainiertes Modell laden

### 5.1 In Python laden

```python
import torch
from mentalroberta.model import MentalRoBERTaCaps
from transformers import AutoTokenizer

# Konfiguration
MODEL_NAME = "deepset/gbert-base"
CHECKPOINT_PATH = "checkpoints/best_model.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Labels
LABELS = ['depression', 'anxiety', 'bipolar', 'suicidewatch', 'offmychest']
LABELS_DE = ['Depression', 'Angst', 'Bipolar', 'Suizidalität', 'Ventil']

# Tokenizer laden
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Modell initialisieren
model = MentalRoBERTaCaps(
    num_classes=5,
    num_layers=6,
    model_name=MODEL_NAME
)

# Trainierte Gewichte laden
checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(DEVICE)
model.eval()

print(f"✅ Modell geladen von: {CHECKPOINT_PATH}")
print(f"   Validierungs-F1: {checkpoint['val_f1']:.4f}")
print(f"   Epoche: {checkpoint['epoch'] + 1}")
```

---

## 6. Inference auf neuen Texten

### 6.1 Einzelner Text

```python
import torch.nn.functional as F
import re

def preprocess(text):
    """Text vorverarbeiten"""
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def predict(text, model, tokenizer, device):
    """Vorhersage für einen Text"""
    # Vorverarbeitung
    clean_text = preprocess(text)
    
    # Tokenisierung
    inputs = tokenizer(
        clean_text,
        return_tensors='pt',
        max_length=256,
        truncation=True,
        padding=True
    )
    
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)
    
    # Vorhersage
    with torch.no_grad():
        logits, capsule_outputs = model(input_ids, attention_mask)
        probs = F.softmax(logits, dim=-1)[0]
    
    # Ergebnis
    pred_idx = probs.argmax().item()
    confidence = probs[pred_idx].item()
    
    return {
        'label': LABELS[pred_idx],
        'label_de': LABELS_DE[pred_idx],
        'confidence': confidence,
        'all_probabilities': {
            LABELS_DE[i]: probs[i].item() 
            for i in range(len(LABELS))
        }
    }

# Beispiel-Verwendung
text = "Ich fühle mich so leer und hoffnungslos. Nichts macht mir mehr Freude."
result = predict(text, model, tokenizer, DEVICE)

print(f"Text: {text[:50]}...")
print(f"Vorhersage: {result['label_de']} ({result['confidence']*100:.1f}%)")
print(f"Alle Wahrscheinlichkeiten:")
for label, prob in result['all_probabilities'].items():
    print(f"  {label}: {prob*100:.1f}%")
```

### 6.2 Mehrere Texte (Batch)

```python
def predict_batch(texts, model, tokenizer, device, batch_size=16):
    """Vorhersagen für mehrere Texte"""
    results = []
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        clean_texts = [preprocess(t) for t in batch_texts]
        
        inputs = tokenizer(
            clean_texts,
            return_tensors='pt',
            max_length=256,
            truncation=True,
            padding=True
        )
        
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)
        
        with torch.no_grad():
            logits, _ = model(input_ids, attention_mask)
            probs = F.softmax(logits, dim=-1)
        
        for j, text in enumerate(batch_texts):
            pred_idx = probs[j].argmax().item()
            results.append({
                'text': text,
                'label': LABELS[pred_idx],
                'label_de': LABELS_DE[pred_idx],
                'confidence': probs[j][pred_idx].item()
            })
    
    return results

# Beispiel
texts = [
    "Ich habe ständig Angst, dass etwas Schlimmes passiert.",
    "Die Stimmungsschwankungen machen mich fertig.",
    "Ich muss das einfach mal loswerden."
]

results = predict_batch(texts, model, tokenizer, DEVICE)
for r in results:
    print(f"{r['label_de']}: {r['text'][:40]}... ({r['confidence']*100:.1f}%)")
```

---

## 7. Demo-App starten

### 7.1 Mit untrainiertem Modell (zum Testen)

```bash
streamlit run demo_app.py
```

### 7.2 Mit trainiertem Modell

Erstelle eine neue Datei `demo_trained.py`:

```python
"""
Demo mit trainiertem Modell
"""
import streamlit as st
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
import re
from mentalroberta.model import MentalRoBERTaCaps

# Konfiguration
MODEL_NAME = "deepset/gbert-base"
CHECKPOINT_PATH = "checkpoints/best_model.pt"
LABELS_DE = ['Depression', 'Angst', 'Bipolar', 'Suizidalität', 'Ventil']

st.set_page_config(page_title="Mental Health Classifier", page_icon="🧠")

@st.cache_resource
def load_trained_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    model = MentalRoBERTaCaps(num_classes=5, num_layers=6, model_name=MODEL_NAME)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model, tokenizer, device, checkpoint['val_f1']

def predict(text, model, tokenizer, device):
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    inputs = tokenizer(text, return_tensors='pt', max_length=256, 
                       truncation=True, padding=True)
    
    with torch.no_grad():
        logits, _ = model(inputs['input_ids'].to(device), 
                         inputs['attention_mask'].to(device))
        probs = F.softmax(logits, dim=-1)[0].cpu().numpy()
    
    return probs

# App
st.title("🧠 Mental Health Text Classifier")

model, tokenizer, device, val_f1 = load_trained_model()
st.success(f"✅ Trainiertes Modell geladen (Val-F1: {val_f1:.2%})")

text = st.text_area("Text eingeben:", height=150)

if st.button("Analysieren"):
    if text.strip():
        probs = predict(text, model, tokenizer, device)
        
        st.subheader("Ergebnisse:")
        for i, (label, prob) in enumerate(zip(LABELS_DE, probs)):
            st.progress(float(prob), text=f"{label}: {prob*100:.1f}%")
        
        top_idx = probs.argmax()
        st.info(f"**Vorhersage: {LABELS_DE[top_idx]}** ({probs[top_idx]*100:.1f}%)")
```

Dann starten:

```bash
streamlit run demo_trained.py
```

---

## 8. Batch-Verarbeitung

### 8.1 CSV-Datei verarbeiten

```python
import pandas as pd
import json

# CSV laden
df = pd.read_csv("meine_texte.csv")  # Spalte "text" erwartet

# Vorhersagen
predictions = predict_batch(df['text'].tolist(), model, tokenizer, DEVICE)

# Zu DataFrame hinzufügen
df['predicted_label'] = [p['label_de'] for p in predictions]
df['confidence'] = [p['confidence'] for p in predictions]

# Speichern
df.to_csv("meine_texte_mit_vorhersagen.csv", index=False)
print(f"✅ {len(df)} Texte verarbeitet und gespeichert.")
```

### 8.2 JSON-Datei verarbeiten

```python
# JSON laden
with open("eingabe.json", "r", encoding="utf-8") as f:
    data = json.load(f)

texts = [item['text'] for item in data]

# Vorhersagen
predictions = predict_batch(texts, model, tokenizer, DEVICE)

# Ergebnisse hinzufügen
for item, pred in zip(data, predictions):
    item['predicted_label'] = pred['label_de']
    item['confidence'] = pred['confidence']

# Speichern
with open("ausgabe.json", "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)
```

---

## 9. Modell evaluieren

### 9.1 Auf Test-Daten evaluieren

```python
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# Test-Daten laden
with open("test_data.json", "r", encoding="utf-8") as f:
    test_data = json.load(f)

# Vorhersagen machen
texts = [item['text'] for item in test_data]
true_labels = [item['label'] for item in test_data]

predictions = predict_batch(texts, model, tokenizer, DEVICE)
pred_labels = [p['label'] for p in predictions]

# Classification Report
print("\n📊 Classification Report:")
print(classification_report(true_labels, pred_labels, target_names=LABELS))

# Confusion Matrix
print("\n🔢 Confusion Matrix:")
cm = confusion_matrix(true_labels, pred_labels, labels=LABELS)
print(pd.DataFrame(cm, index=LABELS, columns=LABELS))
```

### 9.2 Test-Report vom Training

Nach dem Training findest du `checkpoints/test_report.json`:

```python
import json

with open("checkpoints/test_report.json", "r") as f:
    report = json.load(f)

print(f"Macro F1: {report['macro avg']['f1-score']:.4f}")
print(f"Macro Recall: {report['macro avg']['recall']:.4f}")

print("\nPer-Klasse:")
for label in LABELS:
    if label in report:
        print(f"  {label}: F1={report[label]['f1-score']:.3f}")
```

---

## 10. Modell exportieren

### 10.1 Für Produktion speichern

```python
# Nur Modell-Gewichte (kleiner)
torch.save(model.state_dict(), "model_weights.pt")

# Komplettes Checkpoint (empfohlen)
torch.save({
    'model_state_dict': model.state_dict(),
    'model_config': {
        'num_classes': 5,
        'num_layers': 6,
        'model_name': MODEL_NAME,
        'num_primary_caps': 8,
        'primary_cap_dim': 16,
        'class_cap_dim': 16
    },
    'labels': LABELS,
    'labels_de': LABELS_DE
}, "model_production.pt")
```

### 10.2 Modell wieder laden

```python
# Checkpoint laden
checkpoint = torch.load("model_production.pt", map_location=DEVICE)

# Modell mit gespeicherter Config erstellen
config = checkpoint['model_config']
model = MentalRoBERTaCaps(
    num_classes=config['num_classes'],
    num_layers=config['num_layers'],
    model_name=config['model_name'],
    num_primary_caps=config['num_primary_caps'],
    primary_cap_dim=config['primary_cap_dim'],
    class_cap_dim=config['class_cap_dim']
)

model.load_state_dict(checkpoint['model_state_dict'])
model.to(DEVICE)
model.eval()

labels = checkpoint['labels']
labels_de = checkpoint['labels_de']
```

### 10.3 ONNX Export (für Deployment)

```python
import torch.onnx

# Dummy-Input für Export
dummy_input_ids = torch.randint(0, 30000, (1, 128)).to(DEVICE)
dummy_attention_mask = torch.ones(1, 128).to(DEVICE)

# Export
torch.onnx.export(
    model,
    (dummy_input_ids, dummy_attention_mask),
    "model.onnx",
    input_names=['input_ids', 'attention_mask'],
    output_names=['logits', 'capsule_outputs'],
    dynamic_axes={
        'input_ids': {0: 'batch_size', 1: 'sequence'},
        'attention_mask': {0: 'batch_size', 1: 'sequence'}
    }
)
print("✅ ONNX Export erfolgreich!")
```

---

## 11. Tipps & Troubleshooting

### 11.1 Häufige Fehler

**"CUDA out of memory"**
```bash
# Kleinere Batch-Größe verwenden
python -m mentalroberta.training.train --data data/german_augmented.json --batch_size 8
```

**"Model not found"**
```bash
# Prüfe ob Checkpoint existiert
ls checkpoints/
# Sollte zeigen: best_model.pt
```

**Schlechte Vorhersagen**
- Mehr Trainingsdaten verwenden (Faktor erhöhen)
- Mehr Epochen trainieren
- Lernrate anpassen

### 11.2 Performance verbessern

```bash
# Mehr Daten
python augment_german.py --input german_data.json --output german_xlarge.json --factor 20 --balance

# Längeres Training
python -m mentalroberta.training.train --data data/german_xlarge.json --epochs 30 --batch_size 16

# Andere Lernrate
python -m mentalroberta.training.train --data data/german_augmented.json --learning_rate 1e-5 --epochs 20
```

### 11.3 Modell-Auswahl

| Modell | Geschwindigkeit | Qualität | RAM |
|--------|-----------------|----------|-----|
| `deepset/gbert-base` | ⭐⭐⭐ | ⭐⭐⭐ | ~2GB |
| `bert-base-german-cased` | ⭐⭐⭐ | ⭐⭐⭐ | ~2GB |
| `xlm-roberta-base` | ⭐⭐ | ⭐⭐⭐⭐ | ~3GB |
| `deepset/gbert-large` | ⭐ | ⭐⭐⭐⭐⭐ | ~4GB |

### 11.4 Für Psychotherapie-Analyse

Für echte Therapie-Transkripte empfehle ich:
1. Eigene Trainingsdaten aus anonymisierten Transkripten erstellen
2. Labels an deinen Use-Case anpassen (z.B. spezifische Symptome)
3. Längere `max_length` verwenden (512) für längere Aussagen
4. DSGVO-Compliance beachten!

---

## 📞 Support

Bei Fragen zum Paper:
- **Paper:** [PMC12284574](https://pmc.ncbi.nlm.nih.gov/articles/PMC12284574/)
- **MentalBERT:** [HuggingFace](https://huggingface.co/mental)

Bei technischen Fragen:
- Transformers Docs: https://huggingface.co/docs/transformers
- PyTorch Docs: https://pytorch.org/docs

---

## ⚠️ Wichtiger Hinweis

Dieses Modell ist für **Forschungszwecke** gedacht und sollte **NICHT** für klinische Diagnosen verwendet werden. Bei psychischen Problemen bitte professionelle Hilfe suchen:

📞 **Telefonseelsorge:** 0800 111 0 111 (kostenlos, 24/7)
🌐 **Online:** https://online.telefonseelsorge.de
