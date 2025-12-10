# MentalRoBERTa-Caps

🧠 **A Capsule-Enhanced Transformer Model for Mental Health Classification**

Based on the paper by Wagay et al. (2025):
> *"MentalRoBERTa-Caps: A capsule-enhanced transformer model for mental health classification"*
> [PMC12284574](https://pmc.ncbi.nlm.nih.gov/articles/PMC12284574/)

---

## 📋 Overview

MentalRoBERTa-Caps is a hybrid architecture that combines:

1. **MentalRoBERTa** (6 of 12 layers) - A domain-specific RoBERTa model pretrained on mental health Reddit data
2. **Capsule Network Layer** - For hierarchical feature learning with dynamic routing
3. **Classification Head** - For final mental health category prediction

### Key Features

- ✅ **Efficient**: Uses only 6 encoder layers (~125M parameters vs 355M for full model)
- ✅ **Interpretable**: Capsule vector lengths represent class confidence
- ✅ **Accurate**: Achieves competitive F1 scores on mental health benchmarks
- ✅ **Fast Inference**: Suitable for real-time applications

### Supported Categories

| Dataset | Categories |
|---------|-----------|
| SWMH | Depression, Anxiety, Bipolar, SuicideWatch, OffMyChest |
| Dreaddit | Stressed, Not Stressed |
| SAD | School, Financial, Family, Social, Work, Health, Emotional, Decision, Other |

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install torch transformers streamlit plotly pandas numpy
```

### 2. Run Quick Test

```bash
python -m mentalroberta.tools.quick_test
```

### 3. Start Interactive Demo

```bash
streamlit run mentalroberta/apps/demo_app.py
```

### 4. Export to ONNX (optional, for client/offline inference)

```bash
python -m mentalroberta.tools.export_onnx \
  --checkpoint checkpoints/best_model.pt \
  --output checkpoints/model.onnx \
  --quantize
```

In the Streamlit app, pick the inference backend in the sidebar: PyTorch (server), ONNX (server CPU), or download the ONNX model for client-side use with `onnxruntime-web`.

---

## 🏗️ Architecture

```
Input Text
    │
    ▼
┌───────────────────────┐
│   Preprocessing       │  ← URL, mention, HTML removal
└───────────────────────┘
    │
    ▼
┌───────────────────────┐
│   WordPiece Tokenizer │  ← HuggingFace Transformers
└───────────────────────┘
    │
    ▼
┌───────────────────────┐
│   MentalRoBERTa       │  ← 6 Transformer layers
│   (768-dim output)    │
└───────────────────────┘
    │ [CLS] token
    ▼
┌───────────────────────┐
│   Primary Capsules    │  ← 8 capsules, 16-dim each
└───────────────────────┘
    │ Dynamic Routing
    ▼
┌───────────────────────┐
│   Class Capsules      │  ← 5 capsules (one per class)
└───────────────────────┘
    │
    ▼
┌───────────────────────┐
│   Linear Classifier   │  ← Softmax output
└───────────────────────┘
    │
    ▼
  Prediction
```

### Key Equations (from paper)

**Squash Function** (Eq. 12):
```
squash(s) = (||s||² / (1 + ||s||²)) * (s / ||s||)
```

**Dynamic Routing** (Eq. 14-15):
```
c_ij = softmax(b_ij)
s_j = Σ c_ij * u_hat_ij
v_j = squash(s_j)
```

---

## 📊 Performance (from paper)

| Dataset | Recall | F1-Score |
|---------|--------|----------|
| SWMH | 71.72 | 71.68 |
| Dreaddit | 80.97 | 80.86 |
| SAD | 65.47 | 65.22 |

With data augmentation:
| Dataset | Recall | F1-Score |
|---------|--------|----------|
| SWMH | 71.91 | 72.08 |
| Dreaddit | 81.49 | 81.28 |
| SAD | 65.92 | 65.71 |

---

## 💻 Usage

### Python API

```python
from mentalroberta.model import MentalRoBERTaCaps, MentalRoBERTaCapsClassifier

# Option 1: Use the high-level classifier
classifier = MentalRoBERTaCapsClassifier(num_classes=5)
prediction, probabilities = classifier.predict("I feel so empty inside...")
print(f"Prediction: {prediction}")
print(f"Probabilities: {probabilities}")

# Option 2: Use the model directly
model = MentalRoBERTaCaps(
    num_classes=5,
    num_layers=6,
    num_primary_caps=8,
    primary_cap_dim=16,
    class_cap_dim=16,
    num_routing_iterations=3
)

# Forward pass
logits, capsule_outputs = model(input_ids, attention_mask)
capsule_lengths = model.get_capsule_lengths(capsule_outputs)
```

### Training (Custom)

```python
import torch
from torch.optim import AdamW
from mentalroberta.model import MentalRoBERTaCaps

# Initialize
model = MentalRoBERTaCaps(num_classes=5)
optimizer = AdamW(model.parameters(), lr=2e-5)
criterion = torch.nn.CrossEntropyLoss()

# Training loop
for batch in dataloader:
    optimizer.zero_grad()
    logits, _ = model(batch['input_ids'], batch['attention_mask'])
    loss = criterion(logits, batch['labels'])
    loss.backward()
    optimizer.step()
```

---

## 📁 File Structure

```
mentalroberta/               # Package
├── model.py                 # Core model implementation
├── inference.py             # Inference CLI utilities
├── apps/
│   └── demo_app.py          # Streamlit interactive demo
├── tools/
│   └── quick_test.py        # Quick architecture test
└── training/
    ├── train.py             # Training loop and dataset prep
    ├── augment_data.py      # English data augmentation
    ├── augment_german.py    # German data augmentation
    ├── download_dataset.py  # Dataset downloader (full)
    └── download_simple.py   # Minimal dataset downloader

data/                        # Sample datasets (JSON/CSV)
checkpoints/                 # Saved models (git-ignored)
requirements.txt             # Python dependencies
README.md                    # This file
```

---

## ⚠️ Important Notes

1. **This is an UNTRAINED model** - The weights are random. For real predictions, you need to:
   - Download/prepare training data (SWMH, Dreaddit, SAD datasets)
   - Train the model on the data
   - Or wait for the official pretrained weights from the authors

2. **Not for clinical use** - This is a research demo. Do not use for actual mental health diagnosis.

3. **Official code pending** - The paper states "Code link will be provided after acceptance."

---

## 🔗 Resources

- **Paper**: [PMC12284574](https://pmc.ncbi.nlm.nih.gov/articles/PMC12284574/)
- **MentalRoBERTa Base Model**: [HuggingFace](https://huggingface.co/mental/mental-roberta-base)
- **Original MentalBERT Paper**: [ACL Anthology](https://aclanthology.org/2022.lrec-1.778/)

---

## 📞 Crisis Resources

If you or someone you know is struggling with mental health:

- **Germany**: 0800 111 0 111 (Telefonseelsorge)
- **International**: [findahelpline.com](https://findahelpline.com)
- **US**: 988 (Suicide & Crisis Lifeline)

---

## 📄 Citation

```bibtex
@article{wagay2025mentalrobertacaps,
  title={MentalRoBERTa-Caps: A Capsule-Enhanced Transformer Model for Mental Health Classification},
  author={Wagay, Faheem Ahmad and Jahiruddin and Altaf, Yasir},
  journal={MethodsX},
  year={2025},
  publisher={Elsevier}
}
```

---

*Implementation by Claude based on the published paper. Not affiliated with the original authors.*
