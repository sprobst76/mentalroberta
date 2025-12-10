"""
MentalRoBERTa-Caps Demo-Anwendung (mit trainiertem Modell)
Interaktive Demo für Mental-Health-Textklassifikation

Ausführen mit: streamlit run demo_app.py
"""

import streamlit as st
import torch
import torch.nn.functional as F
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import re
from pathlib import Path
from model import MentalRoBERTaCaps

# Seiten-Konfiguration
st.set_page_config(
    page_title="MentalRoBERTa-Caps Demo",
    page_icon="🧠",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        background-color: #f0f2f6;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Labels und Farben (Deutsch)
LABELS = ['depression', 'anxiety', 'bipolar', 'suicidewatch', 'offmychest']
LABELS_DE = ['Depression', 'Angst', 'Bipolar', 'Suizidalität', 'Ventil']
COLORS = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A']

# Konfiguration
DEFAULT_MODEL = "deepset/gbert-base"
CHECKPOINT_PATH = "checkpoints/best_model.pt"


@st.cache_resource
def load_trained_model():
    """Lade das trainierte Modell"""
    try:
        from transformers import AutoTokenizer
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Prüfe ob Checkpoint existiert
        checkpoint_path = Path(CHECKPOINT_PATH)
        
        if checkpoint_path.exists():
            # Trainiertes Modell laden
            
            # Tokenizer laden
            tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL)
            
            # Modell initialisieren
            model = MentalRoBERTaCaps(
                num_classes=5, 
                num_layers=6, 
                model_name=DEFAULT_MODEL
            )
            
            # Checkpoint laden
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            
            val_f1 = checkpoint.get('val_f1', 0)
            epoch = checkpoint.get('epoch', 0)
            
            model.to(device)
            model.eval()
            
            return model, tokenizer, device, True, val_f1, epoch
        else:
            # Kein Checkpoint - untrainiertes Modell
            
            tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL)
            model = MentalRoBERTaCaps(num_classes=5, num_layers=6, model_name=DEFAULT_MODEL)
            model.to(device)
            model.eval()
            
            return model, tokenizer, device, False, 0, 0
            
    except Exception as e:
        st.error(f"Fehler beim Laden des Modells: {e}")
        return None, None, None, False, 0, 0


def preprocess_text(text):
    """Text bereinigen und vorverarbeiten"""
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def predict(text, model, tokenizer, device):
    """Vorhersage für Eingabetext"""
    clean_text = preprocess_text(text)
    
    inputs = tokenizer(
        clean_text,
        return_tensors='pt',
        max_length=256,
        truncation=True,
        padding=True
    )
    
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)
    
    with torch.no_grad():
        logits, capsule_outputs = model(input_ids, attention_mask)
        probs = F.softmax(logits, dim=-1)[0]
        caps_lengths = model.get_capsule_lengths(capsule_outputs)[0]
    
    return probs.cpu().numpy(), caps_lengths.cpu().numpy()


def create_probability_chart(probs):
    """Erstelle Balkendiagramm für Vorhersage-Wahrscheinlichkeiten"""
    df = pd.DataFrame({
        'Kategorie': LABELS_DE,
        'Wahrscheinlichkeit': probs * 100
    })
    
    fig = px.bar(
        df, 
        x='Wahrscheinlichkeit', 
        y='Kategorie',
        orientation='h',
        color='Kategorie',
        color_discrete_sequence=COLORS,
        title='Vorhersage-Wahrscheinlichkeiten'
    )
    
    fig.update_layout(
        showlegend=False,
        xaxis_title='Wahrscheinlichkeit (%)',
        yaxis_title='',
        height=300
    )
    
    fig.update_traces(texttemplate='%{x:.1f}%', textposition='outside')
    
    return fig


def create_capsule_viz(caps_lengths):
    """Erstelle Visualisierung der Capsule-Aktivierungen"""
    fig = go.Figure(data=[
        go.Bar(
            x=LABELS_DE,
            y=caps_lengths,
            marker_color=COLORS,
            text=[f'{v:.3f}' for v in caps_lengths],
            textposition='outside'
        )
    ])
    
    fig.update_layout(
        title='Capsule-Vektorlängen (Klassen-Konfidenz)',
        xaxis_title='Kategorie',
        yaxis_title='Capsule-Länge',
        height=300
    )
    
    return fig


def main():
    st.markdown('<div class="main-header">🧠 MentalRoBERTa-Caps</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Capsule-Enhanced Transformer für Mental-Health-Klassifikation<br>'
        '<small>Basierend auf Wagay et al. (2025) | Deutsche Version</small></div>', 
        unsafe_allow_html=True
    )
    
    # Modell laden
    with st.spinner("Lade Modell..."):
        model, tokenizer, device, is_trained, val_f1, epoch = load_trained_model()
    
    if model is None:
        st.error("Modell konnte nicht geladen werden. Bitte Installation prüfen.")
        st.stop()
    
    # Status anzeigen
    if is_trained:
        st.success(f"✅ Trainiertes Modell geladen (Epoche {epoch+1}, Val-F1: {val_f1:.2%}) auf {device.upper()}")
    else:
        st.warning(f"⚠️ Untrainiertes Modell geladen auf {device.upper()}")
    
    # Sidebar mit Informationen
    with st.sidebar:
        st.header("ℹ️ Über das Modell")
        st.markdown("""
        **MentalRoBERTa-Caps** ist ein Hybrid-Modell:
        
        - 🔤 **German BERT** (6 Layer) - Deutsches Sprachmodell von deepset
        - 🧩 **Capsule Network** - Hierarchisches Feature-Learning mit Dynamic Routing
        
        **Kategorien:**
        - 😔 Depression
        - 😰 Angst  
        - 🔄 Bipolar
        - ⚠️ Suizidalität
        - 💭 Ventil (OffMyChest)
        
        **Paper:** [PMC12284574](https://pmc.ncbi.nlm.nih.gov/articles/PMC12284574/)
        """)
        
        st.header("⚙️ Modell-Info")
        total_params = sum(p.numel() for p in model.parameters())
        st.metric("Parameter", f"{total_params/1e6:.1f}M")
        st.metric("Gerät", device.upper())
        st.metric("Encoder-Layer", "6 von 12")
        
        if is_trained:
            st.success(f"✅ Trainiert (Epoche {epoch+1})")
            st.metric("Val-F1", f"{val_f1:.2%}")
        else:
            st.warning("❌ Nicht trainiert!")
    
    # Hauptinhalt
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📝 Text eingeben")
        
        # Beispieltexte (Deutsch)
        examples = {
            "-- Beispiel wählen --": "",
            "😔 Depression": "Ich fühle mich seit Wochen so leer. Nichts macht mir mehr Freude, selbst Dinge, die ich früher geliebt habe. Ich liege nur noch im Bett und finde keine Motivation für irgendetwas.",
            "😰 Angst": "Mein Herz rast und ich bekomme keine Luft. Ich habe ständig Angst, dass etwas Schlimmes passiert. Die Panikattacken kommen aus dem Nichts und ich kann sie nicht kontrollieren.",
            "🔄 Bipolar": "Letzte Woche war ich voller Energie und habe kaum geschlafen. Ich habe drei Projekte angefangen und mich unbesiegbar gefühlt. Jetzt komme ich kaum aus dem Bett.",
            "⚠️ Suizidalität": "Ich habe aufgehört zu kämpfen. Es ist einfach zu viel. Ich will nicht mehr aufwachen. Der Schmerz soll einfach aufhören.",
            "💭 Ventil": "Ich muss das einfach mal loswerden. Mein Chef ist unerträglich und ich kann nicht kündigen, weil ich das Geld brauche. Die Situation frisst mich auf.",
        }
        
        selected_example = st.selectbox("📋 Beispiel laden:", list(examples.keys()))
        
        default_text = examples.get(selected_example, "")
        
        input_text = st.text_area(
            "Text zur Analyse eingeben:",
            value=default_text,
            height=200,
            placeholder="Deutschen Text hier eingeben oder einfügen zur Mental-Health-Klassifikation..."
        )
        
        analyze_button = st.button("🔍 Text analysieren", type="primary", use_container_width=True)
    
    with col2:
        st.header("📊 Ergebnisse")
        
        if analyze_button and input_text.strip():
            with st.spinner("Analysiere..."):
                probs, caps_lengths = predict(input_text, model, tokenizer, device)
            
            # Top-Vorhersage
            top_idx = probs.argmax()
            top_label = LABELS_DE[top_idx]
            top_prob = probs[top_idx] * 100
            
            st.markdown(f"""
            <div class="prediction-box">
                <h3>🎯 Vorhersage: <span style="color: {COLORS[top_idx]}">{top_label}</span></h3>
                <h2>{top_prob:.1f}%</h2>
            </div>
            """, unsafe_allow_html=True)
            
            # Wahrscheinlichkeits-Diagramm
            prob_chart = create_probability_chart(probs)
            st.plotly_chart(prob_chart, use_container_width=True)
            
            # Capsule-Visualisierung
            with st.expander("🧩 Capsule-Aktivierungen anzeigen"):
                caps_chart = create_capsule_viz(caps_lengths)
                st.plotly_chart(caps_chart, use_container_width=True)
                st.caption("Die Capsule-Längen repräsentieren die Konfidenz des Modells für jede Klasse.")
            
        elif analyze_button:
            st.warning("⚠️ Bitte gib einen Text zur Analyse ein.")
        else:
            st.info("👈 Text eingeben und 'Text analysieren' klicken um Ergebnisse zu sehen.")
            
            if not is_trained:
                st.warning("""
                **⚠️ Hinweis:** Dieses Modell ist noch nicht trainiert! 
                
                Für echte Vorhersagen muss das Modell erst trainiert werden:
                ```bash
                python train.py --data german_large.json --epochs 30
                ```
                """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #888;">
        <small>
        ⚠️ <b>Hinweis:</b> Dies ist eine Forschungs-Demo und sollte NICHT für klinische Diagnosen verwendet werden. 
        Wenn du oder jemand den du kennst mit psychischen Problemen kämpft, suche bitte professionelle Hilfe.<br><br>
        📞 <b>Telefonseelsorge (24/7):</b> 0800 111 0 111 oder 0800 111 0 222 (kostenlos)<br>
        🌐 <b>Online:</b> <a href="https://online.telefonseelsorge.de">online.telefonseelsorge.de</a>
        </small>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
