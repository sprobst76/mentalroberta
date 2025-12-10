"""
MentalRoBERTa-Caps Demo-Anwendung (mit trainiertem Modell)
Interaktive Demo für Mental-Health-Textklassifikation

Ausführen mit: streamlit run mentalroberta/apps/demo_app.py
"""

import json
import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
import sys

# Ensure repository root is on sys.path when running via streamlit
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import torch
import torch.nn.functional as F
import shutil

from mentalroberta.model import MentalRoBERTaCaps

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
BASE_MODEL_NAME = os.getenv("MENTALROBERTA_BASE_MODEL", DEFAULT_MODEL)
CHECKPOINT_PATH = Path("checkpoints/best_model.pt")
ONNX_PATH = Path("checkpoints/model.onnx")
ONNX_QUANT_PATH = Path("checkpoints/model.int8.onnx")
DEFAULT_BACKEND = os.getenv("MENTALROBERTA_BACKEND", "pytorch")
ACCESS_TOKEN = os.getenv("MENTALROBERTA_APP_TOKEN")
USAGE_LOG_PATH = Path(os.getenv("MENTALROBERTA_USAGE_LOG", "checkpoints/usage.log"))
BROWSER_ONNX_URL = os.getenv("MENTALROBERTA_BROWSER_ONNX_URL", "/static/model.onnx")


@st.cache_resource
def load_pytorch_model():
    """Lade das trainierte Modell (PyTorch-Backend)"""
    try:
        from transformers import AutoTokenizer

        device = "cuda" if torch.cuda.is_available() else "cpu"

        checkpoint_path = Path(CHECKPOINT_PATH)
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
        model = MentalRoBERTaCaps(num_classes=5, num_layers=6, model_name=BASE_MODEL_NAME)

        is_trained = False
        val_f1 = 0
        epoch = 0

        if checkpoint_path.exists():
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint["model_state_dict"])
            val_f1 = checkpoint.get("val_f1", 0)
            epoch = checkpoint.get("epoch", 0)
            is_trained = True

        model.to(device)
        model.eval()

        return model, tokenizer, device, is_trained, val_f1, epoch

    except Exception as e:
        st.error(f"Fehler beim Laden des PyTorch-Modells: {e}")
        return None, None, None, False, 0, 0


@st.cache_resource
def load_onnx_session(onnx_path: Path):
    """Lade ONNX-Session (CPU)."""
    try:
        import onnxruntime as ort
        from transformers import AutoTokenizer
    except Exception as exc:
        st.error(f"ONNX-Backend nicht verfügbar: {exc}")
        return None, None, None

    if not onnx_path.exists():
        st.error(f"ONNX-Modell nicht gefunden: {onnx_path}")
        return None, None, None

    sess_opts = ort.SessionOptions()
    providers = ["CPUExecutionProvider"]
    session = ort.InferenceSession(onnx_path.as_posix(), sess_options=sess_opts, providers=providers)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)

    return session, tokenizer, "cpu"


def preprocess_text(text):
    """Text bereinigen und vorverarbeiten"""
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def predict_pytorch(text, model, tokenizer, device):
    """Vorhersage mit PyTorch-Backend"""
    clean_text = preprocess_text(text)

    inputs = tokenizer(
        clean_text,
        return_tensors="pt",
        max_length=256,
        truncation=True,
        padding=True,
    )

    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    with torch.no_grad():
        logits, capsule_outputs = model(input_ids, attention_mask)
        probs = F.softmax(logits, dim=-1)[0]
        caps_lengths = model.get_capsule_lengths(capsule_outputs)[0]

    return probs.cpu().numpy(), caps_lengths.cpu().numpy()


def predict_onnx(text, session, tokenizer):
    """Vorhersage mit ONNX-Backend (CPU)"""
    clean_text = preprocess_text(text)
    inputs = tokenizer(
        clean_text,
        return_tensors="pt",
        max_length=256,
        truncation=True,
        padding=True,
    )
    ort_inputs = {
        "input_ids": inputs["input_ids"].numpy(),
        "attention_mask": inputs["attention_mask"].numpy(),
    }
    logits, capsule_outputs = session.run(None, ort_inputs)
    probs = torch.softmax(torch.from_numpy(logits[0]), dim=-1).numpy()
    caps_tensor = torch.from_numpy(capsule_outputs[0])
    caps_lengths = torch.sqrt((caps_tensor ** 2).sum(dim=-1)).numpy()
    return probs, caps_lengths


def get_session_id():
    if "session_id" not in st.session_state:
        st.session_state["session_id"] = uuid.uuid4().hex
    return st.session_state["session_id"]


def log_usage(event: str, backend: str, text_len: int, success: bool, is_trained: bool):
    if not USAGE_LOG_PATH:
        return
    try:
        USAGE_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "session": get_session_id(),
            "event": event,
            "backend": backend,
            "text_len": text_len,
            "success": success,
            "trained": is_trained,
        }
        with open(USAGE_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
    except Exception:
        # Don't break the UI due to logging issues
        pass


def ensure_static_onnx():
    """
    Copy ONNX model to .streamlit/static for browser consumption and return URL.
    """
    static_dir = ROOT_DIR / ".streamlit" / "static"
    static_dir.mkdir(parents=True, exist_ok=True)
    src = ONNX_QUANT_PATH if ONNX_QUANT_PATH.exists() else ONNX_PATH
    if not src.exists():
        return None
    dst = static_dir / "model.onnx"
    shutil.copy(src, dst)
    return BROWSER_ONNX_URL or "/static/model.onnx"


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


def render_browser_component(text: str, model_url: str):
    """Render a lightweight browser-side ONNX inference block."""
    labels_js = json.dumps(LABELS_DE)
    # Simple HTML/JS component leveraging onnxruntime-web + xenova tokenizers
    st.components.v1.html(
        f"""
        <div id="browser-out" style="padding:10px; background:#f7f9fc; border-radius:8px; border:1px solid #e0e7ef;">
            Lade Browser-Inferenz...
        </div>
        <script src="https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.min.js"></script>
        <script src="https://cdn.jsdelivr.net/npm/@xenova/transformers/dist/transformers.min.js"></script>
        <script>
        const TEXT = {json.dumps(text)};
        const MODEL_URL = "{model_url}";
        const TOKENIZER_MODEL = "{BASE_MODEL_NAME}";
        const LABELS = {labels_js};
        (async () => {{
            try {{
                const {{ AutoTokenizer }} = window.transformers;
                const tokenizer = await AutoTokenizer.from_pretrained(TOKENIZER_MODEL);
                const encoded = await tokenizer(TEXT, {{
                    padding: true,
                    truncation: true,
                    max_length: 256,
                    return_tensors: "np"
                }});
                const toBigInt = arr => BigInt64Array.from(arr.map(BigInt));
                const input_ids = new ort.Tensor('int64', toBigInt(encoded.input_ids.data), encoded.input_ids.dims);
                const attention_mask = new ort.Tensor('int64', toBigInt(encoded.attention_mask.data), encoded.attention_mask.dims);
                const session = await ort.InferenceSession.create(MODEL_URL);
                const outputs = await session.run({{input_ids, attention_mask}});
                const logits = Array.from(outputs.logits.data);
                const exps = logits.map(Math.exp);
                const sum = exps.reduce((a,b)=>a+b,0);
                const probs = exps.map(v => v / sum);
                const top = probs.indexOf(Math.max(...probs));
                const lines = probs.map((p,i)=>`${{LABELS[i]}}: ${{(p*100).toFixed(1)}}%`);
                document.getElementById('browser-out').innerHTML = `
                    <div><b>Vorhersage:</b> ${{LABELS[top]}} (${(probs[top]*100).toFixed(1)}%)</div>
                    <div style="margin-top:6px;">${{lines.join('<br>')}}</div>
                `;
            }} catch (err) {{
                document.getElementById('browser-out').innerText = 'Fehler im Browser-Inferenzpfad: ' + err;
            }}
        }})();
        </script>
        """,
        height=260,
    )


def main():
    # Optional Access Gate (simple shared token)
    if ACCESS_TOKEN:
        token = st.session_state.get("access_token") or st.query_params.get("token", [None])[0]
        if token != ACCESS_TOKEN:
            st.title("🔒 Zugriff beschränkt")
            token_input = st.text_input("Access Token eingeben", type="password")
            if st.button("Freischalten") and token_input == ACCESS_TOKEN:
                st.session_state["access_token"] = ACCESS_TOKEN
                st.experimental_rerun()
            st.stop()

    st.markdown('<div class="main-header">🧠 MentalRoBERTa-Caps</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Capsule-Enhanced Transformer für Mental-Health-Klassifikation<br>'
        '<small>Basierend auf Wagay et al. (2025) | Deutsche Version</small></div>', 
        unsafe_allow_html=True
    )
    
    backend_options = ["PyTorch (Server)", "ONNX (Server-CPU)", "ONNX (Client-Download)", "ONNX (Browser)"]
    backend = st.sidebar.selectbox(
        "Inference-Backend",
        backend_options,
        index={"pytorch": 0, "onnx": 1}.get(DEFAULT_BACKEND, 0),
    )

    # Backend laden
    if backend == "PyTorch (Server)":
        with st.spinner("Lade PyTorch-Modell..."):
            model, tokenizer, device, is_trained, val_f1, epoch = load_pytorch_model()
        backend_label = f"PyTorch auf {device.upper()}"
        onnx_session = None
    elif backend == "ONNX (Server-CPU)":
        with st.spinner("Lade ONNX-Modell..."):
            onnx_session, tokenizer, device = load_onnx_session(
                ONNX_QUANT_PATH if ONNX_QUANT_PATH.exists() else ONNX_PATH
            )
        model = None
        is_trained = ONNX_PATH.exists() or ONNX_QUANT_PATH.exists()
        val_f1 = None
        epoch = None
        backend_label = "ONNX (CPU)"
    elif backend == "ONNX (Client-Download)":
        model = None
        tokenizer = None
        device = "client"
        is_trained = ONNX_PATH.exists() or ONNX_QUANT_PATH.exists()
        val_f1 = None
        epoch = None
        backend_label = "Client-ONNX"
        onnx_session = None
    else:  # ONNX (Browser)
        model = None
        tokenizer = None
        device = "browser"
        is_trained = ONNX_PATH.exists() or ONNX_QUANT_PATH.exists()
        val_f1 = None
        epoch = None
        backend_label = "Browser-ONNX"
        onnx_session = None

    if backend == "ONNX (Browser)":
        model_url = ensure_static_onnx()
        if model_url is None:
            st.error("Kein ONNX-Modell gefunden. Bitte zuerst exportieren.")
            st.stop()
    else:
        model_url = None

    if backend == "ONNX (Server-CPU)" and onnx_session is None:
        st.error("ONNX-Backend nicht verfügbar. Bitte ONNX-Modell exportieren oder onnxruntime installieren.")
        st.stop()

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
        st.metric("Backend", backend_label)
        if model is not None:
            total_params = sum(p.numel() for p in model.parameters())
            st.metric("Parameter", f"{total_params/1e6:.1f}M")
            st.metric("Gerät", device.upper())
            st.metric("Encoder-Layer", "6 von 12")
        if is_trained and val_f1 is not None:
            st.success(f"✅ Trainiert (Epoche {epoch+1})")
            st.metric("Val-F1", f"{val_f1:.2%}")
        elif is_trained:
            st.success("✅ Trainiertes ONNX-Modell gefunden")
        else:
            st.warning("❌ Nicht trainiert!")
        if backend == "ONNX (Client-Download)":
            st.info("Dieses Backend führt keine Server-Inferenz aus. ONNX-Modell herunterladen und im Browser/Client verwenden.")
            if ONNX_PATH.exists():
                st.download_button(
                    "ONNX herunterladen",
                    ONNX_PATH.read_bytes(),
                    file_name=ONNX_PATH.name,
                    mime="application/octet-stream",
                )
            if ONNX_QUANT_PATH.exists():
                st.download_button(
                    "Quantisiertes ONNX herunterladen",
                    ONNX_QUANT_PATH.read_bytes(),
                    file_name=ONNX_QUANT_PATH.name,
                    mime="application/octet-stream",
                )
            st.markdown("""
            Beispiel-Client (onnxruntime-web):
            ```js
            import { InferenceSession } from 'onnxruntime-web';
            const session = await InferenceSession.create('model.onnx');
            // Tokenize Text mit z.B. @xenova/transformers, dann:
            const outputs = await session.run({input_ids, attention_mask});
            ```
            """)
        if backend == "ONNX (Browser)" and model_url:
            st.info(f"Browser-ONNX aktiv. Modell wird über {model_url} geladen (erfordert Internet/CORS-fähige Bereitstellung).")
    
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
            if backend == "ONNX (Browser)":
                if model_url:
                    log_usage(
                        event="analyze",
                        backend=backend,
                        text_len=len(input_text),
                        success=True,
                        is_trained=is_trained,
                    )
                    render_browser_component(input_text, model_url)
                else:
                    st.error("Kein Browser-ONNX-Modell verfügbar.")
            else:
                with st.spinner("Analysiere..."):
                    if backend == "PyTorch (Server)":
                        probs, caps_lengths = predict_pytorch(input_text, model, tokenizer, device)
                    elif backend == "ONNX (Server-CPU)":
                        probs, caps_lengths = predict_onnx(input_text, onnx_session, tokenizer)
                    else:
                        st.warning("Client-Modus aktiv: Bitte ONNX-Modell herunterladen und lokal im Browser/Client ausführen.")
                        probs, caps_lengths = None, None
                    log_usage(
                        event="analyze",
                        backend=backend,
                        text_len=len(input_text),
                        success=probs is not None,
                        is_trained=is_trained,
                    )
                
                if probs is not None:
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
                python -m mentalroberta.training.train --data data/german_large.json --epochs 30
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
