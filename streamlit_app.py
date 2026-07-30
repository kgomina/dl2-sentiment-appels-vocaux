"""
streamlit_app.py
-----------------
Interface Streamlit "entreprise" pour le pipeline audio -> transcription -> sentiment.

Déploiement : Streamlit Community Cloud (gratuit), connecté au dépôt GitHub.

Lancer localement :
    streamlit run streamlit_app.py
"""

import os
import tempfile
from datetime import datetime

import streamlit as st

from app.pipeline import run_pipeline, PipelineError

# ----------------------------------------------------------------------------
# Configuration générale de la page
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Sentiment Vocal — Analyse d'Appels Clients",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

SENTIMENT_STYLE = {
    "positif": {"emoji": "", "label": "POSITIF", "color": "#16a34a", "bg": "#dcfce7"},
    "negatif": {"emoji": "", "label": "NÉGATIF", "color": "#dc2626", "bg": "#fee2e2"},
    "neutre":  {"emoji": "", "label": "NEUTRE",  "color": "#64748b", "bg": "#f1f5f9"},
}

EXAMPLES = {
    "-- Aucun exemple --": None,
    " Exemple Positif": "examples/positif_1.wav",
    " Exemple Négatif": "examples/negatif_1.wav",
    " Exemple Neutre": "examples/neutre_1.wav",
}

# ----------------------------------------------------------------------------
# CSS personnalisé — identité visuelle "entreprise"
# ----------------------------------------------------------------------------
st.markdown(
    """
    <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}

        html, body, [class*="css"] {
            font-family: 'Segoe UI', 'Inter', sans-serif;
        }

        .hero-banner {
            background: linear-gradient(120deg, #0f172a 0%, #1e3a8a 55%, #1d4ed8 100%);
            padding: 2.2rem 2.5rem;
            border-radius: 16px;
            color: white;
            margin-bottom: 1.8rem;
            box-shadow: 0 10px 30px rgba(15, 23, 42, 0.25);
        }
        .hero-banner h1 {
            font-size: 2.1rem;
            font-weight: 700;
            margin-bottom: 0.3rem;
        }
        .hero-banner p {
            font-size: 1.02rem;
            opacity: 0.9;
            margin: 0;
        }
        .badge-row {
            margin-top: 1rem;
        }
        .pill {
            display: inline-block;
            background: rgba(255,255,255,0.15);
            padding: 0.25rem 0.75rem;
            border-radius: 999px;
            font-size: 0.78rem;
            margin-right: 0.5rem;
            border: 1px solid rgba(255,255,255,0.3);
        }

        .upload-card {
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 14px;
            padding: 1.5rem;
            box-shadow: 0 2px 10px rgba(0,0,0,0.04);
            margin-bottom: 1.2rem;
        }

        .result-card {
            border-radius: 16px;
            padding: 1.8rem;
            margin-top: 1rem;
            border: 1px solid #e2e8f0;
            box-shadow: 0 4px 16px rgba(0,0,0,0.06);
        }

        .sentiment-badge {
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            font-size: 1.4rem;
            font-weight: 700;
            padding: 0.5rem 1.2rem;
            border-radius: 999px;
        }

        .confidence-track {
            width: 100%;
            background: #e2e8f0;
            border-radius: 999px;
            height: 10px;
            margin-top: 0.4rem;
            overflow: hidden;
        }
        .confidence-fill {
            height: 100%;
            border-radius: 999px;
        }

        .metric-box {
            background: #f8fafc;
            border-radius: 12px;
            padding: 0.9rem 1rem;
            text-align: center;
            border: 1px solid #eef2f7;
        }
        .metric-box .value {
            font-size: 1.3rem;
            font-weight: 700;
            color: #1e293b;
        }
        .metric-box .label {
            font-size: 0.78rem;
            color: #64748b;
            text-transform: uppercase;
            letter-spacing: 0.04em;
        }

        .transcript-box {
            background: #f8fafc;
            border-left: 4px solid #1d4ed8;
            padding: 1rem 1.2rem;
            border-radius: 8px;
            font-style: italic;
            color: #1e293b;
        }

        .history-item {
            padding: 0.6rem 0.8rem;
            border-radius: 10px;
            background: #f8fafc;
            margin-bottom: 0.5rem;
            font-size: 0.85rem;
            border: 1px solid #eef2f7;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# Bannière principale
# ----------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero-banner">
        <h1>🎙️ Plateforme d'Analyse de Sentiment — Appels Clients</h1>
        <p>Pipeline IA de bout en bout : reconnaissance vocale (Wav2Vec 2.0) et
        classification de sentiment (BERT), pour l'analyse automatisée des
        interactions clients.</p>
        <div class="badge-row">
            <span class="pill">🧠 Wav2Vec 2.0 — ASR</span>
            <span class="pill">🧠 BERT — Sentiment</span>
            <span class="pill">⚡ Traitement en temps réel</span>
            <span class="pill">🇫🇷 Optimisé pour le français</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# Historique de session
# ----------------------------------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history = []

# ----------------------------------------------------------------------------
# Sidebar — informations techniques
# ----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 📊 À propos du pipeline")
    st.markdown(
        """
        **Architecture :**
        1. Prétraitement audio (mono, 16 kHz)
        2. Transcription — `wav2vec2-large-xlsr-53-french`
        3. Analyse de sentiment — `bert-base-multilingual-uncased-sentiment`

        **Formats acceptés :** `.wav`, `.mp3`
        **Durée max :** 5 minutes
        """
    )
    st.markdown("---")
    st.markdown("### 📈 Statistiques de session")
    total = len(st.session_state.history)
    st.metric("Analyses effectuées", total)
    if total > 0:
        pos = sum(1 for h in st.session_state.history if h["sentiment"] == "positif")
        neg = sum(1 for h in st.session_state.history if h["sentiment"] == "negatif")
        neu = sum(1 for h in st.session_state.history if h["sentiment"] == "neutre")
        c1, c2, c3 = st.columns(3)
        c1.metric("😀", pos)
        c2.metric("😠", neg)
        c3.metric("😐", neu)

    st.markdown("---")
    st.markdown(
        "🔗 [Code source sur GitHub]"
        "(https://github.com/kgomina/dl2-sentiment-appels-vocaux)"
    )
    st.caption("Projet Deep Learning 2 — DIT 2026")

# ----------------------------------------------------------------------------
# Zone d'analyse
# ----------------------------------------------------------------------------
col_left, col_right = st.columns([1.3, 1])

with col_left:
    st.markdown('<div class="upload-card">', unsafe_allow_html=True)
    st.subheader("📤 Charger un appel à analyser")

    uploaded_file = st.file_uploader(
        "Glissez-déposez un fichier audio, ou cliquez pour parcourir",
        type=["wav", "mp3"],
        label_visibility="collapsed",
    )

    example_choice = st.selectbox("Ou choisissez un exemple de démonstration :", list(EXAMPLES.keys()))
    st.markdown("</div>", unsafe_allow_html=True)

    filepath_to_analyze, filename_to_analyze, temp_path = None, None, None

    if uploaded_file is not None:
        ext = os.path.splitext(uploaded_file.name)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            tmp.write(uploaded_file.getbuffer())
            temp_path = tmp.name
        filepath_to_analyze = temp_path
        filename_to_analyze = uploaded_file.name
        st.audio(uploaded_file)
    elif EXAMPLES[example_choice] is not None:
        filepath_to_analyze = EXAMPLES[example_choice]
        filename_to_analyze = EXAMPLES[example_choice]
        if os.path.exists(filepath_to_analyze):
            st.audio(filepath_to_analyze)

    analyze_clicked = st.button(
        "🔍 Analyser l'appel", type="primary", use_container_width=True,
        disabled=(filepath_to_analyze is None),
    )

with col_right:
    result_placeholder = st.empty()

    if analyze_clicked:
        with st.spinner("⏳ Transcription et analyse du sentiment en cours..."):
            try:
                result = run_pipeline(filepath_to_analyze, filename=filename_to_analyze)
                style = SENTIMENT_STYLE.get(
                    result.sentiment, {"emoji": "❓", "label": result.sentiment, "color": "#64748b", "bg": "#f1f5f9"}
                )
                confidence_pct = round(result.confidence * 100, 1)

                with result_placeholder.container():
                    st.markdown(
                        f"""
                        <div class="result-card" style="background:{style['bg']}22;">
                            <div class="sentiment-badge" style="background:{style['bg']}; color:{style['color']};">
                                {style['emoji']} {style['label']}
                            </div>
                            <p style="margin-top:0.8rem; margin-bottom:0.2rem; font-size:0.85rem; color:#475569;">
                                Score de confiance
                            </p>
                            <div class="confidence-track">
                                <div class="confidence-fill" style="width:{confidence_pct}%; background:{style['color']};"></div>
                            </div>
                            <p style="text-align:right; font-size:0.85rem; color:#475569; margin-top:0.2rem;">
                                {confidence_pct}%
                            </p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    m1, m2, m3 = st.columns(3)
                    m1.markdown(
                        f'<div class="metric-box"><div class="value">{result.audio_duration_sec}s</div>'
                        f'<div class="label">Durée audio</div></div>',
                        unsafe_allow_html=True,
                    )
                    m2.markdown(
                        f'<div class="metric-box"><div class="value">{result.processing_time_sec}s</div>'
                        f'<div class="label">Temps traitement</div></div>',
                        unsafe_allow_html=True,
                    )
                    m3.markdown(
                        f'<div class="metric-box"><div class="value">{result.raw_star_rating}★</div>'
                        f'<div class="label">Note brute</div></div>',
                        unsafe_allow_html=True,
                    )

                    st.markdown("<br>", unsafe_allow_html=True)
                    st.markdown("**📝 Transcription automatique**")
                    st.markdown(
                        f'<div class="transcript-box">"{result.transcription or "(vide)"}"</div>',
                        unsafe_allow_html=True,
                    )

                st.session_state.history.insert(
                    0,
                    {
                        "time": datetime.now().strftime("%H:%M:%S"),
                        "file": filename_to_analyze,
                        "sentiment": result.sentiment,
                        "confidence": confidence_pct,
                    },
                )

            except PipelineError as exc:
                result_placeholder.error(f"❌ {exc}")
            except Exception as exc:
                result_placeholder.error(f"❌ Erreur inattendue : {exc}")
            finally:
                if temp_path and os.path.exists(temp_path):
                    os.remove(temp_path)
    else:
        result_placeholder.info("👈 Chargez un fichier audio et cliquez sur **Analyser l'appel** pour voir les résultats ici.")

# ----------------------------------------------------------------------------
# Historique des analyses de la session
# ----------------------------------------------------------------------------
if st.session_state.history:
    st.markdown("---")
    st.subheader("🕒 Historique de la session")
    for item in st.session_state.history[:8]:
        style = SENTIMENT_STYLE.get(item["sentiment"], {"emoji": "❓", "color": "#64748b"})
        st.markdown(
            f'<div class="history-item">'
            f'<b>{item["time"]}</b> — {os.path.basename(str(item["file"]))} → '
            f'<span style="color:{style["color"]}; font-weight:600;">{style["emoji"]} {item["sentiment"]}</span> '
            f'({item["confidence"]}%)</div>',
            unsafe_allow_html=True,
        )

st.markdown("---")
st.caption(
    "Projet d'examen *Deep Learning 2* — Dakar Institute of Technology (2026) · "
    "[Code source](https://github.com/kgomina/dl2-sentiment-appels-vocaux)"
)