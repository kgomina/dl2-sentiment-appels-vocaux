"""
streamlit_app.py
-----------------
Interface Streamlit pour tester le pipeline complet : audio -> transcription -> sentiment.

Déploiement : Streamlit Community Cloud (https://share.streamlit.io), gratuit,
connecté directement au dépôt GitHub.

Lancer localement :
    streamlit run streamlit_app.py
"""

import tempfile
import os

import streamlit as st

from app.pipeline import run_pipeline, PipelineError

st.set_page_config(page_title="Sentiment Appels Vocaux", page_icon="🎙️", layout="centered")

SENTIMENT_DISPLAY = {
    "positif": ("😀 Positif", "green"),
    "negatif": ("😠 Négatif", "red"),
    "neutre": ("😐 Neutre", "gray"),
}

st.title("🎙️ Détection Automatique de Sentiment dans des Appels Vocaux")
st.markdown(
    "Pipeline : **Wav2Vec 2.0** (transcription) → **BERT** (sentiment)\n\n"
    "Formats acceptés : `.wav`, `.mp3` — durée max : 5 minutes."
)

uploaded_file = st.file_uploader("Choisissez un fichier audio", type=["wav", "mp3"])

st.markdown("**Ou testez avec un exemple :**")
example_choice = st.selectbox(
    "Exemples fournis",
    ["-- Aucun --", "examples/positif_1.wav", "examples/negatif_1.wav", "examples/neutre_1.wav"],
)

filepath_to_analyze = None
filename_to_analyze = None
temp_path = None

if uploaded_file is not None:
    ext = os.path.splitext(uploaded_file.name)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        tmp.write(uploaded_file.getbuffer())
        temp_path = tmp.name
    filepath_to_analyze = temp_path
    filename_to_analyze = uploaded_file.name
    st.audio(uploaded_file)
elif example_choice != "-- Aucun --":
    filepath_to_analyze = example_choice
    filename_to_analyze = example_choice
    if os.path.exists(example_choice):
        st.audio(example_choice)

if st.button("Analyser", type="primary", disabled=(filepath_to_analyze is None)):
    with st.spinner("Analyse en cours (transcription + sentiment)..."):
        try:
            result = run_pipeline(filepath_to_analyze, filename=filename_to_analyze)

            st.subheader("📝 Transcription")
            st.write(result.transcription if result.transcription else "_(vide)_")

            label, color = SENTIMENT_DISPLAY.get(result.sentiment, (result.sentiment, "gray"))
            st.subheader("💬 Sentiment détecté")
            st.markdown(f":{color}[**{label}**]")

            col1, col2, col3 = st.columns(3)
            col1.metric("Confiance", f"{result.confidence * 100:.1f}%")
            col2.metric("Durée audio", f"{result.audio_duration_sec}s")
            col3.metric("Temps de traitement", f"{result.processing_time_sec}s")

        except PipelineError as exc:
            st.error(f"❌ {exc}")
        except Exception as exc:
            st.error(f"❌ Erreur inattendue : {exc}")
        finally:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)

st.markdown("---")
st.caption(
    "Projet d'examen Deep Learning 2 — DIT 2026. "
    "Code source : [GitHub](https://github.com/kgomina/dl2-sentiment-appels-vocaux)"
)