"""
gradio_app.py
-------------
Interface Gradio pour tester le pipeline complet : audio -> transcription -> sentiment.

Lancer localement :
    python -m app.gradio_app
"""

import gradio as gr

from app.pipeline import run_pipeline, PipelineError

SENTIMENT_EMOJI = {"positif": "😀 Positif", "negatif": "😠 Négatif", "neutre": "😐 Neutre"}


def analyze(audio_filepath: str):
    if audio_filepath is None:
        return "⚠️ Aucun fichier fourni.", "-", "-"

    try:
        result = run_pipeline(audio_filepath, filename=audio_filepath)
    except PipelineError as exc:
        return f"❌ {exc}", "-", "-"
    except Exception as exc:
        return f"❌ Erreur inattendue : {exc}", "-", "-"

    sentiment_display = SENTIMENT_EMOJI.get(result.sentiment, result.sentiment)
    confidence_display = f"{result.confidence * 100:.1f}%"
    return result.transcription, sentiment_display, confidence_display


with gr.Blocks(title="Détection de Sentiment - Appels Vocaux") as demo:
    gr.Markdown(
        "# 🎙️ Détection Automatique de Sentiment dans des Appels Vocaux\n"
        "Pipeline : **Wav2Vec 2.0** (transcription) → **BERT** (sentiment)\n\n"
        "Formats acceptés : `.wav`, `.mp3` — durée max : 5 minutes."
    )

    with gr.Row():
        audio_input = gr.Audio(sources=["upload", "microphone"], type="filepath", label="Fichier audio")

    analyze_btn = gr.Button("Analyser", variant="primary")

    with gr.Row():
        transcription_output = gr.Textbox(label="Transcription (ASR)", lines=4)
    with gr.Row():
        sentiment_output = gr.Textbox(label="Sentiment prédit")
        confidence_output = gr.Textbox(label="Score de confiance")

    analyze_btn.click(
        fn=analyze,
        inputs=[audio_input],
        outputs=[transcription_output, sentiment_output, confidence_output],
    )

    gr.Examples(
        examples=[
            ["examples/positif_1.wav"],
            ["examples/negatif_1.wav"],
            ["examples/neutre_1.wav"],
        ],
        inputs=[audio_input],
        label="Exemples de test (un par classe)",
    )

if __name__ == "__main__":
    demo.launch(share=True)
