"""
pipeline.py
-----------
Orchestration complète du pipeline : Audio -> Prétraitement -> ASR -> Sentiment.

Point d'entrée unique utilisé à la fois par l'API FastAPI et par l'interface Gradio.
"""

from dataclasses import dataclass, asdict
import time

from app.audio_utils import load_and_preprocess, AudioValidationError
from app.asr import get_asr_model
from app.sentiment import get_sentiment_model


class PipelineError(Exception):
    """Erreur générique du pipeline (englobe les erreurs de validation / inférence)."""
    pass


@dataclass
class PipelineResult:
    transcription: str
    sentiment: str
    confidence: float
    raw_star_rating: int
    audio_duration_sec: float
    processing_time_sec: float

    def to_dict(self) -> dict:
        return asdict(self)


def run_pipeline(filepath: str, filename: str | None = None) -> PipelineResult:
    """
    Exécute le pipeline complet sur un fichier audio.

    Args:
        filepath: chemin du fichier audio sur disque.
        filename: nom original (pour vérifier l'extension .wav/.mp3).

    Returns:
        PipelineResult

    Raises:
        PipelineError: en cas de fichier invalide ou d'erreur d'inférence.
    """
    start = time.time()

    try:
        processed_audio = load_and_preprocess(filepath, filename=filename)
    except AudioValidationError as exc:
        raise PipelineError(f"Erreur de validation audio : {exc}") from exc

    try:
        asr_model = get_asr_model()
        transcription = asr_model.transcribe(
            processed_audio.waveform, sample_rate=processed_audio.sample_rate
        )
    except Exception as exc:
        raise PipelineError(f"Erreur lors de la transcription (ASR) : {exc}") from exc

    try:
        sentiment_model = get_sentiment_model()
        sentiment_result = sentiment_model.predict(transcription)
    except Exception as exc:
        raise PipelineError(f"Erreur lors de l'analyse de sentiment : {exc}") from exc

    processing_time = time.time() - start

    return PipelineResult(
        transcription=transcription,
        sentiment=sentiment_result.label,
        confidence=round(sentiment_result.confidence, 4),
        raw_star_rating=sentiment_result.raw_star_rating,
        audio_duration_sec=round(processed_audio.duration_sec, 2),
        processing_time_sec=round(processing_time, 2),
    )
