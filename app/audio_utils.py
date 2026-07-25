"""
audio_utils.py
--------------
Chargement, validation et prétraitement des fichiers audio.

Étapes appliquées :
1. Chargement (.wav / .mp3) via librosa (gère le décodage mp3 par audioread/soundfile).
2. Conversion en mono (moyenne des canaux si stéréo).
3. Rééchantillonnage à 16 kHz (obligatoire pour Wav2Vec2).
4. Normalisation d'amplitude (peak normalization dans [-1, 1]).
5. Vérifications : fichier vide, silence total, durée > 5 minutes.
"""

from dataclasses import dataclass
import numpy as np
import librosa

TARGET_SAMPLE_RATE = 16_000
MAX_DURATION_SECONDS = 5 * 60  # 5 minutes
SILENCE_RMS_THRESHOLD = 1e-4    # en dessous -> considéré comme silence total
SUPPORTED_EXTENSIONS = {".wav", ".mp3"}


class AudioValidationError(Exception):
    """Erreur levée quand un fichier audio est invalide (format, vide, silencieux, trop long)."""
    pass


@dataclass
class ProcessedAudio:
    waveform: np.ndarray   # signal mono, float32, normalisé
    sample_rate: int       # toujours TARGET_SAMPLE_RATE après traitement
    duration_sec: float


def _check_extension(filename: str) -> None:
    ext = "." + filename.lower().rsplit(".", 1)[-1] if "." in filename else ""
    if ext not in SUPPORTED_EXTENSIONS:
        raise AudioValidationError(
            f"Format non supporté : '{ext}'. Formats acceptés : {SUPPORTED_EXTENSIONS}"
        )


def load_and_preprocess(filepath: str, filename: str | None = None) -> ProcessedAudio:
    """
    Charge un fichier audio depuis le disque et applique le prétraitement complet.

    Args:
        filepath: chemin vers le fichier audio sur disque.
        filename: nom original du fichier (pour vérifier l'extension), sinon déduit de filepath.

    Returns:
        ProcessedAudio: signal mono 16kHz normalisé.

    Raises:
        AudioValidationError: si le fichier est invalide (format, vide, silencieux, trop long).
    """
    name_to_check = filename or filepath
    _check_extension(name_to_check)

    try:
        # sr=None -> on garde le sample rate d'origine pour le resampling explicite ensuite
        waveform, original_sr = librosa.load(filepath, sr=None, mono=False)
    except Exception as exc:
        raise AudioValidationError(f"Impossible de lire le fichier audio : {exc}") from exc

    if waveform is None or waveform.size == 0:
        raise AudioValidationError("Le fichier audio est vide.")

    # Conversion mono : si stéréo (shape = (channels, samples)), on moyenne les canaux
    if waveform.ndim > 1:
        waveform = np.mean(waveform, axis=0)

    # Rééchantillonnage à 16 kHz si nécessaire
    if original_sr != TARGET_SAMPLE_RATE:
        waveform = librosa.resample(
            waveform.astype(np.float32), orig_sr=original_sr, target_sr=TARGET_SAMPLE_RATE
        )

    duration_sec = len(waveform) / TARGET_SAMPLE_RATE

    if duration_sec > MAX_DURATION_SECONDS:
        raise AudioValidationError(
            f"Durée maximale dépassée : {duration_sec:.1f}s > {MAX_DURATION_SECONDS}s (5 min)."
        )

    # Détection de silence total via RMS
    rms = float(np.sqrt(np.mean(np.square(waveform)))) if len(waveform) > 0 else 0.0
    if rms < SILENCE_RMS_THRESHOLD:
        raise AudioValidationError("Le fichier audio semble silencieux (aucun signal détecté).")

    # Normalisation d'amplitude (peak normalization)
    peak = np.max(np.abs(waveform))
    if peak > 0:
        waveform = waveform / peak

    return ProcessedAudio(
        waveform=waveform.astype(np.float32),
        sample_rate=TARGET_SAMPLE_RATE,
        duration_sec=duration_sec,
    )
