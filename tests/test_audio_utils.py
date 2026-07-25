"""
test_audio_utils.py
--------------------
Tests unitaires pour la validation et le prétraitement audio.
Ces tests ne nécessitent PAS les modèles ASR/BERT (rapides, sans téléchargement).
"""

import numpy as np
import soundfile as sf
import pytest

from app.audio_utils import load_and_preprocess, AudioValidationError, TARGET_SAMPLE_RATE


def _write_wav(tmp_path, name, signal, sr):
    path = tmp_path / name
    sf.write(str(path), signal, sr)
    return str(path)


def test_valid_audio_is_resampled_and_normalized(tmp_path):
    sr = 44_100
    t = np.linspace(0, 1, sr, endpoint=False)
    signal = 0.3 * np.sin(2 * np.pi * 440 * t).astype(np.float32)  # note La 440Hz, 1s
    path = _write_wav(tmp_path, "valid.wav", signal, sr)

    result = load_and_preprocess(path, filename="valid.wav")

    assert result.sample_rate == TARGET_SAMPLE_RATE
    assert np.max(np.abs(result.waveform)) <= 1.0 + 1e-6
    assert 0.9 < result.duration_sec < 1.1


def test_unsupported_extension_raises(tmp_path):
    sr = 16_000
    signal = np.zeros(sr, dtype=np.float32)
    path = _write_wav(tmp_path, "audio.flac", signal, sr)

    with pytest.raises(AudioValidationError):
        load_and_preprocess(path, filename="audio.flac")


def test_silent_audio_raises(tmp_path):
    sr = 16_000
    signal = np.zeros(sr, dtype=np.float32)
    path = _write_wav(tmp_path, "silence.wav", signal, sr)

    with pytest.raises(AudioValidationError):
        load_and_preprocess(path, filename="silence.wav")


def test_too_long_audio_raises(tmp_path):
    sr = 16_000
    duration_sec = 6 * 60  # 6 minutes > 5 min max
    t = np.linspace(0, duration_sec, sr * duration_sec, endpoint=False)
    signal = (0.2 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    path = _write_wav(tmp_path, "long.wav", signal, sr)

    with pytest.raises(AudioValidationError):
        load_and_preprocess(path, filename="long.wav")
