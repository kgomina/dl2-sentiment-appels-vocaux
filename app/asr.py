"""
asr.py
------
Transcription vocale (Speech-to-Text) via Wav2Vec 2.0.

Modèle utilisé par défaut :
  jonatasgrosman/wav2vec2-large-xlsr-53-french
  -> Wav2Vec2 XLSR-53 fine-tuné sur du français (Common Voice), très utilisé
     en production pour l'ASR français, bon compromis performance/latence,
     disponible librement sur Hugging Face Hub.

Le modèle est chargé une seule fois (singleton) pour éviter de recharger
les poids à chaque requête (coûteux en temps et mémoire).
"""

from functools import lru_cache
import numpy as np
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

DEFAULT_ASR_MODEL = "jonatasgrosman/wav2vec2-large-xlsr-53-french"


class ASRModel:
    def __init__(self, model_name: str = DEFAULT_ASR_MODEL, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = Wav2Vec2ForCTC.from_pretrained(model_name).to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def transcribe(self, waveform: np.ndarray, sample_rate: int = 16_000) -> str:
        """
        Transcrit un signal audio (mono, 16kHz, float32 normalisé) en texte.
        """
        inputs = self.processor(
            waveform, sampling_rate=sample_rate, return_tensors="pt", padding=True
        )
        input_values = inputs.input_values.to(self.device)
        attention_mask = getattr(inputs, "attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        logits = self.model(input_values, attention_mask=attention_mask).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = self.processor.batch_decode(predicted_ids)[0]

        return transcription.strip().lower()


@lru_cache(maxsize=1)
def get_asr_model(model_name: str = DEFAULT_ASR_MODEL) -> ASRModel:
    """Charge (et met en cache) une instance unique du modèle ASR."""
    return ASRModel(model_name=model_name)
