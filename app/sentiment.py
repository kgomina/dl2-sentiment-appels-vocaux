"""
sentiment.py
------------
Analyse de sentiment à partir du texte transcrit, via un modèle de type BERT.

Modèle utilisé par défaut :
  nlptown/bert-base-multilingual-uncased-sentiment
  -> BERT multilingue fine-tuné pour la classification de sentiment sur une
     échelle de 1 à 5 étoiles (avis clients, multilingue dont le français).
     Choisi car : (1) disponible librement sur Hugging Face, (2) entraîné
     spécifiquement sur des données de type "avis client" proches du contexte
     "appels vocaux clients", (3) gère nativement le français sans fine-tuning
     supplémentaire.

  Alternative possible : cmarkea/distilcamembert-base-sentiment (CamemBERT,
  plus léger, entraîné spécifiquement en français) — à activer via le paramètre
  `model_name` si on veut privilégier un modèle 100% francophone.

Les 5 classes (1..5 étoiles) sont regroupées en 3 classes métier :
  1-2 étoiles -> "negatif"
  3 étoiles   -> "neutre"
  4-5 étoiles -> "positif"
"""

from functools import lru_cache
from dataclasses import dataclass
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

DEFAULT_SENTIMENT_MODEL = "nlptown/bert-base-multilingual-uncased-sentiment"

STAR_TO_LABEL = {
    1: "negatif",
    2: "negatif",
    3: "neutre",
    4: "positif",
    5: "positif",
}


@dataclass
class SentimentResult:
    label: str          # "positif" | "negatif" | "neutre"
    confidence: float    # probabilité associée à la classe prédite (0-1)
    raw_star_rating: int  # note brute 1-5 renvoyée par le modèle (traçabilité)


class SentimentModel:
    def __init__(self, model_name: str = DEFAULT_SENTIMENT_MODEL, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def predict(self, text: str) -> SentimentResult:
        if not text or not text.strip():
            # Transcription vide -> impossible d'analyser un sentiment fiable
            return SentimentResult(label="neutre", confidence=0.0, raw_star_rating=3)

        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=512, padding=True
        ).to(self.device)

        logits = self.model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)[0]
        star_idx = int(torch.argmax(probs).item())  # index 0..4
        star_rating = star_idx + 1                   # note 1..5
        confidence = float(probs[star_idx].item())

        label = STAR_TO_LABEL[star_rating]
        return SentimentResult(label=label, confidence=confidence, raw_star_rating=star_rating)


@lru_cache(maxsize=1)
def get_sentiment_model(model_name: str = DEFAULT_SENTIMENT_MODEL) -> SentimentModel:
    """Charge (et met en cache) une instance unique du modèle de sentiment."""
    return SentimentModel(model_name=model_name)
