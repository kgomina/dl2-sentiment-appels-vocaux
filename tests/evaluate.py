"""
evaluate.py
-----------
Évaluation quantitative du pipeline (bonus +1 pt) :
  - WER (Word Error Rate) pour l'ASR, via `jiwer`.
  - Accuracy / F1-score pour le sentiment, via `scikit-learn`.

Utilisation :
    python tests/evaluate.py --data tests/eval_dataset.json

Format attendu du fichier JSON d'évaluation (à créer par l'étudiant avec
ses propres fichiers audio annotés) :
[
  {
    "audio_path": "examples/positif_1.wav",
    "reference_transcription": "je suis très satisfait du service merci",
    "reference_sentiment": "positif"
  },
  ...
]
"""

import argparse
import json
import os
import sys

# Ajoute la racine du projet au path pour pouvoir importer le package "app"
# même quand ce script est lancé directement (python tests/evaluate.py)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jiwer import wer
from sklearn.metrics import accuracy_score, f1_score, classification_report

from app.pipeline import run_pipeline, PipelineError


def evaluate(dataset_path: str):
    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    references_text, hypotheses_text = [], []
    references_sentiment, predictions_sentiment = [], []

    for item in dataset:
        try:
            result = run_pipeline(item["audio_path"], filename=item["audio_path"])
        except PipelineError as exc:
            print(f"[SKIP] {item['audio_path']} -> erreur pipeline : {exc}")
            continue

        references_text.append(item["reference_transcription"].lower())
        hypotheses_text.append(result.transcription.lower())

        references_sentiment.append(item["reference_sentiment"])
        predictions_sentiment.append(result.sentiment)

        print(f"[OK] {item['audio_path']} -> sentiment={result.sentiment} "
              f"(attendu={item['reference_sentiment']})")

    if references_text:
        global_wer = wer(references_text, hypotheses_text)
        print(f"\n=== ASR ===\nWER global : {global_wer:.3f}")

    if references_sentiment:
        acc = accuracy_score(references_sentiment, predictions_sentiment)
        f1 = f1_score(references_sentiment, predictions_sentiment, average="macro")
        print(f"\n=== Sentiment ===\nAccuracy : {acc:.3f}\nF1-macro : {f1:.3f}\n")
        print(classification_report(references_sentiment, predictions_sentiment))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Chemin vers le fichier JSON d'évaluation.")
    args = parser.parse_args()
    evaluate(args.data)
