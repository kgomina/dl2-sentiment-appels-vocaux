---
title: Sentiment Appels Vocaux
emoji: 🎙️
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 4.26.0
app_file: app.py
pinned: false
license: mit
---

# 🎙️ Détection Automatique de Sentiment dans des Appels Vocaux

Pipeline complet **Audio → Transcription (ASR) → Analyse de Sentiment (NLP)**
combinant **Wav2Vec 2.0** et **BERT**, avec interface **Gradio**, interface **Streamlit** et **API REST (FastAPI)**.

> Projet réalisé dans le cadre du module _Deep Learning 2_ — Dakar Institute of Technology (2026).

**Dépôt GitHub** : https://github.com/kgomina/dl2-sentiment-appels-vocaux
**Démo en ligne (Streamlit Cloud)** : https://dl2-sentiment-appels-vocaux-4ajbjvgtrsxdgpbrgceuq8.streamlit.app/
**Déploiement Hugging Face Spaces** : code poussé, test en ligne en attente d'un plan payant (voir démo Streamlit ci-dessus comme alternative fonctionnelle).

---

## 1. Architecture
Audio (.wav/.mp3)
│
▼
Prétraitement (mono, 16kHz, normalisation amplitude)
│
▼
ASR — Wav2Vec 2.0 (jonatasgrosman/wav2vec2-large-xlsr-53-french)
│ (texte transcrit)
▼
Sentiment — BERT (nlptown/bert-base-multilingual-uncased-sentiment)
│
▼
Sortie : { transcription, sentiment (positif/négatif/neutre), confiance }

### Structure du dépôt
sentiment_call_pipeline/
├── app/
│ ├── audio_utils.py # Chargement + prétraitement audio
│ ├── asr.py # Modèle Wav2Vec2 (transcription)
│ ├── sentiment.py # Modèle BERT (sentiment)
│ ├── pipeline.py # Orchestration complète + gestion d'erreurs
│ ├── api.py # API FastAPI (endpoint POST /predict)
│ └── gradio_app.py # Interface Gradio
├── tests/
│ ├── test_audio_utils.py # Tests unitaires (prétraitement)
│ └── evaluate.py # Évaluation quantitative (WER, Accuracy, F1)
├── examples/ # 3 fichiers audio de démo (1 par classe)
├── requirements.txt
├── app.py # Point d'entrée Hugging Face Spaces (Gradio)
├── streamlit_app.py # Interface Streamlit (design entreprise, déployée)
├── Dockerfile
└── README.md

---

## 2. Choix des modèles et justification

| Tâche         | Modèle                                                                                                                        | Justification                                                                                                                                                                                                                        |
| ------------- | ----------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **ASR**       | [`jonatasgrosman/wav2vec2-large-xlsr-53-french`](https://huggingface.co/jonatasgrosman/wav2vec2-large-xlsr-53-french)         | Wav2Vec2 XLSR-53 fine-tuné spécifiquement sur du français (Common Voice), largement utilisé en production, bon compromis qualité/latence, aucun fine-tuning supplémentaire requis.                                                   |
| **Sentiment** | [`nlptown/bert-base-multilingual-uncased-sentiment`](https://huggingface.co/nlptown/bert-base-multilingual-uncased-sentiment) | BERT multilingue fine-tuné sur des avis clients (1 à 5 étoiles), inclut le français nativement, contexte proche des appels clients. Les 5 classes sont regroupées en 3 classes métier (1-2★ → négatif, 3★ → neutre, 4-5★ → positif). |

Alternative documentée dans le code (`app/sentiment.py`) : `cmarkea/distilcamembert-base-sentiment`
(CamemBERT, 100% francophone, plus léger) — activable en changeant `model_name`.

---

## 3. Installation

### Prérequis

- Python ≥ 3.9
- ffmpeg installé sur le système (pour la lecture des `.mp3`) :
```bash
  sudo apt-get install ffmpeg libsndfile1   # Linux
  brew install ffmpeg                        # macOS
```

### Étapes

```bash
git clone https://github.com/kgomina/dl2-sentiment-appels-vocaux.git
cd dl2-sentiment-appels-vocaux

python -m venv venv
source venv/bin/activate        # Windows : venv\Scripts\activate

pip install -r requirements.txt
```

Les modèles Hugging Face (Wav2Vec2 + BERT) sont téléchargés **automatiquement**
au premier appel (mis en cache localement dans `~/.cache/huggingface`).

---

## 4. Utilisation

### a) Interface Gradio

```bash
python -m app.gradio_app
```

→ ouvre une interface web locale (`http://127.0.0.1:7860`) permettant d'uploader
un audio et de voir : transcription intermédiaire, sentiment, score de confiance.

### b) API REST

```bash
uvicorn app.api:app --reload --host 0.0.0.0 --port 8000
```

Documentation interactive auto-générée : `http://localhost:8000/docs`

**Exemple d'appel curl :**

```bash
curl -X POST "http://localhost:8000/predict" \
     -F "file=@examples/positif_1.wav"
```

**Exemple de réponse JSON :**

```json
{
  "transcription": "je suis très satisfait du service merci beaucoup",
  "sentiment": "positif",
  "confidence": 0.87,
  "raw_star_rating": 5,
  "audio_duration_sec": 3.2,
  "processing_time_sec": 1.4
}
```

**Exemple d'appel Python :**

```python
import requests

with open("examples/positif_1.wav", "rb") as f:
    response = requests.post("http://localhost:8000/predict", files={"file": f})

print(response.json())
```

### c) Docker (bonus)

```bash
docker build -t sentiment-vocal .
docker run -p 8000:8000 sentiment-vocal
```

✅ Testé avec succès : build et exécution du conteneur validés, l'endpoint
`/predict` répond correctement une fois le conteneur lancé.

### d) Démo en ligne (Streamlit Cloud)

Une interface web complète (design "entreprise" : dashboard, historique de
session, métriques visuelles) est déployée publiquement et directement
testable, sans aucune installation :

🔗 **https://dl2-sentiment-appels-vocaux-4ajbjvgtrsxdgpbrgceuq8.streamlit.app/**

Pour la lancer en local :
```bash
streamlit run streamlit_app.py
```

Code source de cette interface : `streamlit_app.py` (à la racine du dépôt).

---

## 5. Tests et évaluation

### Tests unitaires (prétraitement audio)

```bash
pytest tests/test_audio_utils.py -v
```

### Évaluation quantitative (bonus — WER / Accuracy / F1)

Créer un fichier `tests/eval_dataset.json` annoté (voir format dans `tests/evaluate.py`), puis :

```bash
python tests/evaluate.py --data tests/eval_dataset.json
```

Calcule :

- **WER** (Word Error Rate) pour la qualité de transcription (via `jiwer`) ;
- **Accuracy** et **F1-macro** pour la classification de sentiment (via `scikit-learn`).

### Résultats obtenus (sur 3 échantillons de test, un par classe)

| Métrique             | Valeur |
| --------------------- | ------ |
| WER (ASR)             | 0.130  |
| Accuracy (sentiment) | 1.000  |
| F1-macro (sentiment)  | 1.000  |

[OK] examples/positif_1.wav -> sentiment=positif (attendu=positif)
[OK] examples/negatif_1.wav -> sentiment=negatif (attendu=negatif)
[OK] examples/neutre_1.wav -> sentiment=neutre (attendu=neutre)

Un WER de 0.13 signifie qu'environ 13% des mots transcrits diffèrent de la
référence (erreurs mineures de reconnaissance sur quelques mots), ce qui reste
un très bon résultat pour du Wav2Vec2 pré-entraîné sans fine-tuning
supplémentaire. La classification de sentiment est parfaite sur cet échantillon
réduit (3 fichiers) — un test à plus grande échelle serait nécessaire pour une
évaluation statistiquement robuste (voir section Limites).

---

## 6. Gestion des erreurs

Le pipeline gère explicitement :

- **Format non supporté** (autre que `.wav`/`.mp3`) → `AudioValidationError` → HTTP 400
- **Fichier vide** → `AudioValidationError` → HTTP 400
- **Audio silencieux** (RMS quasi nul) → `AudioValidationError` → HTTP 422
- **Durée > 5 minutes** → `AudioValidationError` → HTTP 400
- **Erreurs d'inférence** (ASR ou sentiment) → `PipelineError` → HTTP 422/500

---

## 7. Cas d'usage

- Analyse post-appel de centres de contact (satisfaction client) ;
- Priorisation des tickets/appels à fort mécontentement pour un rappel rapide ;
- Tableaux de bord agrégés de satisfaction client par période/agent/produit.

---

## 8. Limites connues

- **Qualité ASR dépendante du bruit de fond** : les appels téléphoniques réels
  (bande passante réduite, bruit ambiant) dégradent la transcription par rapport
  aux données d'entraînement (Common Voice, plus propres).
- **Sentiment basé uniquement sur le texte** : ignore la prosodie (ton de la voix),
  ce qui peut manquer du sarcasme ou une colère contenue.
- **Regroupement 5★→3 classes** : la frontière entre "neutre" (3★) et
  "légèrement négatif/positif" (2★/4★) reste parfois floue.
- **Limite de durée à 5 minutes** : les appels plus longs doivent être découpés
  en amont (non géré automatiquement dans cette version).
- **Pas de diarisation** : le pipeline ne distingue pas agent/client dans l'audio.

---

## 9. Auteur: Kéléfing GOMINA

Projet réalisé dans le cadre de l'examen _Deep Learning 2_ — DIT (Dakar Institute
of Technology), 2026.