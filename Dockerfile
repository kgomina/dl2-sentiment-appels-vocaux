FROM python:3.11-slim

# Dépendances système nécessaires pour librosa / soundfile / pydub (ffmpeg)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/
COPY examples/ ./examples/

# Pré-télécharge les modèles à la construction de l'image (optionnel, accélère le 1er appel)
# RUN python -c "from app.asr import get_asr_model; from app.sentiment import get_sentiment_model; get_asr_model(); get_sentiment_model()"

EXPOSE 8000 7860

# Par défaut : lance l'API. Pour Gradio -> CMD ["python", "-m", "app.gradio_app"]
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
