"""
api.py
------
API REST (FastAPI) exposant le pipeline audio -> sentiment.

Lancer localement :
    uvicorn app.api:app --reload --host 0.0.0.0 --port 8000

Exemple d'appel :
    curl -X POST "http://localhost:8000/predict" \
         -F "file=@examples/positif_1.wav"
"""

import os
import shutil
import tempfile

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse

from app.pipeline import run_pipeline, PipelineError

app = FastAPI(
    title="API de Détection de Sentiment dans des Appels Vocaux",
    description="Pipeline Wav2Vec2 (ASR) + BERT (Sentiment) pour appels clients.",
    version="1.0.0",
)

ALLOWED_EXTENSIONS = {".wav", ".mp3"}
MAX_FILE_SIZE_MB = 25  # garde-fou raisonnable pour ~5 min audio


@app.get("/")
def root():
    return {"status": "ok", "message": "API de sentiment vocal opérationnelle."}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Reçoit un fichier audio (.wav ou .mp3) et retourne :
    - transcription : texte transcrit par le modèle ASR
    - sentiment : "positif" | "negatif" | "neutre"
    - confidence : score de confiance (0-1)
    """
    filename = file.filename or "audio"
    ext = os.path.splitext(filename)[1].lower()

    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Format non supporté '{ext}'. Formats acceptés : {ALLOWED_EXTENSIONS}",
        )

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name

        size_mb = os.path.getsize(tmp_path) / (1024 * 1024)
        if size_mb > MAX_FILE_SIZE_MB:
            raise HTTPException(
                status_code=400,
                detail=f"Fichier trop volumineux ({size_mb:.1f} Mo > {MAX_FILE_SIZE_MB} Mo).",
            )
        if size_mb == 0:
            raise HTTPException(status_code=400, detail="Le fichier envoyé est vide.")

        result = run_pipeline(tmp_path, filename=filename)
        return JSONResponse(content=result.to_dict())

    except PipelineError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except HTTPException:
        raise
    except Exception as exc:  # filet de sécurité générique
        raise HTTPException(status_code=500, detail=f"Erreur interne : {exc}")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)
