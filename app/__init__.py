"""
Pipeline de détection automatique de sentiment dans des appels vocaux.

Modules :
- audio_utils : chargement / prétraitement audio
- asr         : transcription (Wav2Vec 2.0)
- sentiment   : classification de sentiment (CamemBERT)
- pipeline    : orchestration complète + gestion d'erreurs
"""

__version__ = "1.0.0"
