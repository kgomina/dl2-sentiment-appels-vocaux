"""
generate_examples.py
---------------------
Génère 3 fichiers audio de démonstration (un par classe de sentiment) à partir
de phrases françaises types, via le TTS de gTTS (nécessite une connexion internet)
ou pyttsx3 (hors-ligne) selon ce qui est disponible.

Utilisation :
    python examples/generate_examples.py

NOTE : Pour la soutenance, il est recommandé d'utiliser de VRAIS enregistrements
vocaux (vous-même ou collègues) plutôt que du TTS, pour un test plus représentatif
du cas d'usage réel (appels clients).
"""

import os

PHRASES = {
    "positif_1.wav": "Je suis très satisfait du service, merci beaucoup pour votre aide.",
    "negatif_1.wav": "Je suis vraiment mécontent, ma commande est arrivée cassée et en retard.",
    "neutre_1.wav": "Je voudrais connaître le statut de ma commande numéro douze mille trois cent quarante-cinq.",
}

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))


def main():
    try:
        from gtts import gTTS
    except ImportError:
        raise SystemExit(
            "gTTS n'est pas installé. Lancez : pip install gTTS\n"
            "(nécessite une connexion internet pour synthétiser la voix)."
        )

    for filename, text in PHRASES.items():
        mp3_path = os.path.join(OUTPUT_DIR, filename.replace(".wav", ".mp3"))
        tts = gTTS(text=text, lang="fr")
        tts.save(mp3_path)
        print(f"Généré : {mp3_path}  -->  \"{text}\"")

    print("\nConversion en .wav recommandée avec ffmpeg :")
    print("  ffmpeg -i positif_1.mp3 positif_1.wav")


if __name__ == "__main__":
    main()
