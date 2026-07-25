# Exemples audio de démonstration

⚠️ **Important** : les fichiers `positif_1.wav`, `negatif_1.wav`, `neutre_1.wav`
fournis ici sont des **tonalités synthétiques placeholder** (générées par script,
sans parole), utilisées uniquement pour valider que le pipeline technique
tourne de bout en bout (formats, tailles, gestion d'erreurs).

**Avant la soutenance, vous devez les remplacer par de VRAIS enregistrements
vocaux en français** (vous-même, un(e) collègue, ou via `generate_examples.py`
avec gTTS si vous avez internet), correspondant à :

| Fichier | Contenu attendu |
|---|---|
| `positif_1.wav` | Client satisfait (ex : "Je suis très content du service, merci !") |
| `negatif_1.wav` | Client mécontent (ex : "C'est inadmissible, ma commande est en retard") |
| `neutre_1.wav` | Question factuelle sans charge émotionnelle (ex : "Quel est le statut de ma commande ?") |

Le script `generate_examples.py` peut générer ces fichiers automatiquement via
gTTS (nécessite internet), à convertir ensuite en `.wav` avec ffmpeg.
