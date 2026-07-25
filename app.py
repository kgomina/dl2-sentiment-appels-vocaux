"""
app.py
------
Point d'entrée pour le déploiement sur Hugging Face Spaces (SDK: Gradio).
"""

from app.gradio_app import demo

if __name__ == "__main__":
    demo.launch(share=True)