#!/bin/bash

# Script d'installation pour le système RAG multimodal
# ------------------------------------------------------

echo "Installation des dépendances du système RAG multimodal..."

# Vérifier Python 3.9+
python_version=$(python3 --version 2>&1 | awk '{print $2}')
python_major=$(echo $python_version | cut -d. -f1)
python_minor=$(echo $python_version | cut -d. -f2)

if [ "$python_major" -lt 3 ] || ([ "$python_major" -eq 3 ] && [ "$python_minor" -lt 9 ]); then
    echo "Python 3.9 ou supérieur est requis. Version détectée: $python_version"
    echo "Veuillez installer une version compatible de Python."
    exit 1
fi

echo "Version Python compatible détectée: $python_version"

# Créer un environnement virtuel
if [ ! -d ".venv" ]; then
    echo "Création de l'environnement virtuel..."
    python3 -m venv .venv
fi

# Activer l'environnement virtuel
echo "Activation de l'environnement virtuel..."
source .venv/bin/activate

# Mettre à jour pip
echo "Mise à jour de pip..."
pip install --upgrade pip

# Installer les dépendances
echo "Installation des dépendances..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install transformers sentence-transformers pillow faiss-cpu pymupdf
pip install fastapi uvicorn python-multipart pydantic

# Vérifier Ollama
echo "Vérification de l'installation Ollama..."
if ! command -v ollama &> /dev/null; then
    echo "Ollama n'est pas installé. Pour l'installer, consultez: https://ollama.ai/download"
    echo "Vous aurez besoin d'Ollama pour exécuter LLaVA localement."
else
    echo "Ollama est déjà installé."
    
    # Vérifier si le modèle LLaVA est disponible
    if ! ollama list | grep -q "llava"; then
        echo "Installation du modèle LLaVA via Ollama..."
        echo "Cela peut prendre plusieurs minutes selon votre connexion Internet."
        ollama pull llava:7b-v1.6-vicuna-q8_0
    else
        echo "Modèle LLaVA déjà installé."
    fi
fi

# Créer les répertoires nécessaires
echo "Création des répertoires de données..."
mkdir -p data/raw data/vectors data/models

# Finalisation
echo "Installation terminée!"
echo "Pour activer l'environnement: source .venv/bin/activate"
echo "Pour démarrer l'API: python3 -m uvicorn api.server:app --reload"
echo "Pour accéder à l'interface web: ouvrez web/index.html dans votre navigateur" 