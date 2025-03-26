#!/bin/bash

# Script d'installation pour le système RAG multimodal avec UV
# -----------------------------------------------------------

echo "Installation des dépendances du système RAG multimodal avec UV..."

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

# Vérifier si UV est installé
if ! command -v uv &> /dev/null; then
    echo "UV n'est pas installé. Installation en cours..."
    # On macOS and Linux.
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
    
    # Vérifier à nouveau
    if ! command -v uv &> /dev/null; then
        echo "Échec de l'installation de UV. Veuillez l'installer manuellement: https://github.com/astral-sh/uv"
        exit 1
    fi
    echo "UV installé avec succès!"
else
    echo "UV déjà installé."
fi

# Créer un environnement virtuel avec UV
if [ ! -d ".venv" ]; then
    echo "Création de l'environnement virtuel avec UV..."
    uv venv .venv
fi

# Activer l'environnement virtuel
echo "Activation de l'environnement virtuel..."
source .venv/bin/activate

# Installer les dépendances avec UV
echo "Installation des dépendances avec UV..."
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
uv pip install transformers sentence-transformers pillow faiss-cpu pymupdf
uv pip install fastapi uvicorn python-multipart pydantic

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

# Générer requirements.txt pour référence
echo "Génération du fichier requirements.txt..."
uv pip freeze > requirements.txt

# Finalisation
echo "Installation terminée!"
echo "Pour activer l'environnement: source .venv/bin/activate"
echo "Pour démarrer l'API: uv run uvicorn api.server:app --reload"
echo "Pour accéder à l'interface web: ouvrez web/index.html dans votre navigateur" 