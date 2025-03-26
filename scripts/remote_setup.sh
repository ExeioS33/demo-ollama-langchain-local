#!/bin/bash

# Script de configuration pour la récupération du projet depuis le dépôt distant
# -----------------------------------------------------------------------------

echo "Configuration du système RAG multimodal récupéré depuis le dépôt distant..."

# Vérifier si Git est installé
if ! command -v git &> /dev/null; then
    echo "Git n'est pas installé. Veuillez l'installer pour récupérer le projet."
    exit 1
fi

# Vérifier si UV est installé
if ! command -v uv &> /dev/null; then
    echo "UV n'est pas installé. Installation en cours..."
    curl -sSf https://install.ultraviolet.rs | sh
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

# Cloner le dépôt si le répertoire n'existe pas
if [ "$1" ]; then
    REPO_URL=$1
    PROJECT_DIR=$(basename "$REPO_URL" .git)
    
    if [ ! -d "$PROJECT_DIR" ]; then
        echo "Clonage du dépôt $REPO_URL..."
        git clone "$REPO_URL"
        cd "$PROJECT_DIR"
    else
        echo "Le dossier $PROJECT_DIR existe déjà. Utilisation de celui-ci..."
        cd "$PROJECT_DIR"
        echo "Mise à jour du dépôt..."
        git pull
    fi
else
    echo "Utilisation du répertoire courant..."
fi

# Créer et activer l'environnement virtuel
if [ ! -d ".venv" ]; then
    echo "Création de l'environnement virtuel avec UV..."
    uv venv .venv
fi

echo "Activation de l'environnement virtuel..."
source .venv/bin/activate

# Installer les dépendances avec UV
if [ -f "requirements.txt" ]; then
    echo "Installation des dépendances depuis requirements.txt avec UV..."
    uv pip install -r requirements.txt
else
    echo "Aucun fichier requirements.txt trouvé. Installation des dépendances de base..."
    # Installer les dépendances principales
    uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    uv pip install transformers sentence-transformers pillow faiss-cpu pymupdf
    uv pip install fastapi uvicorn python-multipart pydantic
fi

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
echo "Configuration terminée!"
echo ""
echo "Pour activer l'environnement: source .venv/bin/activate"
echo "Pour démarrer l'API: uv run uvicorn api.server:app --reload"
echo "Pour exécuter des scripts: uv run python scripts/ingest.py ..."
echo "Pour accéder à l'interface web: ouvrez web/index.html dans votre navigateur"
echo ""
echo "Utilisation: ./scripts/remote_setup.sh [URL_DU_DEPOT]"
echo "  - Si l'URL est fournie, le script clone ou met à jour le dépôt"
echo "  - Sinon, il utilise le répertoire courant" 