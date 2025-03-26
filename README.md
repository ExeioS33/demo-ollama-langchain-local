# Système RAG Multimodal (POC)

Un système de Retrieval Augmented Generation multimodal permettant de traiter et interroger des données textuelles et des images à l'aide de modèles avancés comme LLaVA.

## Structure du Projet

```
📁 multimodal-poc/
├── 📁 core/                # Cœur fonctionnel RAG
│   ├── 📄 rag.py           # Pipeline principal
│   ├── 📄 embeddings.py    # Génération d'embeddings
│   ├── 📄 llm.py           # Intégration LLaVA
│   └── 📄 vector_operations.py # Opérations FAISS
│
├── 📁 data/                # Données et modèles
│   ├── 📁 raw/             # Données brutes
│   ├── 📁 vectors/         # Index FAISS
│   └── 📁 models/          # Modèles pré-entraînés
│
├── 📁 api/                 # API FastAPI
│   └── 📄 server.py        # Endpoints de l'API
│
├── 📁 web/                 # Interface utilisateur
│   └── 📄 index.html       # Interface web simple
│
├── 📁 scripts/             # Scripts utilitaires
│   ├── 📄 setup.sh         # Installation des dépendances avec UV
│   ├── 📄 remote_setup.sh  # Configuration depuis dépôt distant
│   ├── 📄 ingest.py        # Ingestion de données
│   └── 📄 compare_performance.py # Benchmark des implémentations
│
└── 📁 config/              # Configuration
    └── 📄 config.py        # Paramètres globaux
```

## Fonctionnalités

- Traitement et indexation de documents textuels et d'images
- Recherche sémantique multimodale avec FAISS
- Intégration avec LLaVA via Ollama pour le traitement d'images
- API REST avec FastAPI
- Interface web simple pour les démonstrations

## Prérequis

- Python 3.9+
- UV (gestionnaire de paquets Python rapide)
- Ollama pour l'exécution de LLaVA

## Installation

### Pour un nouveau projet

```bash
# Cloner le dépôt
git clone <repo-url>
cd multimodal-poc

# Exécuter le script d'installation
./scripts/setup.sh
```

### Depuis un dépôt existant

```bash
# Option 1: Spécifier l'URL du dépôt à cloner
./scripts/remote_setup.sh https://github.com/username/multimodal-poc.git

# Option 2: Dans un dépôt déjà cloné
cd multimodal-poc
./scripts/remote_setup.sh
```

## Utilisation

### Démarrer l'API

```bash
source .venv/bin/activate  # Activer l'environnement virtuel
uv run uvicorn api.server:app --reload
```

### Ingérer des documents

```bash
# Ingérer un document PDF
uv run python scripts/ingest.py --input chemin/vers/document.pdf

# Ingérer un répertoire d'images
uv run python scripts/ingest.py --input chemin/vers/images/ --pattern "*.jpg" --recursive
```

### Interface Web

Ouvrez le fichier `web/index.html` dans votre navigateur pour accéder à l'interface utilisateur.

### Comparaison de performance

```bash
# Comparer les performances des deux implémentations
uv run python scripts/compare_performance.py --query "Votre question" --image chemin/vers/image.jpg
```

## Paramètres de Configuration

Les paramètres du système peuvent être configurés via des variables d'environnement:

- `USE_GPU`: Utiliser le GPU (true/false)
- `LLM_MODEL`: Modèle LLM à utiliser (ex: llava:7b-v1.6-vicuna-q8_0)
- `EMBEDDING_MODEL`: Modèle d'embedding (ex: openai/clip-vit-base-patch32)
- `API_PORT`: Port pour l'API (défaut: 8000)

## Licence

Ce projet est sous licence MIT. 