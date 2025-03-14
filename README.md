# Système RAG Multimodal avec CLIP

Ce projet implémente un système de Retrieval Augmented Generation (RAG) multimodal qui utilise les embeddings CLIP pour représenter à la fois les images et le texte dans un même espace vectoriel.

## Caractéristiques

- **Embeddings unifiés** : Utilise CLIP (Contrastive Language-Image Pre-training) pour aligner les représentations textuelles et visuelles dans un même espace vectoriel
- **Requêtes cross-modales** : Interrogez avec du texte et obtenez des images pertinentes, ou inversement
- **Support multi-documents** : Traite les fichiers texte, images et PDF (avec extraction du texte et des images)
- **Base persistante** : Stockage des embeddings dans ChromaDB pour conserver les données entre les sessions
- **Interface utilisateur** : Interface en ligne de commande simple et intuitive

## Installation

### Prérequis

- Python 3.12+
- [Ollama](https://ollama.ai) installé et fonctionnel
- Modèles Ollama requis : llava, qwen2.5:3b (ou autres modèles compatibles)

### Installation des dépendances

```bash
pip install -r requirements.txt
```

## Utilisation

### Initialisation

Vous pouvez réinitialiser la base de données à tout moment :

```bash
python multimodal_rag_demo.py --reset
```

### Ajout de documents

Ajoutez des documents à la base de connaissances :

```bash
# Ajouter un PDF
python multimodal_rag_demo.py --add document.pdf

# Ajouter une image
python multimodal_rag_demo.py --add image.jpg
```

### Interrogation

Interrogez la base de connaissances avec du texte :

```bash
python multimodal_rag_demo.py --query "Que contient ce document ?"
```

Ou avec une image :

```bash
python multimodal_rag_demo.py --image_query image.jpg --query "Que représente cette image ?"
```

### Personnalisation

Vous pouvez personnaliser divers aspects du système :

```bash
# Utiliser un modèle Ollama spécifique
python multimodal_rag_demo.py --model llava:7b --query "Décris ce système"

# Spécifier un chemin de base de données différent
python multimodal_rag_demo.py --db_path /chemin/vers/db --query "Ma question"

# Utiliser une collection spécifique
python multimodal_rag_demo.py --collection ma_collection --query "Ma question"
```

## Fonctionnement interne

1. Les documents (texte, images, PDF) sont transformés en embeddings via CLIP
2. Ces embeddings sont stockés dans une base ChromaDB
3. Les requêtes sont également transformées en embeddings
4. Le système récupère les documents les plus similaires
5. Le LLM (via Ollama) génère une réponse basée sur les documents récupérés

## Notebook de démonstration

Un notebook Jupyter est disponible pour explorer les capacités du système de manière interactive :

```bash
jupyter notebook multimodal_rag_demo.ipynb
```

## Limitations actuelles

- Les modèles d'embedding et LLM fonctionnent localement, ce qui peut limiter les performances sur des machines moins puissantes
- Le traitement des PDF volumineux peut être lent
- La qualité des réponses dépend du modèle LLM utilisé

## Licence

MIT 