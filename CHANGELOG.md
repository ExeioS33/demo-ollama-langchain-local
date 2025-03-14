# Changelog

## [1.0.0] - 2025-03-14

### Ajouté
- Système RAG multimodal avec embeddings CLIP pour représenter texte et images dans le même espace vectoriel
- Support pour l'analyse de texte, images et PDF
- Interface en ligne de commande via `multimodal_rag_demo.py`
- Stockage persistant avec ChromaDB
- Notebook de démonstration `multimodal_rag_demo.ipynb`

### Améliorations techniques
- Utilisation de `langchain_community.llms.Ollama` pour une meilleure compatibilité
- Intégration du modèle CLIP (openai/clip-vit-base-patch32)
- Recherche contextuelle améliorée avec filtrage de similarité (seuil à 0.2)
- Système de prompt optimisé pour éviter les hallucinations
- Gestion des erreurs et logs détaillés

### Fonctionnalités de l'interface
- Option `--reset` pour réinitialiser la base de données
- Support pour les requêtes textuelles (`--query`)
- Support pour les requêtes avec images (`--image_query`)
- Personnalisation du modèle LLM via `--model`
- Personnalisation du chemin de la base de données et du nom de collection

### Dépendances
- Python 3.12
- LangChain
- ChromaDB
- PyTorch
- Transformers (CLIP)
- PyMuPDF (pour extraction de texte et images des PDF)
- Ollama (serveur local de modèles)

### Optimisations
- Normalisation L2 des embeddings
- Fonctionnement possible sur CPU ou GPU
- Stockage des métadonnées enrichies pour une meilleure contextualisation 