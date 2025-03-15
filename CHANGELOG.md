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

## [Unreleased]

### Optimisation de la recherche de similarité (branche: similarity_optimization)

#### Ajout
- Nouveau module `enhanced_vector_store.py` avec implémentation FAISS pour la recherche approximative
- Nouveau module `enhanced_multimodal_rag.py` optimisé pour la précision des résultats
- Script de démonstration amélioré `enhanced_multimodal_rag_demo.py`
- Fonction de migration pour convertir les collections ChromaDB existantes vers FAISS
- Support GPU pour accélérer les calculs vectoriels (si disponible)
- Documentation détaillée des optimisations dans `README_OPTIMISATION.md`

#### Changements
- Remplacement de ChromaDB par FAISS pour la recherche de similarité
- Intégration de cross-encoders pour le reranking avancé des résultats
- Amélioration de la gestion des seuils de similarité
- Options de configuration étendues (GPU, modèle de reranking, etc.)

#### Performance
- Recherche plus rapide, particulièrement sur les grandes collections
- Précision des résultats significativement améliorée
- Meilleure mise à l'échelle pour les grands volumes de données 