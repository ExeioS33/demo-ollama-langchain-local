# Optimisation du Système RAG Multimodal

Ce projet implémente une version optimisée du système RAG (Retrieval Augmented Generation) multimodal, avec des améliorations significatives pour la recherche de similarité et le reranking des résultats.

## Améliorations apportées

### 1. Utilisation de FAISS pour la recherche approximative

Le système original utilisait ChromaDB pour stocker et rechercher des vecteurs. Cette nouvelle implémentation utilise [FAISS (Facebook AI Similarity Search)](https://github.com/facebookresearch/faiss), une bibliothèque optimisée pour la recherche de similitude à grande échelle:

- **Recherche approximative de voisins** plus rapide et plus précise
- **Mise à l'échelle améliorée** pour les grandes collections de documents
- **Support GPU** pour accélérer les calculs (si disponible)
- Structures d'indexation optimisées (HNSW, IVF)

### 2. Reranking avancé avec Cross-Encoders

La nouvelle implémentation intègre un système de reranking basé sur les cross-encoders de la bibliothèque SentenceTransformers:

- Évaluation plus précise de la pertinence des documents candidats
- Reclassement des résultats selon une mesure de pertinence plus sophistiquée
- Amélioration significative de la qualité des résultats retournés

### 3. Gestion améliorée des seuils de similarité

Le système optimisé offre:

- Seuils de similarité configurables
- Filtrage dynamique des résultats
- Meilleure visibilité des scores de similarité et de reranking

### 4. Compatibilité avec l'ancien système

Une fonction de migration permet de convertir les collections ChromaDB existantes vers le nouveau format FAISS:

```bash
python enhanced_multimodal_rag_demo.py --migrate --chroma-path chroma_db --chroma-collection multimodal_collection
```

### 5. Utilisation de CLIP via Transformers

Le système utilise désormais l'implémentation de CLIP de la bibliothèque Transformers de Hugging Face au lieu du package CLIP original:

- Meilleure maintenance et compatibilité
- Intégration plus facile avec l'écosystème Hugging Face
- Support pour plus de modèles CLIP pré-entraînés
- Conversion automatique des noms de modèles (ex: "ViT-B/32" → "openai/clip-vit-base-patch32")

### 6. Compatibilité avec UV

Le système est maintenant compatible avec l'environnement UV, un gestionnaire de paquets Python ultra-rapide:

- Utilisation de `uv run` pour exécuter les scripts Python
- Utilisation de `uv pip` pour installer les dépendances
- Commandes de vérification adaptées à l'environnement UV
- Meilleure gestion des dépendances qui nécessitent une compilation

## Architecture

Le système optimisé comprend trois composants principaux:

1. **`EnhancedVectorStore`**: Magasin vectoriel optimisé utilisant FAISS
   - Gestion des embeddings texte et image
   - Indexation performante et recherche approximative
   - Support du reranking

2. **`EnhancedMultimodalRAG`**: Système RAG amélioré
   - Intégration avec le magasin vectoriel optimisé
   - Gestion améliorée du contexte pour le LLM
   - Support pour les requêtes textuelles et visuelles

3. **Outils de migration**: Pour passer de ChromaDB à FAISS
   - Préservation des données existantes
   - Conversion transparente des métadonnées

## Installation

Les nouvelles dépendances ont été ajoutées dans le fichier `requirements.txt`:

```bash
# Avec pip standard
pip install -r requirements.txt

# Avec UV (recommandé pour de meilleures performances)
uv pip install -r requirements.txt
```

Principales nouvelles dépendances:
- faiss-cpu (ou faiss-gpu)
- transformers (pour CLIP)
- sentence-transformers (pour CrossEncoder)
- cross-encoders (optionnel, pour le reranking)

### Installation avec le Makefile

Un Makefile amélioré est fourni pour faciliter l'installation et l'utilisation:

```bash
# Installation standard
make -f Makefile.enhanced install

# Installation sans packages nécessitant une compilation
make -f Makefile.enhanced install-no-compile

# Vérifier les dépendances
make -f Makefile.enhanced check-deps

# Afficher les informations de configuration
make -f Makefile.enhanced config-info
```

## Utilisation

### Script de démonstration amélioré

```bash
# Avec Python standard
python enhanced_multimodal_rag_demo.py --add-document path/to/document.pdf

# Avec UV
uv run enhanced_multimodal_rag_demo.py --add-document path/to/document.pdf

# Avec le Makefile (utilise UV automatiquement)
make -f Makefile.enhanced add-document
```

Autres commandes:

```bash
# Effectuer une requête textuelle
make -f Makefile.enhanced query

# Effectuer une requête par image
make -f Makefile.enhanced image-query

# Exécuter une démo complète
make -f Makefile.enhanced demo
```

### Utilisation dans votre code

```python
from enhanced_multimodal_rag import EnhancedMultimodalRAG

# Initialiser le système
rag = EnhancedMultimodalRAG(
    llm_name="qwen2.5:3b",  # Modèle Ollama à utiliser
    use_gpu=True,           # Utiliser GPU si disponible
    similarity_threshold=0.25  # Seuil de similarité pour les résultats
)

# Ajouter des documents
rag.add_document("path/to/document.pdf")
rag.add_document("path/to/image.jpg", description="Description de l'image")

# Requête textuelle
results = rag.query("Quelle information est présente dans ces documents?")
print(results["answer"])

# Requête par image
from PIL import Image
image = Image.open("path/to/query_image.jpg")
results = rag.query(image)
print(results["answer"])
```

## Comparaison des performances

| Fonctionnalité | Système original | Système optimisé |
|----------------|------------------|------------------|
| Indexation     | ChromaDB         | FAISS            |
| Recherche      | Distance euclidienne | HNSW/IVF avec produit scalaire |
| Reranking      | Manuel           | Cross-encoders   |
| Vitesse (grande collection) | Modérée | Rapide (surtout avec GPU) |
| Précision      | Bonne            | Excellente       |
| Support GPU    | Non              | Oui              |
| Migration      | N/A              | Oui              |
| Embeddings     | CLIP (package original) | CLIP via Transformers |
| Gestionnaire de paquets | pip | pip ou UV |

## Utilisation avancée

### Personnalisation du reranking

```python
rag = EnhancedMultimodalRAG(
    reranking_model="cross-encoder/ms-marco-MiniLM-L-12-v2"  # Modèle plus grand, plus précis
)
```

### Configuration optimale pour grandes collections

```python
rag = EnhancedMultimodalRAG(
    use_gpu=True,  # Crucial pour les grandes collections
    persist_directory="faiss_index_large"
)
```

### Utilisation avec différents modèles CLIP

```python
# Avec un modèle CLIP plus grand (meilleure qualité, plus lent)
rag = EnhancedMultimodalRAG(
    clip_model_name="openai/clip-vit-large-patch14"
)

# Avec un modèle CLIP plus petit (plus rapide, qualité moindre)
rag = EnhancedMultimodalRAG(
    clip_model_name="openai/clip-vit-base-patch16"
)
```

## Contribuer

Les contributions pour améliorer davantage le système sont les bienvenues. Voici quelques pistes d'amélioration:

- Support pour d'autres modèles d'embedding multimodal
- Optimisation supplémentaire des index FAISS
- Ajout de techniques de clustering pour améliorer la diversité des résultats
- Implémentation de fonctionnalités de recherche sémantique hybride

## Licence

Ce projet est sous licence MIT. 