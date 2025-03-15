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
pip install -r requirements.txt
```

Principales nouvelles dépendances:
- faiss-cpu (ou faiss-gpu)
- cross-encoders
- annoy (optionnel, pour certains cas d'usage)

## Utilisation

### Script de démonstration amélioré

```bash
# Ajouter un document au système
python enhanced_multimodal_rag_demo.py --add-document path/to/document.pdf

# Effectuer une requête textuelle
python enhanced_multimodal_rag_demo.py --query "Que contient ce document?"

# Effectuer une requête par image
python enhanced_multimodal_rag_demo.py --image-query path/to/query_image.jpg

# Utiliser le GPU pour les calculs (si disponible)
python enhanced_multimodal_rag_demo.py --use-gpu --query "Ma requête"

# Désactiver le reranking pour cette requête
python enhanced_multimodal_rag_demo.py --no-reranking --query "Ma requête"

# Modifier le seuil de similarité
python enhanced_multimodal_rag_demo.py --similarity-threshold 0.3 --query "Ma requête"
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

## Contribuer

Les contributions pour améliorer davantage le système sont les bienvenues. Voici quelques pistes d'amélioration:

- Support pour d'autres modèles d'embedding multimodal
- Optimisation supplémentaire des index FAISS
- Ajout de techniques de clustering pour améliorer la diversité des résultats
- Implémentation de fonctionnalités de recherche sémantique hybride

## Licence

Ce projet est sous licence MIT. 