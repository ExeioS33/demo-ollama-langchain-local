# Documentation de la classe EnhancedVectorStore

## Description générale

`EnhancedVectorStore` est une classe qui implémente un magasin vectoriel optimisé utilisant FAISS (Facebook AI Similarity Search) pour stocker et rechercher des embeddings multimodaux (texte et image). Cette implémentation apporte des améliorations significatives en termes de performance, d'évolutivité et de précision par rapport à la version précédente basée sur ChromaDB.

## Fonctionnalités clés

- Stockage persistant d'embeddings textuels et visuels avec FAISS
- Recherche approximative rapide de plus proches voisins
- Support GPU pour accélérer les calculs (si disponible)
- Structures d'indexation optimisées (HNSW, IVF)
- Reranking avancé avec cross-encoders
- Gestion améliorée des seuils de similarité
- Utilisation de CLIP via transformers pour la génération d'embeddings
- Conversion automatique depuis ChromaDB

## Dépendances

- FAISS (faiss-cpu ou faiss-gpu)
- Transformers (Hugging Face)
- Torch
- Sentence-Transformers (pour CrossEncoder)
- PyMuPDF (fitz) pour le traitement des PDF
- Pillow (PIL) pour le traitement d'images
- NumPy, TensorFlow

## Initialisation

```python
from enhanced_vector_store import EnhancedVectorStore

# Avec les paramètres par défaut
vector_store = EnhancedVectorStore()

# Avec des paramètres personnalisés
vector_store = EnhancedVectorStore(
    collection_name="enhanced_collection",
    persist_directory="enhanced_vector_store",
    use_gpu=True,
    clip_model_name="openai/clip-vit-base-patch32",
    reranking_model="cross-encoder/ms-marco-MiniLM-L-6-v2"
)
```

### Paramètres d'initialisation

| Paramètre | Type | Défaut | Description |
|-----------|------|--------|-------------|
| `collection_name` | str | "enhanced_multimodal_collection" | Nom de la collection FAISS |
| `persist_directory` | str | "enhanced_vector_store" | Répertoire où stocker l'index FAISS |
| `use_gpu` | bool | False | Utiliser le GPU pour FAISS si disponible |
| `clip_model_name` | str | "ViT-B/32" | Nom du modèle CLIP à utiliser (converti automatiquement au format transformers) |
| `reranking_model` | str | "cross-encoder/ms-marco-MiniLM-L-6-v2" | Modèle de reranking à utiliser |

## Méthodes principales

### `add_texts(texts: List[str], metadatas: Optional[List[Dict]] = None) -> List[str]`

Ajoute des textes à l'index FAISS.

#### Paramètres

| Paramètre | Type | Description |
|-----------|------|-------------|
| `texts` | List[str] | Liste de textes à ajouter |
| `metadatas` | Optional[List[Dict]] | Métadonnées associées à chaque texte (facultatif) |

#### Retour

| Type | Description |
|------|-------------|
| List[str] | Liste des identifiants uniques générés pour les documents ajoutés |

#### Exemple d'utilisation

```python
# Ajouter des textes simples
texts = [
    "La programmation en Python est facile à apprendre.",
    "Les bases de données vectorielles permettent la recherche sémantique."
]
ids = vector_store.add_texts(texts)

# Ajouter des textes avec métadonnées
texts = ["Contenu du document A", "Contenu du document B"]
metadatas = [
    {"source": "document_A.txt", "auteur": "Alice"},
    {"source": "document_B.txt", "auteur": "Bob"}
]
ids = vector_store.add_texts(texts, metadatas)
```

### `add_images(images: List[Union[str, Image.Image]], descriptions: Optional[List[str]] = None, metadatas: Optional[List[Dict]] = None) -> List[str]`

Ajoute des images à l'index FAISS.

#### Paramètres

| Paramètre | Type | Description |
|-----------|------|-------------|
| `images` | List[Union[str, Image.Image]] | Liste de chemins d'images ou objets PIL.Image |
| `descriptions` | Optional[List[str]] | Descriptions textuelles des images (facultatif) |
| `metadatas` | Optional[List[Dict]] | Métadonnées associées à chaque image (facultatif) |

#### Retour

| Type | Description |
|------|-------------|
| List[str] | Liste des identifiants uniques générés pour les images ajoutées |

#### Exemple d'utilisation

```python
# Ajouter des images avec descriptions
image_paths = ["images/photo1.jpg", "images/photo2.jpg"]
descriptions = ["Un coucher de soleil à la plage", "Une montagne enneigée"]
ids = vector_store.add_images(image_paths, descriptions)

# Ajouter des images avec descriptions et métadonnées
image_paths = ["images/photo3.jpg", "images/photo4.jpg"]
descriptions = ["Une voiture rouge", "Un chat gris"]
metadatas = [
    {"photographe": "Alice", "date": "2023-04-15"},
    {"photographe": "Bob", "date": "2023-05-20"}
]
ids = vector_store.add_images(image_paths, descriptions, metadatas)
```

### `add_pdf(pdf_path: str, metadatas: Optional[Dict] = None) -> List[str]`

Ajoute un document PDF au magasin de vecteurs en extrayant son contenu textuel par page et ses images.

#### Paramètres

| Paramètre | Type | Description |
|-----------|------|-------------|
| `pdf_path` | str | Chemin vers le fichier PDF |
| `metadatas` | Optional[Dict] | Métadonnées de base à associer à tous les éléments du PDF |

#### Retour

| Type | Description |
|------|-------------|
| List[str] | Liste des identifiants générés pour tous les éléments extraits du PDF |

#### Exemple d'utilisation

```python
# Ajouter un PDF
pdf_path = "documents/rapport.pdf"
ids = vector_store.add_pdf(pdf_path)

# Ajouter un PDF avec métadonnées
pdf_path = "documents/livre.pdf"
metadatas = {"auteur": "Alice", "sujet": "Science", "confidentiel": True}
ids = vector_store.add_pdf(pdf_path, metadatas=metadatas)
```

### `query(query: Union[str, Image.Image], top_k: int = 5, filter_metadata: Optional[Dict] = None, use_reranking: bool = True) -> List[Dict]`

Interroge l'index FAISS avec une requête textuelle ou une image, avec option de reranking.

#### Paramètres

| Paramètre | Type | Description |
|-----------|------|-------------|
| `query` | Union[str, Image.Image] | Requête textuelle ou image |
| `top_k` | int | Nombre de résultats à retourner (défaut: 5) |
| `filter_metadata` | Optional[Dict] | Filtre à appliquer sur les métadonnées (facultatif) |
| `use_reranking` | bool | Si True, utilise le reranking pour améliorer les résultats (défaut: True) |

#### Retour

| Type | Description |
|------|-------------|
| List[Dict] | Liste des résultats avec leurs métadonnées et scores de similarité |

#### Structure des résultats

Chaque élément de la liste de résultats est un dictionnaire contenant :
- `id` : Identifiant unique du document
- `content` : Contenu textuel ou description de l'image
- `metadata` : Métadonnées associées au document
- `similarity` : Score de similarité (entre 0 et 1)
- `is_image` : Booléen indiquant s'il s'agit d'une image
- `rerank_score` : Score de reranking (présent uniquement si use_reranking=True)

#### Exemple d'utilisation

```python
# Requête textuelle simple
results = vector_store.query("Comment fonctionne l'apprentissage profond ?")

# Requête textuelle avec reranking désactivé
results = vector_store.query("intelligence artificielle", use_reranking=False)

# Requête textuelle avec filtre sur les métadonnées
results = vector_store.query(
    "changement climatique", 
    filter_metadata={"auteur": "Alice"}
)

# Requête avec une image et nombre de résultats personnalisé
from PIL import Image
image = Image.open("requete.jpg")
results = vector_store.query(image, top_k=10)
```

### `reset()`

Réinitialise l'index FAISS et les métadonnées.

#### Exemple d'utilisation

```python
# Réinitialiser complètement l'index
vector_store.reset()
```

## Attributs

| Attribut | Type | Description |
|----------|------|-------------|
| `index` | faiss.Index | Instance de l'index FAISS |
| `clip_model` | CLIPModel | Modèle CLIP chargé depuis transformers |
| `clip_processor` | CLIPProcessor | Processeur CLIP pour le prétraitement |
| `reranker` | CrossEncoder | Modèle de reranking (si spécifié) |
| `metadata` | List[Dict] | Liste des métadonnées pour chaque élément |
| `ids` | List[str] | Liste des identifiants dans l'index |
| `device` | str | Appareil utilisé pour l'inférence ("cuda" ou "cpu") |
| `embedding_dim` | int | Dimensionnalité de l'espace d'embedding |

## Méthodes d'embeddings

### `_get_text_embedding(text: str) -> np.ndarray`

Génère un embedding pour du texte en utilisant le modèle CLIP via transformers.

### `_get_image_embedding(image: Union[str, Image.Image]) -> np.ndarray`

Génère un embedding pour une image en utilisant le modèle CLIP via transformers.

## Détails techniques

### Types d'index FAISS

La classe utilise différents types d'index FAISS selon la disponibilité du GPU :

- **Avec GPU** : Index IVF (Inverted File) pour les grandes collections
- **Sans GPU** : Index HNSW (Hierarchical Navigable Small World) pour un bon compromis performance/précision

### Normalisation des vecteurs

Les embeddings sont normalisés avec la norme L2, ce qui permet d'utiliser le produit scalaire comme mesure de similarité cosinus.

### Reranking

Le reranking est effectué avec un modèle CrossEncoder de la bibliothèque Sentence-Transformers :

1. Les résultats initiaux sont obtenus avec FAISS (en quantité top_k * 3)
2. Les paires (requête, document) sont évaluées par le CrossEncoder
3. Les résultats sont réordonnés selon les scores de reranking

### Conversion depuis ChromaDB

La classe fournit une fonction utilitaire `convert_chromadb_to_faiss` pour migrer des collections existantes depuis ChromaDB :

```python
from enhanced_vector_store import convert_chromadb_to_faiss

new_store = convert_chromadb_to_faiss(
    chroma_collection_name="multimodal_collection",
    chroma_persist_directory="chroma_db", 
    output_directory="enhanced_vector_store",
    use_gpu=False
)
```

## Performances

| Configuration | Caractéristiques |
|---------------|------------------|
| **CPU, petite collection** | Index HNSW, rapide pour les petites collections |
| **CPU, grande collection** | Index HNSW, performances acceptables |
| **GPU, petite collection** | Index IVF, overhead de transfert GPU, peu d'avantage |
| **GPU, grande collection** | Index IVF sur GPU, performances optimales pour les grandes collections |

## Avantages par rapport à la version ChromaDB

1. **Meilleure évolutivité** : FAISS permet d'indexer des millions de vecteurs efficacement
2. **Recherche plus rapide** : Algorithmes optimisés pour la recherche approximative (ANN)
3. **Support GPU natif** : Calculs vectoriels accélérés sur GPU
4. **Reranking avancé** : Amélioration significative de la pertinence des résultats
5. **Flexibilité des index** : Différents types d'index selon les besoins (HNSW, IVF, etc.)
6. **Intégration moderne avec transformers** : Utilisation de CLIP via la bibliothèque transformers

## Limitations

- Nécessite plus de configuration manuelle que ChromaDB
- Les performances optimales requièrent un GPU pour les grandes collections
- La structure de stockage est plus complexe (plusieurs fichiers)
- Pas d'interface HTTP intégrée (contrairement à ChromaDB)

## Exemple d'utilisation avancée

```python
import numpy as np
from enhanced_vector_store import EnhancedVectorStore
from PIL import Image

# Initialiser avec GPU et modèle de reranking personnalisé
vector_store = EnhancedVectorStore(
    collection_name="documentation_technique",
    persist_directory="vector_db",
    use_gpu=True,
    clip_model_name="openai/clip-vit-large-patch14", 
    reranking_model="cross-encoder/ms-marco-MiniLM-L-12-v2"
)

# Ajouter du contenu
vector_store.add_texts(["Documentation technique du système", "Guide d'utilisation"])
vector_store.add_images(["schemas/architecture.png"], ["Schéma d'architecture du système"])
vector_store.add_pdf("documentation_complete.pdf")

# Recherche avec et sans reranking pour comparer
results_with_reranking = vector_store.query("Comment configurer le système?", top_k=3)
results_without_reranking = vector_store.query("Comment configurer le système?", top_k=3, use_reranking=False)

# Afficher les différences
print("=== Avec reranking ===")
for i, result in enumerate(results_with_reranking):
    print(f"{i+1}. {result['content'][:50]}... (score: {result['rerank_score']:.2f})")

print("\n=== Sans reranking ===")
for i, result in enumerate(results_without_reranking):
    print(f"{i+1}. {result['content'][:50]}... (score: {result['similarity']:.2f})")
``` 