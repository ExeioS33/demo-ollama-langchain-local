# Documentation de la classe MultimodalVectorStore

## Description générale

`MultimodalVectorStore` est une classe qui gère le stockage et l'interrogation des embeddings multimodaux (texte et image) dans une base de données vectorielle. Cette classe fournit une interface unifiée pour ajouter différents types de documents (textes, images, PDF) et effectuer des recherches sémantiques sur ces contenus, indépendamment de leur type.

## Fonctionnalités clés

- Stockage persistant d'embeddings textuels et visuels dans un même espace
- Ajout de textes avec métadonnées
- Ajout d'images avec descriptions et métadonnées
- Extraction et indexation automatique du contenu des PDF (texte et images)
- Recherche par similarité sémantique avec du texte ou des images comme requête
- Filtrage des résultats par seuil de similarité et métadonnées

## Dépendances

- ChromaDB (base de données vectorielle)
- MultimodalEmbedder (pour générer les embeddings)
- PyMuPDF (fitz) pour l'extraction de contenu des PDF
- Pillow (PIL) pour le traitement d'images
- NumPy

## Initialisation

```python
from multimodal_rag import MultimodalVectorStore

# Avec les paramètres par défaut
vector_store = MultimodalVectorStore()

# Avec des paramètres personnalisés
vector_store = MultimodalVectorStore(
    collection_name="ma_collection", 
    persist_directory="/chemin/vers/stockage"
)
```

### Paramètres d'initialisation

| Paramètre | Type | Défaut | Description |
|-----------|------|--------|-------------|
| `collection_name` | str | "multimodal_collection" | Nom de la collection ChromaDB |
| `persist_directory` | str | "chroma_db" | Répertoire où stocker la base ChromaDB |

## Méthodes principales

### `add_texts(texts: List[str], metadatas: Optional[List[Dict]] = None) -> List[str]`

Ajoute des textes à la base vectorielle.

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

Ajoute des images à la base vectorielle.

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

### `add_pdf(pdf_path: str, extract_images: bool = True, page_overlap: int = 0, metadatas: Optional[Dict] = None) -> List[str]`

Ajoute un document PDF au magasin de vecteurs en extrayant son contenu textuel par page et optionnellement ses images.

#### Paramètres

| Paramètre | Type | Description |
|-----------|------|-------------|
| `pdf_path` | str | Chemin vers le fichier PDF |
| `extract_images` | bool | Si True, extrait aussi les images du PDF (défaut: True) |
| `page_overlap` | int | Nombre de lignes qui se chevauchent entre les pages (défaut: 0) |
| `metadatas` | Optional[Dict] | Métadonnées de base à associer à tous les éléments du PDF |

#### Retour

| Type | Description |
|------|-------------|
| List[str] | Liste des identifiants générés pour tous les éléments extraits du PDF |

#### Exemple d'utilisation

```python
# Ajouter un PDF avec extraction d'images
pdf_path = "documents/rapport.pdf"
ids = vector_store.add_pdf(pdf_path)

# Ajouter un PDF sans extraction d'images et avec métadonnées
pdf_path = "documents/livre.pdf"
metadatas = {"auteur": "Alice", "sujet": "Science", "confidentiel": True}
ids = vector_store.add_pdf(pdf_path, extract_images=False, metadatas=metadatas)
```

### `query(query: Union[str, Image.Image], top_k: int = 5, filter_metadata: Optional[Dict] = None) -> List[Dict]`

Interroge la base vectorielle avec une requête textuelle ou une image.

#### Paramètres

| Paramètre | Type | Description |
|-----------|------|-------------|
| `query` | Union[str, Image.Image] | Requête textuelle ou image |
| `top_k` | int | Nombre de résultats à retourner (défaut: 5) |
| `filter_metadata` | Optional[Dict] | Filtre à appliquer sur les métadonnées (facultatif) |

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

#### Exemple d'utilisation

```python
# Requête textuelle simple
results = vector_store.query("Comment fonctionne l'apprentissage profond ?")

# Requête textuelle avec limitation du nombre de résultats
results = vector_store.query("intelligence artificielle", top_k=10)

# Requête textuelle avec filtre sur les métadonnées
results = vector_store.query(
    "changement climatique", 
    filter_metadata={"auteur": "Alice"}
)

# Requête avec une image
from PIL import Image
image = Image.open("requete.jpg")
results = vector_store.query(image)

# Requête avec un chemin d'image
results = vector_store.query("images/requete.jpg")
```

## Attributs

| Attribut | Type | Description |
|----------|------|-------------|
| `embedder` | MultimodalEmbedder | Instance de l'embedder CLIP utilisé |
| `embedding_dim` | int | Dimensionnalité de l'espace d'embedding |
| `client` | chromadb.PersistentClient | Client ChromaDB pour la persistance |
| `collection` | chromadb.Collection | Collection ChromaDB utilisée |
| `image_ids` | set | Ensemble des IDs correspondant à des images |
| `metadata_store` | dict | Dictionnaire stockant des métadonnées supplémentaires |

## Détails techniques

### Gestion de la persistance

Les données sont stockées de manière persistante sur le disque grâce à ChromaDB, permettant de conserver les embeddings entre les sessions et de les réutiliser sans avoir à les recalculer.

### Embeddings personnalisés pour ChromaDB

La classe implémente une fonction d'embedding personnalisée `ClipEmbeddingFunction` pour ChromaDB, qui utilise l'embedder CLIP pour générer les représentations vectorielles.

### Métadonnées enrichies

Chaque document stocké possède deux niveaux de métadonnées :
1. Les métadonnées standard de ChromaDB
2. Des métadonnées enrichies dans `metadata_store` pour permettre une récupération plus détaillée

### Traitement spécifique des PDF

Pour les PDF, la classe :
1. Extrait le texte page par page
2. Extrait optionnellement les images
3. Associe à chaque élément extrait les métadonnées de page et de source
4. Stocke les images extraites dans un dossier temporaire

### Filtrage des résultats

Les résultats de recherche sont filtrés en fonction d'un seuil minimal de similarité (0.2 par défaut) pour ne retourner que les documents réellement pertinents.

## Exemple d'utilisation avancée

```python
from multimodal_rag import MultimodalVectorStore
from PIL import Image

# Initialiser la base vectorielle
vector_store = MultimodalVectorStore(
    collection_name="ma_base_documentaire",
    persist_directory="./ma_base_persistante"
)

# Ajouter divers types de documents
vector_store.add_texts(["Le contenu du premier document", "Le contenu du second document"])
vector_store.add_images(["image1.jpg", "image2.jpg"], ["Description de l'image 1", "Description de l'image 2"])
vector_store.add_pdf("document.pdf")

# Effectuer une requête textuelle avec filtre sur les métadonnées
results = vector_store.query(
    "Concept important", 
    top_k=3,
    filter_metadata={"type": "pdf_text"}
)

# Analyser les résultats
for i, result in enumerate(results):
    print(f"Résultat {i+1} - Score de similarité: {result['similarity']:.4f}")
    print(f"Type: {'Image' if result['is_image'] else 'Texte'}")
    print(f"Contenu: {result['content'][:100]}...")
    print(f"Source: {result['metadata'].get('source', 'Inconnue')}")
    print("---")
```

## Limites et considérations

- La qualité des résultats dépend de la qualité des embeddings générés par le modèle CLIP
- L'extraction d'images de PDFs complexes peut être imparfaite
- La taille de la base peut croître rapidement avec de nombreux documents volumineux
- Les performances de recherche peuvent diminuer à mesure que la collection grandit 