# Documentation de la classe MultimodalEmbedder

## Description générale

`MultimodalEmbedder` est une classe qui génère des représentations vectorielles (embeddings) pour du texte et des images en utilisant le modèle CLIP (Contrastive Language-Image Pre-training). Cette classe permet de transformer différentes modalités de données (texte et images) dans un même espace vectoriel, ce qui facilite la comparaison et la recherche de similarités entre ces différents types de contenus.

## Fonctionnalités clés

- Génération d'embeddings pour des textes
- Génération d'embeddings pour des images (à partir de fichiers, URLs ou objets PIL)
- Normalisation des embeddings pour la recherche par similarité cosinus
- Détection automatique du matériel disponible (CPU/GPU)

## Dépendances

- PyTorch
- Transformers (Hugging Face)
- Pillow (PIL)
- NumPy
- Requests (pour les images depuis URLs)

## Initialisation

```python
from multimodal_rag import MultimodalEmbedder

# Avec le modèle par défaut
embedder = MultimodalEmbedder()

# Avec un modèle CLIP personnalisé
embedder = MultimodalEmbedder(model_name="laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
```

### Paramètres d'initialisation

| Paramètre | Type | Défaut | Description |
|-----------|------|--------|-------------|
| `model_name` | str | "openai/clip-vit-base-patch32" | Nom ou chemin du modèle CLIP à utiliser |

## Méthodes principales

### `embed_text(texts: List[str]) -> np.ndarray`

Génère des embeddings pour une liste de textes.

#### Paramètres

| Paramètre | Type | Description |
|-----------|------|-------------|
| `texts` | List[str] | Liste de textes à encoder |

#### Retour

| Type | Description |
|------|-------------|
| np.ndarray | Matrice d'embeddings normalisés (N × embedding_dim) |

#### Exemple d'utilisation

```python
texts = [
    "Un chien qui court dans un parc",
    "Une voiture rouge",
    "Un coucher de soleil sur la plage"
]
text_embeddings = embedder.embed_text(texts)
print(f"Forme des embeddings: {text_embeddings.shape}")
```

### `embed_image(images: List[Union[str, Image.Image]]) -> np.ndarray`

Génère des embeddings pour une liste d'images.

#### Paramètres

| Paramètre | Type | Description |
|-----------|------|-------------|
| `images` | List[Union[str, Image.Image]] | Liste de chemins d'images, URLs ou objets PIL.Image |

#### Retour

| Type | Description |
|------|-------------|
| np.ndarray | Matrice d'embeddings normalisés (N × embedding_dim) |

#### Exemple d'utilisation

```python
# Avec des chemins de fichiers
image_paths = [
    "images/chien.jpg",
    "images/voiture.jpg", 
    "images/coucher_soleil.jpg"
]
image_embeddings = embedder.embed_image(image_paths)

# Avec des URLs
image_urls = [
    "https://example.com/image1.jpg",
    "https://example.com/image2.jpg"
]
image_embeddings = embedder.embed_image(image_urls)

# Avec des objets PIL.Image
from PIL import Image
images = [Image.open("image1.jpg"), Image.open("image2.jpg")]
image_embeddings = embedder.embed_image(images)
```

## Attributs

| Attribut | Type | Description |
|----------|------|-------------|
| `device` | str | Appareil utilisé pour l'inférence ("cuda" pour GPU, "cpu" pour CPU) |
| `model` | CLIPModel | Instance du modèle CLIP chargé |
| `processor` | CLIPProcessor | Processeur associé au modèle CLIP |
| `embedding_dim` | int | Dimensionnalité de l'espace d'embedding (512 pour le modèle par défaut) |

## Détails techniques

### Prétraitement des images

La classe effectue automatiquement plusieurs opérations de prétraitement sur les images :
1. Chargement des images depuis différentes sources (fichiers locaux, URLs)
2. Conversion en mode RGB si nécessaire
3. Redimensionnement et normalisation via le processeur CLIP

### Normalisation des embeddings

Les embeddings générés sont normalisés avec la norme L2, ce qui les place sur une hypersphère unitaire. Cette normalisation est essentielle pour calculer des similarités cosinus entre les vecteurs.

### Performances

La détection automatique de GPU permet d'accélérer considérablement le calcul des embeddings sur les machines disposant de cartes graphiques compatibles CUDA.

## Exemple d'utilisation avancé

```python
import numpy as np
from multimodal_rag import MultimodalEmbedder

# Initialiser l'embedder
embedder = MultimodalEmbedder()

# Préparer des textes et des images
texts = ["Un chien", "Un chat", "Une voiture"]
images = ["dog.jpg", "cat.jpg", "car.jpg"]

# Calculer les embeddings
text_embeddings = embedder.embed_text(texts)
image_embeddings = embedder.embed_image(images)

# Calculer la matrice de similarité entre textes et images
similarity_matrix = np.dot(text_embeddings, image_embeddings.T)

# Afficher les meilleures correspondances
for i, text in enumerate(texts):
    best_match_idx = np.argmax(similarity_matrix[i])
    print(f"Le texte '{text}' correspond le mieux à l'image: {images[best_match_idx]}")
    print(f"Score de similarité: {similarity_matrix[i, best_match_idx]:.4f}")
```

## Limites et considérations

- Les performances dépendent de la qualité et de la taille du modèle CLIP utilisé
- Le traitement par lots de grandes quantités d'images peut nécessiter beaucoup de mémoire GPU
- Le modèle par défaut a une limite de 77 tokens pour les textes
- Les embeddings générés reflètent les biais présents dans les données d'entraînement du modèle CLIP 