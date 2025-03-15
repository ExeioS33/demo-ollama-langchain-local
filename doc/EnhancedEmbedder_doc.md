# Documentation de la classe EnhancedEmbedder

## Description générale

`EnhancedEmbedder` est une classe qui encapsule les fonctionnalités d'embeddings multimodaux en utilisant les modèles CLIP (Contrastive Language-Image Pretraining) de Hugging Face Transformers. Cette implémentation améliorée remplace l'ancienne approche basée sur le package CLIP original, offrant une meilleure robustesse, compatibilité et flexibilité pour convertir à la fois du texte et des images en vecteurs d'embedding de haute qualité ayant un espace de représentation commun.

## Fonctionnalités clés

- Génération d'embeddings textuels et visuels dans un même espace vectoriel
- Support pour différents modèles CLIP (ViT-B/32, ViT-L/14, etc.)
- Normalisation vectorielle pour une meilleure précision des recherches par similarité
- Support GPU optionnel pour des performances accrues
- Traitement par lots (batching) pour une meilleure efficacité avec de grands ensembles de données
- Persistance des modèles chargés pour éviter les rechargements inutiles
- Compatibilité avec l'infrastructure LangChain pour une intégration aisée
- Gestion robuste des erreurs avec remontée d'informations détaillées

## Dépendances

- Transformers (Hugging Face)
- Torch
- Pillow (PIL)
- NumPy
- UV (gestionnaire de paquets Python recommandé)

## Initialisation

```python
from enhanced_embedder import EnhancedEmbedder

# Avec les paramètres par défaut
embedder = EnhancedEmbedder()

# Avec des paramètres personnalisés
embedder = EnhancedEmbedder(
    model_name="openai/clip-vit-large-patch14",
    use_gpu=True,
    batch_size=32
)
```

### Paramètres d'initialisation

| Paramètre | Type | Défaut | Description |
|-----------|------|--------|-------------|
| `model_name` | str | "openai/clip-vit-base-patch32" | Nom du modèle CLIP à utiliser |
| `use_gpu` | bool | False | Utiliser le GPU pour le modèle CLIP si disponible |
| `batch_size` | int | 16 | Taille des lots pour le traitement par lots |

## Méthodes principales

### `embed_texts(texts: List[str]) -> np.ndarray`

Génère des embeddings pour une liste de textes.

#### Paramètres

| Paramètre | Type | Description |
|-----------|------|-------------|
| `texts` | List[str] | Liste de chaînes de texte à convertir en embeddings |

#### Retour

| Type | Description |
|------|-------------|
| np.ndarray | Tableau NumPy contenant les embeddings de dimension [n_texts, embedding_dim] |

#### Exemple d'utilisation

```python
texts = [
    "Un système RAG multimodal avancé",
    "Architecture d'un système de recherche vectorielle",
    "Comment fonctionne le modèle CLIP?"
]

# Obtenir les embeddings textuels
text_embeddings = embedder.embed_texts(texts)
print(f"Forme des embeddings textuels: {text_embeddings.shape}")
```

### `embed_images(images: List[Image.Image]) -> np.ndarray`

Génère des embeddings pour une liste d'images.

#### Paramètres

| Paramètre | Type | Description |
|-----------|------|-------------|
| `images` | List[Image.Image] | Liste d'objets Image PIL à convertir en embeddings |

#### Retour

| Type | Description |
|------|-------------|
| np.ndarray | Tableau NumPy contenant les embeddings de dimension [n_images, embedding_dim] |

#### Exemple d'utilisation

```python
from PIL import Image

images = [
    Image.open("images/system_diagram.jpg"),
    Image.open("images/architecture.png")
]

# Obtenir les embeddings d'images
image_embeddings = embedder.embed_images(images)
print(f"Forme des embeddings d'images: {image_embeddings.shape}")
```

### `embed_image(image: Image.Image) -> np.ndarray`

Génère un embedding pour une seule image.

#### Paramètres

| Paramètre | Type | Description |
|-----------|------|-------------|
| `image` | Image.Image | Image PIL à convertir en embedding |

#### Retour

| Type | Description |
|------|-------------|
| np.ndarray | Tableau NumPy contenant l'embedding de dimension [embedding_dim] |

#### Exemple d'utilisation

```python
from PIL import Image

# Charger une image
image = Image.open("query_image.jpg")

# Obtenir l'embedding
embedding = embedder.embed_image(image)
print(f"Dimension de l'embedding: {embedding.shape}")
```

### `embed_text(text: str) -> np.ndarray`

Génère un embedding pour un seul texte.

#### Paramètres

| Paramètre | Type | Description |
|-----------|------|-------------|
| `text` | str | Chaîne de texte à convertir en embedding |

#### Retour

| Type | Description |
|------|-------------|
| np.ndarray | Tableau NumPy contenant l'embedding de dimension [embedding_dim] |

#### Exemple d'utilisation

```python
# Générer un embedding pour une requête
query_embedding = embedder.embed_text("Comment fonctionne le système RAG?")
print(f"Dimension de l'embedding: {query_embedding.shape}")
```

### `get_embedding_dimension() -> int`

Retourne la dimension des embeddings générés par le modèle.

#### Retour

| Type | Description |
|------|-------------|
| int | Dimension des embeddings (généralement 512 ou 768 selon le modèle) |

#### Exemple d'utilisation

```python
# Obtenir la dimension des embeddings
dim = embedder.get_embedding_dimension()
print(f"Dimension des embeddings du modèle {embedder.model_name}: {dim}")
```

## Attributs

| Attribut | Type | Description |
|----------|------|-------------|
| `model` | CLIPModel | Instance du modèle CLIP |
| `processor` | CLIPProcessor | Instance du processeur CLIP |
| `device` | torch.device | Périphérique de calcul (CPU ou CUDA) |
| `model_name` | str | Nom du modèle CLIP utilisé |
| `batch_size` | int | Taille des lots pour le traitement par lots |
| `embedding_dimension` | int | Dimension des embeddings générés |

## Détails techniques

### Modèles CLIP supportés

Les modèles CLIP suivants sont testés et supportés :

| Nom du modèle | Dimension | Taille | Caractéristiques |
|---------------|-----------|--------|------------------|
| openai/clip-vit-base-patch32 | 512 | ~150 MB | Équilibre performance/rapidité (défaut) |
| openai/clip-vit-base-patch16 | 512 | ~150 MB | Meilleure précision que patch32 |
| openai/clip-vit-large-patch14 | 768 | ~350 MB | Haute précision, plus lent |
| laion/CLIP-ViT-H-14-laion2B-s32B-b79K | 1024 | ~1 GB | Très haute précision, beaucoup plus lent |

### Traitement par lots (Batching)

Le traitement par lots est utilisé pour optimiser les performances lors du traitement de grands ensembles de données. Le paramètre `batch_size` contrôle ce comportement :

- Une valeur plus élevée utilise plus de mémoire mais offre un traitement plus rapide
- Une valeur plus basse est plus économe en mémoire mais peut ralentir le traitement
- La valeur optimale dépend de votre matériel (particulièrement important pour le GPU)

### Normalisation des vecteurs

Tous les embeddings sont normalisés à une norme L2 unitaire, ce qui signifie :
1. Les vecteurs ont une magnitude (norme) de 1.0
2. La similarité cosinus peut être calculée avec un simple produit scalaire
3. La recherche par similarité est plus précise et plus stable

### Optimisations GPU

Lorsque `use_gpu=True` et qu'un GPU compatible CUDA est disponible :
- Les modèles sont chargés directement en mémoire GPU
- Les tenseurs d'entrée sont transférés sur GPU avant le calcul
- Les embeddings résultants sont renvoyés sur CPU sous forme de tableaux NumPy
- Les calculs par lots sont optimisés pour la parallélisation GPU

### Gestion des erreurs et exceptions

La classe implémente une gestion robuste des erreurs :
- `ValueError` pour les entrées invalides (textes vides, images corrompues)
- `RuntimeError` pour les erreurs liées au modèle ou au matériel
- `MemoryError` pour les cas où la mémoire est insuffisante (réduire batch_size)
- Messages d'erreur détaillés pour faciliter le débogage

## Comparaison avec la version précédente

| Fonctionnalité | Version précédente (CLIP) | Version améliorée (Transformers) |
|----------------|---------------------------|----------------------------------|
| Installation | Difficile (dépendances C++) | Simple (pip standard) |
| Compatibilité | Limitée | Excellente |
| Variété des modèles | Limitée | Large choix |
| Support GPU | Limité | Complet |
| Compatibilité LangChain | Basique | Complète |
| Traitement par lots | Non | Oui |
| Gestion des erreurs | Basique | Avancée |
| Compatibilité UV | Non | Oui |

## Exemple d'utilisation avancée

```python
from enhanced_embedder import EnhancedEmbedder
from PIL import Image
import numpy as np
import time
from pathlib import Path
import matplotlib.pyplot as plt

# Initialiser l'embedder avec un modèle plus grand et GPU
embedder = EnhancedEmbedder(
    model_name="openai/clip-vit-large-patch14",
    use_gpu=True,
    batch_size=16
)

# Charger un ensemble de textes
texts = [
    "Système RAG multimodal avec FAISS",
    "Architecture d'un système de recherche vectorielle",
    "Comparaison des performances FAISS vs ChromaDB",
    "Avantages des embeddings multimodaux CLIP",
    "Optimisation des recherches par similarité"
]

# Charger des images
image_paths = list(Path("images/").glob("*.jpg"))
images = [Image.open(path) for path in image_paths]

# Mesurer les performances
start_time = time.time()
text_embeddings = embedder.embed_texts(texts)
text_time = time.time() - start_time

start_time = time.time()
image_embeddings = embedder.embed_images(images)
image_time = time.time() - start_time

print(f"Temps pour {len(texts)} textes: {text_time:.2f}s")
print(f"Temps pour {len(images)} images: {image_time:.2f}s")
print(f"Dimension des embeddings: {embedder.get_embedding_dimension()}")

# Calculer la matrice de similarité texte-image
similarity_matrix = np.dot(text_embeddings, image_embeddings.T)

# Afficher les similitudes texte-image
plt.figure(figsize=(12, 8))
plt.imshow(similarity_matrix, cmap='viridis')
plt.colorbar(label="Similarité cosinus")
plt.xticks(range(len(images)), [path.stem for path in image_paths], rotation=45)
plt.yticks(range(len(texts)), texts)
plt.title("Matrice de similarité texte-image")
plt.tight_layout()
plt.savefig("similarity_matrix.png")

# Trouver le texte le plus similaire à chaque image
most_similar_texts = np.argmax(similarity_matrix, axis=0)
for i, img_path in enumerate(image_paths):
    text_idx = most_similar_texts[i]
    score = similarity_matrix[text_idx, i]
    print(f"Image {img_path.name}: Texte le plus similaire: '{texts[text_idx]}' (score: {score:.4f})")
```

## Script d'illustration `test_enhanced_embedder.py`

Un script d'illustration est disponible pour tester rapidement les fonctionnalités de l'embedder amélioré :

```bash
# Avec Python standard
python test_enhanced_embedder.py --model "openai/clip-vit-base-patch32" --use-gpu

# Avec UV (recommandé)
uv run test_enhanced_embedder.py --model "openai/clip-vit-base-patch32" --use-gpu

# Avec le Makefile (utilise UV automatiquement)
make -f Makefile.enhanced test-embedder
```

Options principales :
- `--model TEXT` : Spécifie le modèle CLIP à utiliser
- `--use-gpu` : Active l'utilisation du GPU si disponible
- `--batch-size INT` : Spécifie la taille des lots
- `--texts FILE` : Fichier contenant des textes à encoder (un par ligne)
- `--images DIR` : Répertoire contenant des images à encoder
- `--output FILE` : Fichier de sortie pour sauvegarder les résultats

## Limites et considérations

- Les performances optimales nécessitent un environnement avec GPU
- Les grands modèles CLIP requièrent plus de mémoire mais offrent une meilleure précision
- Le traitement d'un grand nombre d'images peut être limité par la mémoire disponible
- L'alignement texte-image dépend de la qualité du modèle CLIP utilisé
- La normalisation des vecteurs est essentielle pour la recherche par similarité
- L'utilisation du GPU peut nécessiter des adaptations selon l'environnement d'exécution

## Compatibilité avec UV

Le système est optimisé pour fonctionner avec UV, un gestionnaire de paquets Python ultra-rapide :

- Installation simplifiée des dépendances sans compilation complexe
- Environnement isolé pour éviter les conflits
- Performance accrue lors de l'installation des bibliothèques 