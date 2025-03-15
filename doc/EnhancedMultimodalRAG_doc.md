# Documentation de la classe EnhancedMultimodalRAG

## Description générale

`EnhancedMultimodalRAG` est une classe qui implémente un système RAG (Retrieval Augmented Generation) multimodal optimisé qui combine le magasin vectoriel FAISS amélioré avec un modèle de langage (LLM) pour répondre à des questions en se basant sur une base de connaissances mixte contenant textes et images. Cette implémentation améliorée offre des performances supérieures grâce à l'utilisation de FAISS pour la recherche vectorielle et de transformers pour les embeddings CLIP.

## Fonctionnalités clés

- Interrogation unifiée d'une base de connaissances multimodale (textes, images, PDF)
- Recherche sémantique rapide et précise avec FAISS
- Reranking avancé des résultats pour une meilleure pertinence
- Support GPU pour les calculs vectoriels (si disponible)
- Compatibilité avec divers modèles LLM via Ollama
- Format de réponse structuré avec sources et scores
- Support pour requêtes textuelles, visuelles et combinées texte-image
- Utilisation du modèle CLIP via transformers
- Compatible avec LangChain Expression Language (LCEL)

## Dépendances

- EnhancedVectorStore (avec FAISS)
- LangChain et LangChain-community
- Transformers (Hugging Face)
- Ollama (pour l'intégration des LLM)
- Torch
- Pillow (PIL)
- UV (gestionnaire de paquets Python recommandé)

## Initialisation

```python
from enhanced_multimodal_rag import EnhancedMultimodalRAG

# Avec les paramètres par défaut
rag = EnhancedMultimodalRAG()

# Avec des paramètres personnalisés
rag = EnhancedMultimodalRAG(
    llm_name="qwen2.5:3b", 
    use_gpu=True,
    collection_name="enhanced_multimodal_collection",
    persist_directory="enhanced_vector_store",
    temperature=0.2,
    similarity_threshold=0.25,
    reranking_model="cross-encoder/ms-marco-MiniLM-L-6-v2"
)
```

### Paramètres d'initialisation

| Paramètre | Type | Défaut | Description |
|-----------|------|--------|-------------|
| `llm_name` | str | "llava:latest" | Nom du modèle LLM à utiliser via Ollama |
| `use_gpu` | bool | False | Utiliser le GPU pour FAISS et CLIP si disponible |
| `collection_name` | str | "enhanced_multimodal_collection" | Nom de la collection FAISS |
| `persist_directory` | str | "enhanced_vector_store" | Répertoire où stocker l'index FAISS |
| `temperature` | float | 0.1 | Température pour la génération du LLM |
| `max_tokens` | int | 1000 | Nombre maximum de tokens à générer |
| `similarity_threshold` | float | 0.2 | Seuil de similarité pour filtrer les résultats |
| `clip_model_name` | str | "ViT-B/32" | Nom du modèle CLIP à utiliser |
| `reranking_model` | str | "cross-encoder/ms-marco-MiniLM-L-6-v2" | Modèle de reranking |

## Méthodes principales

### `add_document(document_path: str, document_type: str = "auto", description: Optional[str] = None) -> Dict`

Ajoute un document à la base de connaissances, avec détection automatique du type.

#### Paramètres

| Paramètre | Type | Description |
|-----------|------|-------------|
| `document_path` | str | Chemin vers le document à ajouter |
| `document_type` | str | Type de document ("text", "image", "pdf" ou "auto" pour détection) |
| `description` | Optional[str] | Description (pour les images) |

#### Retour

| Type | Description |
|------|-------------|
| Dict | Dictionnaire contenant les détails de l'ajout (success, ids, type) |

#### Exemple d'utilisation

```python
# Ajouter un document texte
result = rag.add_document("documents/article.txt")

# Ajouter une image avec description
result = rag.add_document("images/diagramme.jpg", document_type="image", 
                          description="Diagramme d'architecture système")

# Ajouter un PDF avec détection automatique
result = rag.add_document("documents/rapport.pdf")
```

### `query(query: Union[str, Image.Image], top_k: int = 5, filter_metadata: Optional[Dict] = None, use_reranking: bool = True) -> Dict`

Interroge la base de connaissances avec une requête textuelle ou une image et génère une réponse.

#### Paramètres

| Paramètre | Type | Description |
|-----------|------|-------------|
| `query` | Union[str, Image.Image] | Requête textuelle ou image |
| `top_k` | int | Nombre maximum de résultats à utiliser comme contexte |
| `filter_metadata` | Optional[Dict] | Filtre à appliquer sur les métadonnées |
| `use_reranking` | bool | Si True, utilise le reranking pour améliorer la pertinence |

#### Retour

| Type | Description |
|------|-------------|
| Dict | Dictionnaire contenant la réponse, la requête et les sources utilisées |

#### Structure de la réponse

La structure de la réponse est un dictionnaire contenant :
- `query` : La requête originale
- `answer` : La réponse générée par le LLM
- `sources` : Liste des sources utilisées avec leurs métadonnées et scores
- `time_taken` : Temps total d'exécution en secondes

#### Exemple d'utilisation

```python
# Requête textuelle simple
response = rag.query("Comment fonctionne le système RAG multimodal?")
print(response["answer"])

# Requête avec filtre sur les métadonnées
response = rag.query("Quelle est l'architecture du système?", 
                     filter_metadata={"type": "image"})

# Requête avec une image
from PIL import Image
image = Image.open("query_image.jpg")
response = rag.query(image, top_k=3)
```

### `query_text_and_image(text: str, image: Image.Image, top_k: int = 5, filter_metadata: Optional[Dict] = None, use_reranking: bool = True) -> Dict`

Interroge la base de connaissances avec une combinaison de texte et d'image, et génère une réponse.

#### Paramètres

| Paramètre | Type | Description |
|-----------|------|-------------|
| `text` | str | Requête textuelle |
| `image` | Image.Image | Image pour la requête |
| `top_k` | int | Nombre maximum de résultats à utiliser comme contexte |
| `filter_metadata` | Optional[Dict] | Filtre à appliquer sur les métadonnées |
| `use_reranking` | bool | Si True, utilise le reranking pour améliorer la pertinence |

#### Retour

| Type | Description |
|------|-------------|
| Dict | Dictionnaire contenant la réponse, la requête et les sources utilisées |

#### Structure de la réponse

La structure de la réponse est identique à celle de la méthode `query` :
- `query` : La requête originale (texte + référence à l'image)
- `answer` : La réponse générée par le LLM
- `sources` : Liste des sources utilisées avec leurs métadonnées et scores
- `time_taken` : Temps total d'exécution en secondes

#### Exemple d'utilisation

```python
# Requête combinée texte-image
from PIL import Image
image = Image.open("query_image.jpg")
response = rag.query_text_and_image(
    text="Décris cette image et explique comment elle est liée à mon contenu",
    image=image,
    top_k=5
)
print(response["answer"])

# Requête avec filtre sur les métadonnées
response = rag.query_text_and_image(
    text="Quelle est la relation entre cette image et les documents techniques?",
    image=image,
    filter_metadata={"type": "text"}
)
```

### `migrate_from_chromadb(chroma_path: str, chroma_collection: str) -> Dict`

Migre une collection ChromaDB existante vers le système amélioré avec FAISS.

#### Paramètres

| Paramètre | Type | Description |
|-----------|------|-------------|
| `chroma_path` | str | Chemin vers le répertoire ChromaDB |
| `chroma_collection` | str | Nom de la collection ChromaDB à migrer |

#### Retour

| Type | Description |
|------|-------------|
| Dict | Dictionnaire contenant les détails de la migration |

#### Exemple d'utilisation

```python
# Migrer une collection ChromaDB existante
result = rag.migrate_from_chromadb("chroma_db", "multimodal_collection")
print(f"Migration réussie: {result['success']}")
print(f"Nombre d'éléments migrés: {result['count']}")
```

### `reset() -> Dict`

Réinitialise la base de connaissances.

#### Retour

| Type | Description |
|------|-------------|
| Dict | Dictionnaire indiquant le succès de l'opération |

#### Exemple d'utilisation

```python
# Réinitialiser la base de connaissances
result = rag.reset()
print(f"Réinitialisation réussie: {result['success']}")
```

## Attributs

| Attribut | Type | Description |
|----------|------|-------------|
| `vector_store` | EnhancedVectorStore | Instance du magasin vectoriel FAISS |
| `llm` | BaseChatModel | Instance du modèle de langage |
| `prompt_template` | PromptTemplate | Template de prompt pour le LLM |
| `similarity_threshold` | float | Seuil minimal de similarité |
| `chain` | RunnableSequence | Chaîne LangChain pour la génération |

## Détails techniques

### Intégration avec Ollama

La classe utilise Ollama pour accéder à différents modèles de langage locaux :
- `llava:latest` : Modèle multimodal par défaut
- `qwen2.5:3b` : Alternative compacte recommandée
- Autres modèles compatibles : mistral, llama2, etc.

### Format de prompt optimisé

Le prompt utilisé pour le LLM est spécifiquement conçu pour :
1. Fournir un contexte structuré à partir des résultats de recherche
2. Encourager le modèle à citer ses sources explicitement
3. Limiter les hallucinations en soulignant l'importance de se baser sur le contexte fourni

### LCEL (LangChain Expression Language)

Le système utilise la syntaxe LCEL moderne de LangChain pour créer des chaînes de traitement plus flexibles et efficaces :

```python
chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt_template
    | llm
    | output_parser
)
```

Cette approche remplace l'ancienne syntaxe basée sur `LLMChain` et offre plusieurs avantages :
- Streaming de premier ordre
- Support asynchrone
- Exécution parallèle optimisée
- Gestion des retries et fallbacks
- Accès aux résultats intermédiaires

### Reranking avancé

Le système utilise un modèle CrossEncoder pour reclasser les résultats initiaux de la recherche vectorielle :
1. Une première recherche large est effectuée avec FAISS 
2. Le modèle CrossEncoder évalue la pertinence de chaque résultat par rapport à la requête
3. Les résultats sont réordonnés selon leur score de pertinence
4. Seuls les résultats les plus pertinents sont utilisés comme contexte pour le LLM

## Comparaison avec la version précédente

| Fonctionnalité | Version précédente (ChromaDB) | Version améliorée (FAISS) |
|----------------|-------------------------------|---------------------------|
| Vitesse de recherche | Modérée | Rapide (2-5x plus rapide) |
| Support des grandes collections | Limitée (<100k éléments) | Excellente (millions d'éléments) |
| Reranking | Non | Oui (amélioration significative de la pertinence) |
| Support GPU | Non | Oui (accélération des recherches) |
| Structure de réponse | Basique | Enrichie (scores de similarité et reranking) |
| Seuil de similarité | Fixe | Configurable |
| Modèle d'embedding | CLIP (package original) | CLIP via transformers (plus robuste) |
| Syntaxe LangChain | Ancienne (LLMChain) | Moderne (LCEL) |

## Exemple d'utilisation avancée

```python
from enhanced_multimodal_rag import EnhancedMultimodalRAG
from PIL import Image
import json
import time

# Configuration avancée
rag = EnhancedMultimodalRAG(
    llm_name="qwen2.5:3b",
    use_gpu=True,
    collection_name="knowledge_base",
    similarity_threshold=0.3,
    temperature=0.1,
    reranking_model="cross-encoder/ms-marco-MiniLM-L-12-v2"
)

# Ajouter différents types de documents
rag.add_document("documentation/api_spec.txt")
rag.add_document("images/system_architecture.png", 
                 description="Architecture du système RAG amélioré")
rag.add_document("documentation/user_manual.pdf")

# Requête textuelle avec contexte
response = rag.query("Comment configurer le système pour utiliser le GPU?")

# Affichage formaté de la réponse
print("=" * 40)
print("QUESTION:", response["query"])
print("=" * 40)
print("RÉPONSE:")
print(response["answer"])
print("=" * 40)
print("SOURCES UTILISÉES:")

for i, source in enumerate(response["sources"]):
    print(f"{i+1}. {source['content'][:100]}...")
    print(f"   Score de similarité: {source['similarity']:.4f}")
    if 'rerank_score' in source:
        print(f"   Score de reranking: {source['rerank_score']:.4f}")
    print()

# Requête avec image
query_image = Image.open("query_images/component_diagram.jpg")
response = rag.query(query_image)

# Sauvegarder les résultats
with open("results/query_results.json", "w") as f:
    # Convertir l'image en chemin pour la sérialisation JSON
    if isinstance(response["query"], Image.Image):
        response["query"] = "Image query"
    json.dump(response, f, indent=2)

print(f"Temps total d'exécution: {response['time_taken']:.2f} secondes")
```

## Script de démo `enhanced_multimodal_rag_demo.py`

Le script de démonstration fournit une interface en ligne de commande pour utiliser toutes les fonctionnalités du système RAG multimodal amélioré :

```bash
# Avec Python standard
python enhanced_multimodal_rag_demo.py --add-document path/to/document.pdf

# Avec UV (recommandé)
uv run enhanced_multimodal_rag_demo.py --add-document path/to/document.pdf

# Avec le Makefile (utilise UV automatiquement)
make -f Makefile.enhanced add-document
```

Options principales :
- `--add-document PATH` : Ajoute un document (avec détection automatique du type)
- `--description TEXT` : Spécifie une description pour les images
- `--query TEXT` : Effectue une requête textuelle
- `--image-query PATH` : Effectue une requête avec une image
- `--model NAME` : Spécifie le modèle LLM (défaut: llava:latest)
- `--use-gpu` : Utilise le GPU si disponible
- `--similarity-threshold FLOAT` : Spécifie le seuil de similarité
- `--no-reranking` : Désactive le reranking
- `--reset` : Réinitialise la base de connaissances
- `--migrate` : Migre depuis ChromaDB

## Limites et considérations

- Les performances optimales nécessitent un environnement avec GPU
- La qualité des réponses dépend du modèle LLM utilisé
- L'intégration avec Ollama nécessite une installation locale de ce service
- Les grands documents PDF peuvent nécessiter un temps de traitement significatif
- Le reranking ajoute un temps de traitement supplémentaire, mais améliore nettement la pertinence
- L'utilisation du GPU peut nécessiter des adaptations selon l'environnement d'exécution

## Compatibilité avec UV

Le système est optimisé pour fonctionner avec UV, un gestionnaire de paquets Python ultra-rapide :

- Utilisation de `uv run` pour exécuter les scripts Python
- Utilisation de `uv pip` pour installer les dépendances
- Commandes de vérification adaptées à l'environnement UV
- Meilleure gestion des dépendances qui nécessitent une compilation 