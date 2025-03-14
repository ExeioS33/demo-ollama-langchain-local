# Documentation de la classe MultimodalRAG

## Description générale

`MultimodalRAG` est une classe qui implémente un système de Retrieval Augmented Generation (RAG) multimodal. Elle intègre un magasin de vecteurs contenant des embeddings de textes et d'images avec un modèle de langage (LLM) pour générer des réponses contextuelles à des requêtes. Cette classe permet d'interroger une base de connaissances hétérogène (textes, images, PDF) et de recevoir des réponses cohérentes basées sur les informations les plus pertinentes.

## Fonctionnalités clés

- Interface unifiée pour la gestion des documents et des requêtes
- Ajout de différents types de documents (texte, image, PDF)
- Génération de réponses contextuelles basées sur les informations les plus pertinentes
- Support des requêtes textuelles et des requêtes basées sur des images
- Présentation des sources utilisées avec leur score de pertinence

## Dépendances

- MultimodalVectorStore (pour la gestion des embeddings)
- Ollama (pour l'interface avec les modèles de langage)
- LangChain (pour la gestion des prompts et des chaînes)
- Pillow (PIL) pour le traitement d'images
- PyTorch (indirectement via MultimodalEmbedder)

## Initialisation

```python
from multimodal_rag import MultimodalRAG

# Avec les paramètres par défaut
rag = MultimodalRAG()

# Avec des paramètres personnalisés
rag = MultimodalRAG(
    llm_name="llava",
    collection_name="ma_collection",
    temperature=0.3,
    max_tokens=2000,
    persist_directory="./ma_base_donnees"
)
```

### Paramètres d'initialisation

| Paramètre | Type | Défaut | Description |
|-----------|------|--------|-------------|
| `llm_name` | str | "qwen2.5:3b" | Nom du modèle LLM à utiliser via Ollama |
| `collection_name` | str | "multimodal_collection" | Nom de la collection pour le magasin de vecteurs |
| `temperature` | float | 0.2 | Température pour le LLM (créativité vs. précision) |
| `max_tokens` | int | 1000 | Nombre maximum de tokens dans la réponse |
| `persist_directory` | str | "chroma_db" | Répertoire où stocker la base ChromaDB |

## Méthodes principales

### `add_document(document_path: str, document_type: str = "auto", description: Optional[str] = None, metadata: Optional[Dict] = None) -> List[str]`

Ajoute un document au système RAG.

#### Paramètres

| Paramètre | Type | Description |
|-----------|------|-------------|
| `document_path` | str | Chemin vers le document à ajouter |
| `document_type` | str | Type de document ("pdf", "image", "text" ou "auto") |
| `description` | Optional[str] | Description du document (utile pour les images) |
| `metadata` | Optional[Dict] | Métadonnées supplémentaires à associer au document |

#### Retour

| Type | Description |
|------|-------------|
| List[str] | Liste des identifiants générés pour les éléments ajoutés |

#### Exemple d'utilisation

```python
# Ajouter un document texte
ids = rag.add_document("documents/article.txt")

# Ajouter une image avec description
ids = rag.add_document(
    "images/photo.jpg", 
    document_type="image", 
    description="Une photographie de montagnes au coucher du soleil"
)

# Ajouter un PDF avec métadonnées
ids = rag.add_document(
    "documents/rapport.pdf",
    metadata={"auteur": "Alice", "département": "Recherche"}
)
```

### `query(query: Union[str, Image.Image], top_k: int = 5, filter_metadata: Optional[Dict] = None) -> Dict`

Interroge le système RAG avec une requête textuelle ou une image.

#### Paramètres

| Paramètre | Type | Description |
|-----------|------|-------------|
| `query` | Union[str, Image.Image] | Requête textuelle ou image |
| `top_k` | int | Nombre de résultats à récupérer pour le contexte |
| `filter_metadata` | Optional[Dict] | Filtre à appliquer sur les métadonnées |

#### Retour

| Type | Description |
|------|-------------|
| Dict | Dictionnaire contenant la réponse générée et les sources utilisées |

#### Structure du retour

Le dictionnaire retourné contient :
- `answer` : La réponse générée par le LLM
- `sources` : Liste des sources utilisées, chacune avec :
  - `type` : Type de source (image ou texte)
  - `path` ou `content` : Chemin de l'image ou contenu textuel
  - `metadata` : Métadonnées associées
  - `similarity` : Score de similarité entre la requête et la source

#### Exemple d'utilisation

```python
# Requête textuelle simple
result = rag.query("Quels sont les principaux enjeux du changement climatique ?")
print(result["answer"])

# Requête avec image
from PIL import Image
img = Image.open("image_requete.jpg")
result = rag.query(img)
print(result["answer"])

# Requête avec filtrage des métadonnées
result = rag.query(
    "Expliquer le concept d'énergie renouvelable",
    filter_metadata={"département": "Recherche"}
)
print(result["answer"])

# Afficher les sources utilisées
for source in result["sources"]:
    print(f"Source: {source['type']} - Similarité: {source['similarity']:.4f}")
```

## Attributs

| Attribut | Type | Description |
|----------|------|-------------|
| `vector_store` | MultimodalVectorStore | Magasin de vecteurs utilisé pour stocker les embeddings |
| `llm` | Ollama | Instance du modèle de langage |
| `prompt_template` | PromptTemplate | Template pour les requêtes avec contexte |
| `chain` | LLMChain | Chaîne LLM pour la génération de réponses |

## Détails techniques

### Détection automatique du type de document

La méthode `add_document` avec `document_type="auto"` détecte automatiquement le type de document basé sur l'extension du fichier :
- `.pdf` pour les documents PDF
- `.jpg`, `.jpeg`, `.png`, `.gif` pour les images
- `.txt`, `.md`, `.html`, `.htm` pour les documents texte

### Structure du prompt

Le prompt utilisé pour interroger le LLM est structuré comme suit :
```
Tu es un assistant spécialisé dans l'analyse de documents multimodaux (texte et images).
Utilise le contexte fourni pour répondre à la question de l'utilisateur.
Si l'information demandée ne se trouve pas explicitement dans le contexte, indique-le clairement.
Ne fabrique pas de réponse si l'information n'est pas présente dans le contexte.

Contexte:
[Document 1] Contenu du document 1...
[Image 2] Description de l'image 2...
...

Question de l'utilisateur:
<question>

Réponse:
```

### Préparation du contexte

Pour chaque requête, le système :
1. Recherche les documents les plus pertinents dans le magasin de vecteurs
2. Filtre les résultats par score de similarité (minimum 0.2)
3. Formate différemment les sources selon qu'il s'agit de texte ou d'images
4. Combine ces éléments en un contexte structuré pour le LLM

### Traitement des requêtes avec image

Lorsqu'une requête est une image, le système :
1. Génère automatiquement une requête par défaut "Décris cette image en détail"
2. Utilise l'embedding de l'image pour rechercher du contenu similaire
3. Génère une réponse contextualisée basée sur les informations trouvées

## Exemple d'utilisation avancée

```python
from multimodal_rag import MultimodalRAG
from PIL import Image

# Initialiser le système RAG avec un modèle spécifique
rag = MultimodalRAG(
    llm_name="llava",
    temperature=0.3,
    persist_directory="./ma_base_rag"
)

# Ajouter différents types de documents à la base de connaissances
rag.add_document("documents/article_scientifique.pdf")
rag.add_document("documents/rapport_technique.txt")
rag.add_document(
    "images/schema_explicatif.jpg",
    description="Schéma du processus d'électrolyse de l'eau"
)

# Interroger le système avec une question spécifique
result = rag.query(
    "Expliquez le processus d'électrolyse et ses applications industrielles",
    top_k=7
)

# Afficher la réponse
print("=" * 50)
print("RÉPONSE:")
print(result["answer"])
print("=" * 50)

# Afficher les sources utilisées
print("\nSOURCES UTILISÉES:")
for i, source in enumerate(result["sources"]):
    print(f"\nSource {i+1} ({source['type']}):")
    print(f"Similarité: {source['similarity']:.4f}")
    
    if source["type"] == "image":
        print(f"Description: {source['description']}")
        print(f"Chemin: {source['path']}")
    else:
        content = source["content"]
        print(f"Extrait: {content[:100]}..." if len(content) > 100 else f"Contenu: {content}")
    
    if "source" in source["metadata"]:
        print(f"Document source: {source['metadata']['source']}")
```

## Limites et considérations

- La qualité des réponses dépend fortement du modèle LLM utilisé
- Le contexte est limité à 4096 tokens, ce qui peut être insuffisant pour des documents très longs
- L'ajout manuel de descriptions pertinentes pour les images améliore significativement les performances
- Les requêtes complexes peuvent nécessiter un ajustement de `top_k` pour inclure plus de contexte
- Le système fonctionne mieux avec des requêtes spécifiques qu'avec des requêtes très générales
- Un filtrage trop restrictif sur les métadonnées peut limiter les résultats pertinents 