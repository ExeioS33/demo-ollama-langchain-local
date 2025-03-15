# Résumé Technico-Fonctionnel du Système RAG Multimodal avec CLIP

## 1. Architecture Globale

Le système implémente une architecture de Retrieval Augmented Generation (RAG) multimodale qui permet d'unifier les requêtes textuelles et visuelles dans un même espace vectoriel. Il se compose de trois composants principaux :

1. **MultimodalEmbedder** : Génère des embeddings vectoriels à partir de textes et d'images
2. **MultimodalVectorStore** : Stocke et interroge les embeddings dans une base de données vectorielle
3. **MultimodalRAG** : Intègre la recherche vectorielle avec un modèle de langage pour répondre aux requêtes

```
┌───────────────────┐     ┌───────────────────┐     ┌───────────────────┐
│ MultimodalEmbedder│────>│MultimodalVectorStore│───>│  MultimodalRAG    │
│   (CLIP Model)    │     │    (ChromaDB)     │     │  (Ollama LLM)     │
└───────────────────┘     └───────────────────┘     └───────────────────┘
```

## 2. Composants Principaux

### 2.1 MultimodalEmbedder

Cette classe est responsable de la génération d'embeddings pour les textes et les images en utilisant le modèle CLIP (Contrastive Language-Image Pre-training).

**Méthodes principales :**
- `__init__(model_name="openai/clip-vit-base-patch32")` : Initialise l'embedder avec un modèle CLIP spécifique
- `embed_text(texts: List[str]) -> np.ndarray` : Génère des embeddings pour une liste de textes
- `embed_image(images: List[Union[str, Image.Image]]) -> np.ndarray` : Génère des embeddings pour une liste d'images
- `embed_text_and_image(text: str, image: Image.Image) -> np.ndarray` : Génère un embedding combiné pour une paire texte-image

**Caractéristiques techniques :**
- Utilise le modèle CLIP de OpenAI via Hugging Face
- Fonctionne sur CPU ou GPU (détection automatique)
- Dimensionnalité des embeddings : 512 dimensions
- Normalisation L2 des embeddings pour la similarité cosinus
- Stratégie de fusion pour les embeddings combinés texte-image

### 2.2 MultimodalVectorStore

Cette classe gère le stockage et l'interrogation des embeddings multimodaux dans une base de données vectorielle.

**Méthodes principales :**
- `__init__(collection_name="multimodal_collection", persist_directory="chroma_db")` : Initialise le magasin de vecteurs
- `add_texts(texts: List[str], metadatas: Optional[List[Dict]]) -> List[str]` : Ajoute des textes à la base
- `add_images(images: List[Union[str, Image.Image]], descriptions: Optional[List[str]], metadatas: Optional[List[Dict]]) -> List[str]` : Ajoute des images
- `add_pdf(pdf_path: str, extract_images: bool=True) -> List[str]` : Extrait et ajoute le contenu d'un PDF (texte et images)
- `query(query: Union[str, Image.Image], top_k: int=5, filter_metadata: Optional[Dict]=None) -> List[Dict]` : Interroge la base avec du texte ou une image
- `query_text_and_image(text: str, image: Image.Image, top_k: int=5, filter_metadata: Optional[Dict]=None) -> List[Dict]` : Interroge la base avec une combinaison de texte et d'image

**Caractéristiques techniques :**
- Utilise ChromaDB comme base de données vectorielle
- Implémente une fonction d'embedding personnalisée pour ChromaDB
- Stockage persistant des données entre les sessions
- Métadonnées enrichies pour chaque document
- Filtrage des résultats par seuil de similarité (0.2 par défaut)
- Support pour les requêtes multimodales combinant texte et image

### 2.3 MultimodalRAG

Cette classe intègre le magasin de vecteurs avec un modèle de langage (LLM) pour générer des réponses contextuelles.

**Méthodes principales :**
- `__init__(llm_name="qwen2.5:3b", collection_name="multimodal_collection", temperature=0.2, max_tokens=1000, persist_directory="chroma_db")` : Initialise le système RAG
- `add_document(document_path: str, document_type: str="auto", description: Optional[str]=None) -> List[str]` : Ajoute un document (texte, image ou PDF)
- `query(query: Union[str, Image.Image], top_k: int=5, filter_metadata: Optional[Dict]=None) -> Dict` : Interroge le système avec contexte et génère une réponse
- `query_text_and_image(text: str, image: Image.Image, top_k: int=5, filter_metadata: Optional[Dict]=None) -> Dict` : Interroge le système avec une combinaison de texte et d'image et génère une réponse

**Caractéristiques techniques :**
- Utilise Ollama comme interface pour les modèles de langage locaux
- Modèles compatibles : qwen2.5:3b (défaut), llava, llama2, etc.
- Contexte maximal de 4096 tokens
- Prompt template optimisé pour éviter les hallucinations
- Format de réponse structuré avec sources et scores de similarité
- Support pour les requêtes multimodales combinant texte et image

## 3. Technologies Utilisées

### 3.1 Modèles d'Intelligence Artificielle

- **CLIP** (openai/clip-vit-base-patch32)
  - Modèle multimodal qui aligne les représentations textuelles et visuelles
  - Architecture : Vision Transformer (ViT) avec taille de patch 32
  - Pré-entraîné sur 400 millions de paires texte-image

- **Ollama** comme interface LLM
  - Modèles supportés : qwen2.5:3b (par défaut), llava, llama2, etc.
  - Exécution locale avec contrôle de température et de longueur

### 3.2 Base de Données et Stockage

- **ChromaDB**
  - Base de données vectorielle optimisée pour la recherche de similarité
  - Algorithme HNSW (Hierarchical Navigable Small World) avec espace cosinus
  - Méthode de persistance : stockage sur disque dans le répertoire spécifié

### 3.3 Bibliothèques de Support

- **PyTorch** pour l'inférence du modèle CLIP
- **Transformers** (Hugging Face) pour le chargement des modèles
- **PyMuPDF** (fitz) pour l'extraction de texte et d'images des PDF
- **Pillow** (PIL) pour le traitement d'images
- **LangChain** pour la création de prompts et la gestion de chaînes LLM

## 4. Workflow Fonctionnel

### 4.1 Indexation de Documents

1. **Analyse du type de document** : détection automatique (PDF, image, texte) ou spécifié par l'utilisateur
2. **Traitement spécifique** :
   - Texte : lecture et vectorisation directe
   - Image : chargement, prétraitement et génération d'embeddings visuels
   - PDF : extraction de texte page par page, extraction optionnelle des images
3. **Génération d'embeddings** via CLIP
4. **Stockage dans ChromaDB** avec métadonnées enrichies
5. **Attribution d'identifiants uniques** (UUID) pour chaque élément

### 4.2 Traitement des Requêtes

1. **Analyse de la requête** : détection si texte, image ou combinaison texte-image
2. **Génération d'embedding** pour la requête :
   - Texte : utilisation du processeur de texte CLIP
   - Image : utilisation du processeur d'image CLIP
   - Combinaison texte-image : génération d'un embedding fusionné
3. **Recherche de similarité** dans la base vectorielle
4. **Filtrage des résultats** par score de similarité (seuil à 0.2)
5. **Formatage du contexte** pour le LLM
6. **Génération de réponse** par le LLM avec le contexte récupéré
7. **Structuration de la réponse** avec la réponse et les sources utilisées

## 5. Interface Utilisateur

### 5.1 Script de Démonstration

Le script `enhanced_multimodal_rag_demo.py` fournit une interface en ligne de commande avec les options :

- **--add-document** : Ajoute un document au système
- **--query** : Effectue une requête textuelle
- **--image-query** : Utilise une image pour la requête
- **--combined-query** : Effectue une requête combinée texte-image (nécessite --query et --image-query)
- **--model** : Spécifie le modèle LLM à utiliser (défaut: qwen2.5:3b)
- **--db-path** : Chemin vers la base de données (défaut: enhanced_vector_store)
- **--collection** : Nom de la collection (défaut: enhanced_multimodal_collection)
- **--reset** : Réinitialise la base de données
- **--use-gpu** : Utilise le GPU pour FAISS si disponible
- **--no-reranking** : Désactive le reranking pour cette requête

### 5.2 Script de Test pour Requêtes Combinées

Le script `test_combined_query.py` fournit une interface simplifiée pour tester spécifiquement les requêtes combinées texte-image :

- **--model** : Spécifie le modèle LLM à utiliser
- **--collection** : Nom de la collection
- **--db-path** : Chemin vers la base de données
- **--text** : Texte de la requête
- **--image** : Chemin vers l'image à utiliser pour la requête (obligatoire)
- **--top-k** : Nombre de résultats à récupérer
- **--no-reranking** : Désactive le reranking pour cette requête

### 5.3 Makefile

Un Makefile fournit des commandes simplifiées :

- **make install** : Installe les dépendances
- **make notebook** : Convertit le JSON en notebook Jupyter
- **make start-ollama** : Démarre Ollama en arrière-plan
- **make download-models** : Télécharge les modèles nécessaires
- **make reset-db** : Réinitialise la base de données
- **make add-text/add-image/add-pdf** : Ajoute différents types de documents
- **make query/image-query** : Effectue des requêtes

### 5.4 Notebook Interactif

Un notebook Jupyter (`multimodal_rag_demo.ipynb`) fournit une démonstration interactive qui montre :

- La visualisation des embeddings d'images et de textes
- La création d'une base de connaissances multimodale
- L'exécution de différents types de requêtes
- La visualisation des résultats et des sources

## 6. Performances et Limitations

### 6.1 Performances

- **Embeddings** : Dimensionnalité de 512, permettant un bon équilibre entre précision et efficacité
- **Recherche** : Algorithme HNSW optimisé pour la recherche approximative de plus proches voisins
- **Seuil de similarité** : Valeur de 0.2 pour filtrer les résultats peu pertinents
- **Stockage** : Persistance sur disque pour conserver les données entre les sessions

### 6.2 Limitations

- **Dépendance au hardware** : Performances optimales sur GPU, mais fonctionne sur CPU
- **Taille des modèles** : Requiert environ 10 GB d'espace disque pour les modèles
- **Traitement de PDF volumineux** : Peut être lent pour les grands documents
- **Qualité des réponses** : Dépend du modèle LLM utilisé

## 7. Conclusion

Le système RAG multimodal avec CLIP offre une solution unifiée pour interroger des bases de connaissances contenant à la fois du texte et des images. Son architecture modulaire permet une flexibilité d'utilisation et d'extension, tandis que sa capacité à fonctionner localement sans envoi de données à des services externes garantit la confidentialité des informations traitées.

L'approche multimodale avec embeddings unifiés représente une avancée significative par rapport aux systèmes RAG traditionnels qui traitent séparément les modalités textuelles et visuelles. 