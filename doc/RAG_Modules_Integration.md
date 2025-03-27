# Intégration des Modules du Système RAG Multimodal

## Aperçu de l'architecture

L'architecture du système RAG multimodal se compose de plusieurs modules interconnectés, chacun responsable d'une partie spécifique du pipeline:

### `embeddings.py` - Génération des embeddings
- **Classe principale**: `MultimodalEmbedder`
- **Rôle**: Crée des représentations vectorielles de textes et d'images
- **Intégration**: Utilisé par le module `vector_operations.py` pour convertir le contenu en vecteurs avant stockage
- **Fonctionnalités clés**: Supporte différents modèles d'embedding, traitement GPU, et normalisation des vecteurs

### `vector_operations.py` - Gestion du stockage vectoriel
- **Classes principales**: `FAISS` et `TextSplitter`
- **Rôle**: Gère l'indexation, la recherche et le chunking intelligent des documents
- **Intégration**: Utilisé par le module `rag.py` comme couche de stockage et récupération
- **Fonctionnalités clés**: Chunking récursif, indexation rapide avec FAISS, métadonnées enrichies, persistance des index

### `llm.py` - Génération de réponses
- **Classe principale**: `LLaVA`
- **Rôle**: Interface avec les modèles de langage pour générer des réponses
- **Intégration**: Utilisé par `rag.py` pour créer les réponses finales à partir du contexte récupéré
- **Fonctionnalités clés**: Gestion du prompt engineering, traitement multimodal, paramétrage de la génération

### `rag.py` - Orchestration du pipeline
- **Classe principale**: `RAGSystem`
- **Rôle**: Coordonne le flux complet de la requête à la réponse
- **Intégration**: Point d'entrée principal utilisé par les applications et scripts
- **Fonctionnalités clés**: Détection du type de requête, coordination des recherches, formatage du contexte, préparation des métadonnées

## Flux d'information

1. L'utilisateur soumet une requête texte ou image via `RAGSystem.query()`
2. `MultimodalEmbedder` transforme cette requête en vecteur
3. `FAISS` effectue une recherche des documents similaires
4. `TextSplitter` assure que les documents ont été préalablement découpés intelligemment
5. `RAGSystem` prépare le contexte à partir des résultats
6. `LLaVA` génère une réponse basée sur la requête et le contexte

## Améliorations récentes : Chunking Intelligent

### Problématique
Le système initial présentait des limitations dans la façon dont les documents étaient découpés:
- Les textes étaient traités comme des blocs uniques
- Les PDFs étaient découpés par page
- Absence de métadonnées détaillées sur les fragments

### Solution implémentée
Une nouvelle classe `TextSplitter` a été intégrée avec les fonctionnalités suivantes:

1. **Découpage récursif intelligent**:
   - Utilisation de `RecursiveCharacterTextSplitter` de LangChain
   - Respect des structures naturelles du document (paragraphes, phrases, etc.)
   - Paramétrage flexible de la taille des chunks et du chevauchement

2. **Enrichissement des métadonnées**:
   - Position du chunk dans le document (index et total)
   - Taille du chunk
   - Extraction automatique de titres
   - Informations de page pour les PDFs

3. **Intégration dans le pipeline**:
   - La classe `FAISS` utilise maintenant `TextSplitter` pour tous les documents textuels
   - `RAGSystem` expose les paramètres de chunking à l'utilisateur
   - Persistance améliorée avec stockage JSON des métadonnées

### Bénéfices
- Meilleure préservation du contexte dans les chunks
- Recherche sémantique plus précise avec des embeddings de meilleure qualité
- Réduction potentielle des hallucinations du LLM
- Traçabilité accrue grâce aux métadonnées détaillées

## Architecture modularisée

Ce système modulaire permet une grande flexibilité et des améliorations ciblées de chaque composant indépendamment tout en assurant une cohérence globale du pipeline RAG. L'implémentation facilite:

- L'expérimentation avec différents modèles d'embedding
- Le test de diverses stratégies de chunking
- L'optimisation du formatage du contexte pour différents LLMs
- L'extension à de nouveaux types de documents 