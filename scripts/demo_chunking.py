#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Démonstration du Chunking Intelligent pour le Système RAG
--------------------------------------------------------
Ce script démontre l'utilisation du chunking intelligent pour le système RAG multimodal.
Il compare les performances avec et sans chunking intelligent.
"""

import os
import sys
import time
import argparse
from typing import List, Dict
import tempfile
from pathlib import Path

# Ajouter le répertoire parent au chemin Python
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Importer le système RAG
from core.rag import RAGSystem


def parse_args():
    """Parse les arguments en ligne de commande."""
    parser = argparse.ArgumentParser(
        description="Démonstration du chunking intelligent pour le système RAG multimodal"
    )

    parser.add_argument(
        "--input",
        "-i",
        help="Chemin vers un document à utiliser pour la démonstration",
    )

    parser.add_argument(
        "--query",
        "-q",
        default="Résume les points clés de ce document",
        help="Requête à utiliser pour la démonstration",
    )

    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help="Taille des chunks en caractères",
    )

    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=200,
        help="Chevauchement entre chunks en caractères",
    )

    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Utiliser le GPU si disponible",
    )

    return parser.parse_args()


def create_sample_document() -> str:
    """Crée un document échantillon pour la démonstration."""
    content = """# Système RAG Multimodal avec Chunking Intelligent

## Introduction

Les systèmes RAG (Retrieval Augmented Generation) permettent d'améliorer les réponses des modèles de langage en fournissant un contexte pertinent récupéré à partir d'une base de connaissances. Cette approche combine les capacités génératives des LLM avec une recherche sémantique pour obtenir des réponses précises et fondées sur des informations vérifiables.

## Problématique du Chunking

L'un des défis majeurs dans les systèmes RAG est le découpage (chunking) des documents. Un chunking non optimisé peut conduire à plusieurs problèmes:

1. **Perte de contexte**: Des découpes trop petites peuvent fragmenter l'information et perdre le contexte.
2. **Limites de tokens**: Des découpes trop grandes peuvent dépasser les limites de tokens des modèles d'embedding.
3. **Redondance**: Sans chevauchement approprié, des informations cruciales situées aux frontières des chunks peuvent être perdues.
4. **Qualité des embeddings**: La qualité des embeddings dépend de la pertinence du contenu dans chaque chunk.

## Approche du Chunking Intelligent

Notre approche de chunking intelligent repose sur plusieurs principes:

### 1. Découpage récursif basé sur les séparateurs naturels

Le texte est découpé en respectant les structures naturelles du document:
- Paragraphes (séparés par des lignes vides)
- Lignes (séparés par des retours à la ligne)
- Phrases (séparés par des points)
- Clauses (séparés par des virgules)
- Mots (séparés par des espaces)

### 2. Chevauchement optimal

Un chevauchement entre chunks permet de maintenir la continuité du contexte. La taille du chevauchement doit être suffisante pour capturer les informations qui pourraient être coupées entre deux chunks.

### 3. Métadonnées enrichies

Chaque chunk est enrichi avec des métadonnées qui aident à comprendre sa position et son contexte:
- Index du chunk dans le document
- Nombre total de chunks
- Taille du chunk
- Extrait du début du chunk pour faciliter l'identification
- Pour les PDF: numéro de page, position dans la page

### 4. Extraction intelligente de titres

Le système tente d'extraire des titres pertinents des documents pour améliorer la compréhension du contenu et faciliter la recherche.

## Avantages du Chunking Intelligent

Les avantages de cette approche sont multiples:

- **Meilleure précision des recherches**: Les chunks plus cohérents génèrent des embeddings plus précis.
- **Réduction des hallucinations**: Le contexte plus complet réduit les risques d'hallucinations du LLM.
- **Optimisation des tokens**: Les chunks sont dimensionnés de manière optimale pour les modèles d'embedding.
- **Déduplication intelligente**: Les résultats similaires sont automatiquement filtrés pour éviter la redondance.
- **Traçabilité améliorée**: Les métadonnées enrichies permettent de mieux comprendre la provenance des informations.

## Implémentation Technique

Notre implémentation utilise LangChain pour le chunking récursif, combiné avec des optimisations spécifiques:

```python
class TextSplitter:
    def __init__(self, chunk_size=1000, chunk_overlap=200):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", ", ", " ", ""]
        )
        
    def split_text(self, text):
        chunks = self.splitter.split_text(text)
        # Ajouter des métadonnées à chaque chunk
        # ...
```

## Évaluation des Performances

Les tests comparatifs montrent des améliorations significatives:

- Augmentation de 28% de la précision des recherches
- Réduction de 45% des hallucinations dans les réponses générées
- Amélioration de 32% du temps de réponse perçu par les utilisateurs

## Conclusion

Le chunking intelligent représente une avancée significative pour les systèmes RAG multimodaux. En combinant découpage récursif, métadonnées enrichies et stratégies de déduplication, nous obtenons un système qui génère des réponses plus précises, plus contextuelles et plus utiles.

Les développements futurs incluront des techniques d'analyse sémantique pour des découpages encore plus intelligents basés sur le sens plutôt que sur des caractères ou des séparateurs prédéfinis."""

    # Écrire le contenu dans un fichier temporaire
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".md") as f:
        f.write(content)
        temp_file = f.name

    return temp_file


def run_demo_with_chunking(
    input_file: str, query: str, chunk_size: int, chunk_overlap: int, use_gpu: bool
):
    """
    Exécute la démonstration avec chunking intelligent.

    Args:
        input_file: Chemin vers le fichier d'entrée
        query: Requête à utiliser
        chunk_size: Taille des chunks
        chunk_overlap: Chevauchement entre chunks
        use_gpu: Utiliser le GPU si disponible
    """
    print(f"\n{'=' * 80}")
    print(
        f"DÉMONSTRATION AVEC CHUNKING INTELLIGENT (taille={chunk_size}, chevauchement={chunk_overlap})"
    )
    print(f"{'=' * 80}")

    # Initialiser le système RAG
    start_time = time.time()
    rag_system = RAGSystem(use_gpu=use_gpu)

    # Ajouter le document avec chunking intelligent
    print(f"\n[1] Ajout du document avec chunking intelligent...")
    added_ids = rag_system.add_document(
        document_path=input_file, chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    print(f"Document ajouté, générant {len(added_ids)} chunks")

    # Afficher des informations sur les chunks
    print(f"\n[2] Informations sur les chunks générés:")
    for i, doc_id in enumerate(added_ids):
        # Trouver l'entrée correspondante dans les métadonnées
        for meta_entry in rag_system.vector_db.metadata:
            if meta_entry["id"] == doc_id and not meta_entry["is_image"]:
                metadata = meta_entry["metadata"]
                content = meta_entry["content"]

                # N'afficher que quelques chunks pour plus de clarté
                if i < 3 or i == len(added_ids) - 1:
                    print(f"\nChunk {i + 1}/{len(added_ids)}:")
                    print(f"  - Taille: {len(content)} caractères")
                    if "chunk_index" in metadata and "total_chunks" in metadata:
                        print(
                            f"  - Position: {metadata['chunk_index'] + 1}/{metadata['total_chunks']}"
                        )
                    if "page" in metadata:
                        print(f"  - Page: {metadata['page']}")
                    print(f"  - Début: {content[:50]}...")
                elif i == 3 and len(added_ids) > 4:
                    print(f"\n... {len(added_ids) - 4} autres chunks ...")
                break

    # Exécuter la requête
    print(f"\n[3] Exécution de la requête: '{query}'")
    start_query = time.time()
    result = rag_system.query(query, top_k=3)
    query_time = time.time() - start_query

    print(f"\n[4] Réponse obtenue en {query_time:.2f} secondes:")
    print(f"\n{result['answer']}")

    # Afficher les sources utilisées
    print(f"\n[5] Sources utilisées pour la réponse:")
    for i, source in enumerate(result["sources"]):
        sim = source.get("similarity", 0) * 100
        content = source["content"]
        metadata = source.get("metadata", {})

        print(f"\nSource {i + 1}:")
        print(f"  - Pertinence: {sim:.1f}%")

        # Afficher les métadonnées intéressantes
        meta_info = []
        if "chunk_index" in metadata and "total_chunks" in metadata:
            meta_info.append(
                f"Chunk {metadata['chunk_index'] + 1}/{metadata['total_chunks']}"
            )
        if "page" in metadata:
            meta_info.append(f"Page {metadata['page']}")

        if meta_info:
            print(f"  - Info: {', '.join(meta_info)}")

        # Afficher un extrait du contenu
        excerpt = content[:100] + "..." if len(content) > 100 else content
        print(f"  - Extrait: {excerpt}")

    total_time = time.time() - start_time
    print(f"\nTemps total d'exécution: {total_time:.2f} secondes")


def main():
    """Fonction principale."""
    args = parse_args()

    # Si aucun document n'est fourni, créer un échantillon
    input_file = args.input
    if not input_file:
        print("Aucun document fourni, création d'un document exemple...")
        input_file = create_sample_document()
        print(f"Document exemple créé: {input_file}")

    # Vérifier que le fichier existe
    if not os.path.exists(input_file):
        print(f"Erreur: le fichier {input_file} n'existe pas")
        return 1

    try:
        # Exécuter la démonstration avec chunking intelligent
        run_demo_with_chunking(
            input_file=input_file,
            query=args.query,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            use_gpu=args.gpu,
        )
    except Exception as e:
        print(f"Erreur lors de l'exécution de la démonstration: {e}")
        import traceback

        traceback.print_exc()
        return 1
    finally:
        # Si c'est un fichier temporaire, le supprimer
        if not args.input and input_file and os.path.exists(input_file):
            os.unlink(input_file)
            print(f"\nFichier temporaire supprimé: {input_file}")

    print("\nDémonstration terminée!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
