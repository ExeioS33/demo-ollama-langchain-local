#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script de test pour la fonctionnalité de requête combinée texte-image
---------------------------------------------------------------------
Ce script démontre comment utiliser la fonctionnalité de requête combinée
texte-image avec le système RAG multimodal amélioré.
"""

import os
import argparse
from pathlib import Path
from PIL import Image
from enhanced_multimodal_rag import EnhancedMultimodalRAG


def parse_arguments():
    """Parse les arguments de la ligne de commande."""
    parser = argparse.ArgumentParser(
        description="Test de la fonctionnalité de requête combinée texte-image"
    )

    parser.add_argument(
        "--model",
        default="qwen2.5:3b",
        help="Modèle LLM à utiliser via Ollama (par défaut: qwen2.5:3b)",
    )

    parser.add_argument(
        "--collection",
        default="enhanced_multimodal_collection",
        help="Nom de la collection (par défaut: enhanced_multimodal_collection)",
    )

    parser.add_argument(
        "--db-path",
        default="enhanced_vector_store",
        help="Chemin du répertoire pour stocker la base FAISS (par défaut: enhanced_vector_store)",
    )

    parser.add_argument(
        "--text",
        default="Décris cette image et explique comment elle est liée à mon contenu",
        help="Texte de la requête (par défaut: 'Décris cette image et explique comment elle est liée à mon contenu')",
    )

    parser.add_argument(
        "--image",
        required=True,
        help="Chemin vers l'image à utiliser pour la requête (obligatoire)",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Nombre de résultats à récupérer (par défaut: 5)",
    )

    parser.add_argument(
        "--no-reranking",
        action="store_true",
        help="Désactiver le reranking pour cette requête",
    )

    return parser.parse_args()


def print_divider(title=None):
    """Affiche un séparateur avec un titre optionnel."""
    width = 80
    if title:
        print(f"\n{'=' * 10} {title} {'=' * (width - 12 - len(title))}\n")
    else:
        print(f"\n{'=' * width}\n")


def print_results(results):
    """Affiche les résultats d'une requête de manière formatée."""
    answer = results["answer"]
    sources = results["sources"]

    print_divider("Réponse")
    print(answer)

    print_divider("Sources utilisées")
    for i, source in enumerate(sources):
        print(f"Source {i + 1}:")
        if source["type"] == "image":
            print(f"  Type: Image")
            print(f"  Chemin: {source['path']}")
            print(f"  Description: {source['description']}")
        else:
            print(f"  Type: Texte")
            content = source["content"]
            if len(content) > 200:
                content = content[:200] + "..."
            print(f"  Contenu: {content}")

        print(f"  Score similarité: {source['similarity']:.4f}")
        if source.get("rerank_score") is not None:
            print(f"  Score reranking: {source['rerank_score']:.4f}")

        if "metadata" in source:
            if "page" in source["metadata"]:
                print(f"  Page: {source['metadata']['page']}")
            if "filename" in source["metadata"]:
                print(f"  Fichier: {source['metadata']['filename']}")
        print()


def main():
    """Fonction principale du script de test."""
    args = parse_arguments()

    # Vérifier que l'image existe
    if not os.path.exists(args.image):
        print(f"Erreur: L'image {args.image} n'existe pas")
        return

    # Initialiser le système RAG
    print(f"Initialisation du système RAG avec la collection '{args.collection}'...")
    rag_system = EnhancedMultimodalRAG(
        llm_name=args.model,
        collection_name=args.collection,
        persist_directory=args.db_path,
    )

    # Charger l'image
    print(f"Chargement de l'image: {args.image}")
    image = Image.open(args.image).convert("RGB")

    # Effectuer la requête combinée
    print_divider(f"Requête combinée texte-image: '{args.text}' + {args.image}")
    results = rag_system.query_text_and_image(
        text=args.text,
        image=image,
        top_k=args.top_k,
        use_reranking=not args.no_reranking,
    )

    # Afficher les résultats
    print_results(results)


if __name__ == "__main__":
    main()
