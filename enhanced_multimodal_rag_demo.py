#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script de démonstration pour le système RAG multimodal amélioré
---------------------------------------------------------------
Ce script montre comment utiliser le système RAG multimodal amélioré
avec FAISS et les techniques de reranking avancées.
"""

import os
import argparse
from pathlib import Path
from PIL import Image
from enhanced_multimodal_rag import EnhancedMultimodalRAG, migrate_from_original_rag


def parser_arguments():
    """Parse les arguments de la ligne de commande."""
    parser = argparse.ArgumentParser(
        description="Démo du système RAG multimodal amélioré avec FAISS"
    )

    parser.add_argument(
        "--model",
        default="qwen2.5:3b",
        help="Modèle LLM à utiliser via Ollama (par défaut: qwen2.5:3b)",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Température pour le LLM (par défaut: 0.2)",
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
        "--migrate",
        action="store_true",
        help="Migrer une collection ChromaDB existante vers FAISS",
    )

    parser.add_argument(
        "--chroma-collection",
        default="multimodal_collection",
        help="Nom de la collection ChromaDB source pour la migration (par défaut: multimodal_collection)",
    )

    parser.add_argument(
        "--chroma-path",
        default="chroma_db",
        help="Chemin du répertoire ChromaDB source pour la migration (par défaut: chroma_db)",
    )

    parser.add_argument(
        "--reset",
        action="store_true",
        help="Réinitialiser la base de données avant de l'utiliser",
    )

    parser.add_argument(
        "--use-gpu",
        action="store_true",
        help="Utiliser le GPU pour FAISS si disponible",
    )

    parser.add_argument(
        "--reranker",
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        help="Modèle de reranking à utiliser (par défaut: cross-encoder/ms-marco-MiniLM-L-6-v2)",
    )

    parser.add_argument(
        "--add-document",
        metavar="PATH",
        help="Ajouter un document (PDF, image, texte) à la base",
    )

    parser.add_argument("--description", help="Description pour une image ajoutée")

    parser.add_argument("--query", help="Requête textuelle à soumettre au système")

    parser.add_argument(
        "--image-query",
        metavar="PATH",
        help="Chemin vers une image à utiliser comme requête",
    )

    parser.add_argument(
        "--combined-query",
        action="store_true",
        help="Effectuer une requête combinée texte et image (nécessite --query et --image-query)",
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

    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.2,
        help="Seuil de similarité minimal pour inclure un résultat (par défaut: 0.2)",
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
    """Fonction principale du script de démonstration."""
    args = parser_arguments()

    # Initialiser le système RAG
    if args.migrate:
        print(
            f"Migration depuis la collection ChromaDB '{args.chroma_collection}' vers FAISS..."
        )
        rag_system = migrate_from_original_rag(
            original_collection_name=args.chroma_collection,
            original_persist_directory=args.chroma_path,
            new_persist_directory=args.db_path,
            llm_name=args.model,
        )
        if not rag_system:
            print("Erreur lors de la migration. Initialisation d'un nouveau système.")
            rag_system = EnhancedMultimodalRAG(
                llm_name=args.model,
                collection_name=args.collection,
                temperature=args.temperature,
                persist_directory=args.db_path,
                use_gpu=args.use_gpu,
                reranking_model=args.reranker,
                similarity_threshold=args.similarity_threshold,
            )
    else:
        rag_system = EnhancedMultimodalRAG(
            llm_name=args.model,
            collection_name=args.collection,
            temperature=args.temperature,
            persist_directory=args.db_path,
            use_gpu=args.use_gpu,
            reranking_model=args.reranker,
            similarity_threshold=args.similarity_threshold,
        )

    # Réinitialiser la base si demandé
    if args.reset:
        print("Réinitialisation de la base de données...")
        rag_system.vector_store.reset()

    # Ajouter un document si spécifié
    if args.add_document:
        if not os.path.exists(args.add_document):
            print(f"Erreur: Le fichier {args.add_document} n'existe pas")
            return

        print(f"Ajout du document: {args.add_document}")
        doc_ids = rag_system.add_document(
            document_path=args.add_document, description=args.description
        )
        print(f"Document ajouté avec succès! {len(doc_ids)} IDs générés.")

    # Traiter une requête textuelle
    if args.query and not (args.combined_query and args.image_query):
        print_divider(f"Requête: {args.query}")
        results = rag_system.query(
            query=args.query, top_k=args.top_k, use_reranking=not args.no_reranking
        )
        print_results(results)

    # Traiter une requête par image
    if args.image_query:
        if not os.path.exists(args.image_query):
            print(f"Erreur: L'image {args.image_query} n'existe pas")
            return

        # Si c'est une requête combinée texte-image
        if args.combined_query and args.query:
            print_divider(
                f"Requête combinée texte-image: '{args.query}' + {args.image_query}"
            )
            image = Image.open(args.image_query).convert("RGB")
            results = rag_system.query_text_and_image(
                text=args.query,
                image=image,
                top_k=args.top_k,
                use_reranking=not args.no_reranking,
            )
            print_results(results)
        # Sinon, c'est une requête par image uniquement
        elif not args.combined_query:
            print_divider(f"Requête par image: {args.image_query}")
            image = Image.open(args.image_query).convert("RGB")
            results = rag_system.query(
                query=image, top_k=args.top_k, use_reranking=not args.no_reranking
            )
            print_results(results)
        else:
            print(
                "Erreur: Pour une requête combinée, vous devez spécifier à la fois --query et --image-query"
            )
            return

    # Si aucune action spécifiée, afficher un exemple d'utilisation
    if not args.add_document and not args.query and not args.image_query:
        print_divider("Exemple d'utilisation")
        print("Exemples de commandes:")
        print("  1. Ajouter un document PDF:")
        print(f"     python {__file__} --add-document chemin/vers/document.pdf")
        print("\n  2. Ajouter une image avec description:")
        print(
            f'     python {__file__} --add-document chemin/vers/image.jpg --description "Description de l\'image"'
        )
        print("\n  3. Effectuer une requête textuelle:")
        print(
            f'     python {__file__} --query "Quelle information contient ce document?"'
        )
        print("\n  4. Effectuer une requête par image:")
        print(f"     python {__file__} --image-query chemin/vers/image.jpg")

        print("\n  5. Effectuer une requête combinée texte-image:")
        print(
            f'     python {__file__} --query "Décris cette image" --image-query chemin/vers/image.jpg --combined-query'
        )

        print("\n  6. Migrer une collection ChromaDB existante:")
        print(
            f"     python {__file__} --migrate --chroma-path chroma_db --chroma-collection multimodal_collection"
        )
        print("\nConsultez l'aide pour plus d'options:")
        print(f"  python {__file__} --help")


if __name__ == "__main__":
    main()
