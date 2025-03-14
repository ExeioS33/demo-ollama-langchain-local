#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Démonstration du système RAG multimodal
---------------------------------------
Ce script montre comment utiliser le système RAG multimodal
créé dans multimodal_rag.py avec des exemples concrets.
"""

import os
import argparse
import sys
from PIL import Image
from multimodal_rag import MultimodalRAG


def main():
    parser = argparse.ArgumentParser(
        description="Démonstration du système RAG multimodal"
    )
    parser.add_argument(
        "--add", help="Ajouter un document (PDF, image, texte) au système"
    )
    parser.add_argument("--query", help="Requête textuelle pour le système")
    parser.add_argument("--image_query", help="Chemin vers une image pour une requête")
    parser.add_argument(
        "--model", default="qwen2.5:3b", help="Modèle Ollama à utiliser"
    )
    parser.add_argument(
        "--db_path", default="chroma_db", help="Chemin vers la base de données ChromaDB"
    )
    parser.add_argument(
        "--collection",
        default="multimodal_collection",
        help="Nom de la collection à utiliser",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Réinitialiser la base de données (supprime le dossier chroma_db)",
    )

    args = parser.parse_args()

    # Réinitialiser la base de données si demandé
    if args.reset and os.path.exists(args.db_path):
        import shutil

        print(f"Suppression de la base de données existante: {args.db_path}")
        shutil.rmtree(args.db_path)
        print("Base de données réinitialisée")

    # Créer le dossier de données s'il n'existe pas
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    os.makedirs(data_dir, exist_ok=True)

    # Afficher un message en ASCII art
    print("""
    ╔═══════════════════════════════════════════════════╗
    ║                                                   ║
    ║        SYSTÈME RAG MULTIMODAL AVEC CLIP           ║
    ║    Recherche de texte et d'images en un seul      ║
    ║                     espace                        ║
    ║                                                   ║
    ╚═══════════════════════════════════════════════════╝
    """)

    # Initialiser le système RAG
    print(f"Initialisation du système RAG multimodal avec le modèle {args.model}...")
    print(f"Base de données: {args.db_path}, Collection: {args.collection}")
    try:
        rag = MultimodalRAG(
            llm_name=args.model,
            collection_name=args.collection,
            persist_directory=args.db_path,
        )
        print("Système RAG multimodal initialisé avec succès!")
    except Exception as e:
        print(f"Erreur lors de l'initialisation du système RAG: {e}")
        sys.exit(1)

    # Si l'utilisateur veut ajouter un document
    if args.add:
        if not os.path.exists(args.add):
            print(f"Erreur: Le fichier {args.add} n'existe pas.")
            return

        print(f"Ajout du document: {args.add}")
        try:
            doc_ids = rag.add_document(args.add)
            print(f"Document ajouté avec succès! ({len(doc_ids)} éléments indexés)")

            # Afficher les premiers éléments indexés
            if doc_ids:
                print("Éléments indexés (premiers IDs):")
                for i, doc_id in enumerate(doc_ids[:3]):
                    print(f"  {i + 1}. {doc_id}")
                if len(doc_ids) > 3:
                    print(f"  ... et {len(doc_ids) - 3} autres")

        except Exception as e:
            print(f"Erreur lors de l'ajout du document: {e}")
            return

    # Si l'utilisateur fait une requête
    if args.query or args.image_query:
        # Requête avec image
        if args.image_query:
            if not os.path.exists(args.image_query):
                print(f"Erreur: L'image {args.image_query} n'existe pas.")
                return

            query = args.query if args.query else "Décris cette image en détail."
            print(f"Requête avec l'image {args.image_query}: '{query}'")

            try:
                # Charger l'image ou utiliser le chemin selon le type de requête
                if query.lower() in [
                    "décris cette image en détail.",
                    "décris cette image.",
                    "qu'y a-t-il sur cette image?",
                ]:
                    # Si la requête demande directement à décrire l'image, passer le chemin
                    print("Mode: Description directe de l'image")
                    result = rag.query(args.image_query)
                else:
                    # Sinon, effectuer une recherche avec l'image comme requête multimodale
                    print("Mode: Recherche contextuelle avec l'image + texte")
                    image = Image.open(args.image_query)
                    result = rag.query(query)

                # Afficher la réponse
                print("\nRÉPONSE:")
                print("=" * 50)
                print(result["answer"])
                print("=" * 50)

                # Afficher les sources utilisées
                if result["sources"]:
                    print("\nSOURCES UTILISÉES:")
                    for i, source in enumerate(result["sources"]):
                        if source["type"] == "image":
                            print(
                                f"  Image {i + 1}: {source['description']} (Similarité: {source['similarity']:.2f})"
                            )
                        else:
                            source_info = source["metadata"].get(
                                "source", "source inconnue"
                            )
                            page_info = (
                                f" (page {source['metadata']['page']})"
                                if "page" in source["metadata"]
                                else ""
                            )
                            print(
                                f"  Document {i + 1}{page_info}: {source_info} (Similarité: {source['similarity']:.2f})"
                            )
                else:
                    print("\nAucune source pertinente trouvée pour cette requête.")

            except Exception as e:
                print(f"Erreur lors de la requête avec image: {e}")
                import traceback

                traceback.print_exc()

        # Requête texte simple
        elif args.query:
            print(f"Requête: '{args.query}'")

            try:
                result = rag.query(args.query)

                # Afficher la réponse
                print("\nRÉPONSE:")
                print("=" * 50)
                print(result["answer"])
                print("=" * 50)

                # Afficher les sources utilisées
                if result["sources"]:
                    print("\nSOURCES UTILISÉES:")
                    for i, source in enumerate(result["sources"]):
                        if source["type"] == "image":
                            print(
                                f"  Image {i + 1}: {source['description']} (Similarité: {source['similarity']:.2f})"
                            )
                        else:
                            source_info = source["metadata"].get(
                                "source", "source inconnue"
                            )
                            page_info = (
                                f" (page {source['metadata']['page']})"
                                if "page" in source["metadata"]
                                else ""
                            )
                            print(
                                f"  Document {i + 1}{page_info}: {source_info} (Similarité: {source['similarity']:.2f})"
                            )
                else:
                    print("\nAucune source pertinente trouvée pour cette requête.")

            except Exception as e:
                print(f"Erreur lors de la requête texte: {e}")
                import traceback

                traceback.print_exc()

    # Si aucune opération n'est spécifiée, afficher un message d'aide
    if not args.add and not args.query and not args.image_query and not args.reset:
        print(
            "Aucune opération spécifiée. Utilisez --add pour ajouter un document ou --query/--image_query pour effectuer une requête."
        )
        print("Exemples d'utilisation:")
        print("  python multimodal_rag_demo.py --add document.pdf")
        print("  python multimodal_rag_demo.py --add image.jpg")
        print(
            '  python multimodal_rag_demo.py --query "Quelle information contient le document?"'
        )
        print(
            '  python multimodal_rag_demo.py --image_query image.jpg --query "Que représente cette image?"'
        )
        print(
            "  python multimodal_rag_demo.py --reset   # Réinitialiser la base de données"
        )


if __name__ == "__main__":
    main()
