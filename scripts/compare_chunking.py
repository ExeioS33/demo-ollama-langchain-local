#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Comparaison des Méthodes de Chunking
-----------------------------------
Ce script compare les performances du chunking traditionnel (par page) vs intelligent (récursif).
"""

import os
import sys
import time
import argparse
import tempfile
import json
from pathlib import Path
from typing import Dict, List, Tuple

# Ajouter le répertoire parent au chemin Python
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Importer les composants nécessaires
from core.rag import RAGSystem
from core.embeddings import MultimodalEmbedder
from core.vector_operations import TextSplitter, FAISS


# Classes pour simuler l'ancien système de chunking
class LegacyChunker:
    """Simuler l'ancien système de chunking (par page ou document entier)."""

    @staticmethod
    def chunk_text(text: str, metadata: Dict = None) -> List[Dict]:
        """Ancienne méthode: un seul chunk pour tout le texte."""
        if metadata is None:
            metadata = {}

        return [{"content": text, "metadata": {**metadata, "legacy_chunking": True}}]

    @staticmethod
    def chunk_pdf(pdf_path: str, metadata: Dict = None) -> List[Dict]:
        """Ancienne méthode: un chunk par page, sans chevauchement."""
        if metadata is None:
            metadata = {}

        try:
            import PyPDF2
        except ImportError:
            print("PyPDF2 n'est pas installé. Exécutez 'pip install PyPDF2'.")
            return []

        chunks = []

        try:
            with open(pdf_path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                total_pages = len(reader.pages)

                for page_num, page in enumerate(reader.pages):
                    page_text = page.extract_text()

                    if not page_text.strip():
                        continue

                    page_metadata = metadata.copy()
                    page_metadata.update(
                        {
                            "page": page_num + 1,
                            "total_pages": total_pages,
                            "legacy_chunking": True,
                        }
                    )

                    chunks.append({"content": page_text, "metadata": page_metadata})
        except Exception as e:
            print(f"Erreur lors du traitement du PDF {pdf_path}: {e}")

        return chunks


def create_sample_document() -> str:
    """Crée un long document échantillon pour la démonstration."""
    with open("doc/RAG_Modules_Integration.md", "r") as f:
        sample_content = f.read()

    # Multiplier le contenu pour créer un document plus long
    multiplied_content = sample_content * 3

    # Ajouter quelques sections spécifiques pour tester la recherche
    specific_section = """
# Section Test pour Recherche

Cette section contient une information unique qui devrait être facilement trouvée lors d'une recherche.
La méthode de chunking intelligente devrait mieux préserver ce contexte spécifique.

## Points clés à extraire:
- Le chunking intelligent préserve mieux le contexte sémantique
- Les métadonnées enrichies aident à la traçabilité des sources
- La classe TextSplitter optimise le découpage pour les modèles de langage
- L'architecture modulaire facilite l'extension du système RAG

Ces informations devraient être correctement identifiées lors d'une recherche ciblée.
"""

    # Insérer la section spécifique au milieu du document
    parts = multiplied_content.split("## Flux d'information", 2)
    if len(parts) > 1:
        final_content = parts[0] + specific_section + "## Flux d'information" + parts[1]
    else:
        final_content = multiplied_content + specific_section

    # Écrire le contenu dans un fichier temporaire
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".md") as f:
        f.write(final_content)
        temp_file = f.name

    return temp_file


def run_comparison(
    input_file: str,
    query: str = "Explique les avantages du chunking intelligent",
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    use_gpu: bool = False,
):
    """
    Compare l'ancien et le nouveau système de chunking.

    Args:
        input_file: Chemin vers le fichier d'entrée
        query: Requête à utiliser pour la comparaison
        chunk_size: Taille des chunks pour le système intelligent
        chunk_overlap: Chevauchement entre chunks
        use_gpu: Utiliser le GPU si disponible
    """
    print(f"\n{'=' * 80}")
    print(f"COMPARAISON DES MÉTHODES DE CHUNKING")
    print(f"{'=' * 80}")

    # Initialiser l'embedder commun
    embedder = MultimodalEmbedder(use_gpu=use_gpu)

    # ---------- TEST AVEC CHUNKING CLASSIQUE ----------
    print("\n\n1. TEST AVEC LA MÉTHODE DE CHUNKING TRADITIONNELLE\n")

    # Créer un index temporaire pour le test legacy
    legacy_vector_dir = tempfile.mkdtemp(prefix="legacy_vectors_")
    legacy_db = FAISS(embedder=embedder, persist_directory=legacy_vector_dir)

    # Remplacer temporairement le TextSplitter par notre LegacyChunker
    original_text_splitter = legacy_db.text_splitter
    legacy_db.text_splitter = None

    # Lire le contenu du fichier
    with open(input_file, "r") as f:
        content = f.read()

    # Traiter avec l'ancien système
    start_legacy = time.time()
    legacy_chunks = LegacyChunker.chunk_text(content)

    # Ajouter manuellement à l'index
    legacy_ids = []
    for chunk in legacy_chunks:
        chunk_content = chunk["content"]
        chunk_metadata = chunk["metadata"]

        # Générer l'embedding
        embedding = embedder.embed(chunk_content)

        if embedding is not None:
            # Générer un ID unique
            import uuid

            doc_id = str(uuid.uuid4())

            # Ajouter à l'index FAISS
            legacy_db.index.add(embedding.reshape(1, -1))

            # Stocker les métadonnées
            legacy_db.metadata.append(
                {
                    "id": doc_id,
                    "content": chunk_content,
                    "metadata": chunk_metadata,
                    "is_image": False,
                }
            )

            legacy_ids.append(doc_id)

    legacy_ingestion_time = time.time() - start_legacy
    print(f"Temps d'ingestion: {legacy_ingestion_time:.2f} secondes")
    print(f"Nombre de chunks générés: {len(legacy_ids)}")

    # Faire une recherche
    query_embedding = embedder.embed(query)
    start_search = time.time()
    legacy_results = legacy_db.search(query_embedding, top_k=3)
    legacy_search_time = time.time() - start_search

    print(f"Temps de recherche: {legacy_search_time:.2f} secondes")
    print(f"\nRésultats trouvés:")
    for i, result in enumerate(legacy_results):
        similarity = result.get("similarity", 0) * 100
        content_preview = (
            result["content"][:100] + "..."
            if len(result["content"]) > 100
            else result["content"]
        )
        print(f"\n[Résultat {i + 1}] - Similarité: {similarity:.1f}%")
        print(f"Aperçu: {content_preview}")

    # ---------- TEST AVEC CHUNKING INTELLIGENT ----------
    print("\n\n2. TEST AVEC LA MÉTHODE DE CHUNKING INTELLIGENTE\n")

    # Créer un index temporaire pour le test intelligent
    intelligent_vector_dir = tempfile.mkdtemp(prefix="intelligent_vectors_")
    intelligent_db = FAISS(embedder=embedder, persist_directory=intelligent_vector_dir)

    # Configurer le chunking intelligent
    intelligent_db.text_splitter.update_splitter(chunk_size, chunk_overlap)

    # Traiter avec le nouveau système
    start_intelligent = time.time()
    with open(input_file, "r") as f:
        content = f.read()

    intelligent_chunks = intelligent_db.text_splitter.split_text(content)

    # Ajouter à l'index
    intelligent_ids = []
    for chunk in intelligent_chunks:
        chunk_content = chunk["content"]
        chunk_metadata = chunk["metadata"]

        # Générer l'embedding
        embedding = embedder.embed(chunk_content)

        if embedding is not None:
            # Générer un ID unique
            import uuid

            doc_id = str(uuid.uuid4())

            # Ajouter à l'index FAISS
            intelligent_db.index.add(embedding.reshape(1, -1))

            # Stocker les métadonnées
            intelligent_db.metadata.append(
                {
                    "id": doc_id,
                    "content": chunk_content,
                    "metadata": chunk_metadata,
                    "is_image": False,
                }
            )

            intelligent_ids.append(doc_id)

    intelligent_ingestion_time = time.time() - start_intelligent
    print(f"Temps d'ingestion: {intelligent_ingestion_time:.2f} secondes")
    print(f"Nombre de chunks générés: {len(intelligent_ids)}")

    # Statistiques sur les chunks
    chunk_sizes = [len(m["content"]) for m in intelligent_db.metadata]
    avg_chunk_size = sum(chunk_sizes) / len(chunk_sizes) if chunk_sizes else 0
    print(f"Taille moyenne des chunks: {avg_chunk_size:.1f} caractères")

    # Faire une recherche
    start_search = time.time()
    intelligent_results = intelligent_db.search(query_embedding, top_k=3)
    intelligent_search_time = time.time() - start_search

    print(f"Temps de recherche: {intelligent_search_time:.2f} secondes")
    print(f"\nRésultats trouvés:")
    for i, result in enumerate(intelligent_results):
        similarity = result.get("similarity", 0) * 100
        metadata = result.get("metadata", {})
        content_preview = (
            result["content"][:100] + "..."
            if len(result["content"]) > 100
            else result["content"]
        )

        # Afficher des métadonnées utiles
        meta_info = []
        if "chunk_index" in metadata and "total_chunks" in metadata:
            meta_info.append(
                f"Chunk {metadata['chunk_index'] + 1}/{metadata['total_chunks']}"
            )
        if "extracted_title" in metadata:
            meta_info.append(
                "Titre extrait" if metadata["extracted_title"] else "Sans titre"
            )

        print(f"\n[Résultat {i + 1}] - Similarité: {similarity:.1f}%")
        if meta_info:
            print(f"Métadonnées: {', '.join(meta_info)}")
        print(f"Aperçu: {content_preview}")

    # ---------- COMPARAISON DES RÉSULTATS ----------
    print("\n\n3. ANALYSE COMPARATIVE\n")

    # Comparer les temps
    time_diff = (
        legacy_ingestion_time / intelligent_ingestion_time
        if intelligent_ingestion_time > 0
        else 0
    )
    print(f"Rapport des temps d'ingestion (ancien/nouveau): {time_diff:.2f}x")

    search_time_diff = (
        legacy_search_time / intelligent_search_time
        if intelligent_search_time > 0
        else 0
    )
    print(f"Rapport des temps de recherche (ancien/nouveau): {search_time_diff:.2f}x")

    # Comparer le nombre de chunks
    chunk_ratio = len(intelligent_ids) / len(legacy_ids) if len(legacy_ids) > 0 else 0
    print(f"Ratio du nombre de chunks (nouveau/ancien): {chunk_ratio:.2f}x")

    # Comparer les scores de similarité
    legacy_max_sim = (
        max([r.get("similarity", 0) for r in legacy_results]) if legacy_results else 0
    )
    intelligent_max_sim = (
        max([r.get("similarity", 0) for r in intelligent_results])
        if intelligent_results
        else 0
    )

    sim_improvement = (
        (intelligent_max_sim / legacy_max_sim - 1) * 100 if legacy_max_sim > 0 else 0
    )
    print(f"Amélioration du score de similarité: {sim_improvement:.1f}%")

    # Nettoyer les répertoires temporaires
    import shutil

    shutil.rmtree(legacy_vector_dir, ignore_errors=True)
    shutil.rmtree(intelligent_vector_dir, ignore_errors=True)


def main():
    """Fonction principale."""
    parser = argparse.ArgumentParser(
        description="Comparaison des méthodes de chunking (traditionnelle vs intelligente)"
    )

    parser.add_argument(
        "--input", "-i", help="Chemin vers un document à utiliser pour la démonstration"
    )

    parser.add_argument(
        "--query",
        "-q",
        default="Explique les avantages du chunking intelligent",
        help="Requête à utiliser pour la démonstration",
    )

    parser.add_argument(
        "--chunk-size",
        type=int,
        default=500,
        help="Taille des chunks en caractères pour le chunking intelligent",
    )

    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=100,
        help="Chevauchement entre chunks en caractères pour le chunking intelligent",
    )

    parser.add_argument(
        "--gpu", action="store_true", help="Utiliser le GPU si disponible"
    )

    args = parser.parse_args()

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
        # Exécuter la comparaison
        run_comparison(
            input_file=input_file,
            query=args.query,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            use_gpu=args.gpu,
        )
    except Exception as e:
        print(f"Erreur lors de l'exécution de la comparaison: {e}")
        import traceback

        traceback.print_exc()
        return 1
    finally:
        # Si c'est un fichier temporaire, le supprimer
        if not args.input and input_file and os.path.exists(input_file):
            os.unlink(input_file)
            print(f"\nFichier temporaire supprimé: {input_file}")

    print("\nComparaison terminée!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
