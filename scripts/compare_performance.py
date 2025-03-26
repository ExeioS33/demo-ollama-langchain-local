#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script de comparaison des performances entre les implémentations RAG multimodales
--------------------------------------------------------------------------------
Ce script permet de comparer les performances de la version originale (ChromaDB)
et de la version améliorée (FAISS) du système RAG multimodal,
particulièrement pour les requêtes impliquant des images.
"""

import os
import argparse
import time
import json
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# Importer les deux implémentations
from multimodal_rag import MultimodalRAG
from enhanced_multimodal_rag import EnhancedMultimodalRAG


def parser_arguments():
    """Parse les arguments en ligne de commande."""
    parser = argparse.ArgumentParser(
        description="Comparaison des implémentations RAG multimodales"
    )

    parser.add_argument(
        "--query",
        required=True,
        help="Requête textuelle à utiliser",
    )

    parser.add_argument(
        "--image",
        required=True,
        help="Chemin vers l'image à utiliser pour la requête",
    )

    parser.add_argument(
        "--model",
        default="llava:7b-v1.6-vicuna-q8_0",
        help="Modèle LLM à utiliser (défaut: llava:7b-v1.6-vicuna-q8_0)",
    )

    parser.add_argument(
        "--output",
        default="comparison_results.json",
        help="Fichier de sortie pour les résultats (défaut: comparison_results.json)",
    )

    return parser.parse_args()


def run_test(args):
    """Exécute les tests de comparaison."""
    results = {
        "query": args.query,
        "image": args.image,
        "model": args.model,
        "original": {},
        "enhanced": {},
    }

    # Vérifier que l'image existe
    if not os.path.exists(args.image):
        print(f"Erreur: L'image {args.image} n'existe pas.")
        return None

    # Étape 1: Test de la version originale avec ChromaDB
    print("\n" + "=" * 80)
    print(f"Test de la version originale (ChromaDB) avec {args.model}")
    print("=" * 80)

    start_time = time.time()

    try:
        # Initialiser le système original
        original_rag = MultimodalRAG(
            llm_name=args.model,
            collection_name="multimodal_collection",
            persist_directory="chroma_db",
            temperature=0.1,
        )

        # Exécuter la requête
        image = Image.open(args.image)
        original_results = original_rag.query(args.image, top_k=5)

        # Calculer le temps d'exécution
        original_time = time.time() - start_time

        # Stocker les résultats
        results["original"] = {
            "answer": original_results["answer"],
            "execution_time": original_time,
            "num_sources": len(original_results["sources"]),
            "sources": [
                {
                    "type": source["type"],
                    "similarity": source.get("similarity", 0),
                }
                for source in original_results["sources"]
            ],
        }

        print(f"\nRéponse originale ({original_time:.2f}s):")
        print("-" * 40)
        print(original_results["answer"])

    except Exception as e:
        print(f"Erreur lors du test de la version originale: {e}")
        results["original"]["error"] = str(e)

    # Étape 2: Test de la version améliorée avec FAISS
    print("\n" + "=" * 80)
    print(f"Test de la version améliorée (FAISS) avec {args.model}")
    print("=" * 80)

    start_time = time.time()

    try:
        # Initialiser le système amélioré
        enhanced_rag = EnhancedMultimodalRAG(
            llm_name=args.model,
            collection_name="enhanced_multimodal_collection",
            persist_directory="enhanced_vector_store",
            temperature=0.1,
            use_gpu=False,  # Changer à True si GPU disponible
        )

        # Exécuter la requête
        enhanced_results = enhanced_rag.query(args.image, top_k=5)

        # Calculer le temps d'exécution
        enhanced_time = time.time() - start_time

        # Stocker les résultats
        results["enhanced"] = {
            "answer": enhanced_results["answer"],
            "execution_time": enhanced_time,
            "num_sources": len(enhanced_results["sources"]),
            "sources": [
                {
                    "type": source["type"],
                    "similarity": source.get("similarity", 0),
                    "rerank_score": source.get("rerank_score", None),
                }
                for source in enhanced_results["sources"]
            ],
        }

        print(f"\nRéponse améliorée ({enhanced_time:.2f}s):")
        print("-" * 40)
        print(enhanced_results["answer"])

    except Exception as e:
        print(f"Erreur lors du test de la version améliorée: {e}")
        results["enhanced"]["error"] = str(e)

    # Comparer les temps d'exécution
    if (
        "execution_time" in results["original"]
        and "execution_time" in results["enhanced"]
    ):
        time_diff = (
            results["original"]["execution_time"]
            - results["enhanced"]["execution_time"]
        )
        time_ratio = results["original"]["execution_time"] / max(
            results["enhanced"]["execution_time"], 0.001
        )

        results["comparison"] = {
            "time_difference": time_diff,
            "time_ratio": time_ratio,
            "faster": "enhanced" if time_diff > 0 else "original",
            "speedup_percentage": (time_diff / results["original"]["execution_time"])
            * 100
            if results["original"]["execution_time"] > 0
            else 0,
        }

        print("\n" + "=" * 80)
        print(f"Comparaison des performances:")
        print("=" * 80)
        print(f"Version originale: {results['original']['execution_time']:.2f}s")
        print(f"Version améliorée: {results['enhanced']['execution_time']:.2f}s")

        if time_diff > 0:
            print(
                f"La version améliorée est {time_ratio:.2f}x plus rapide ({results['comparison']['speedup_percentage']:.1f}% d'amélioration)"
            )
        else:
            print(f"La version originale est {1 / time_ratio:.2f}x plus rapide")

    # Sauvegarder les résultats
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nRésultats sauvegardés dans {args.output}")

    return results


def plot_comparison(results):
    """Génère un graphique de comparaison des résultats."""
    if not results:
        return

    # Créer une figure avec deux sous-graphiques
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Graphique 1: Temps d'exécution
    execution_times = [
        results["original"].get("execution_time", 0),
        results["enhanced"].get("execution_time", 0),
    ]

    ax1.bar(
        ["Original (ChromaDB)", "Amélioré (FAISS)"],
        execution_times,
        color=["blue", "green"],
    )
    ax1.set_ylabel("Temps d'exécution (s)")
    ax1.set_title("Comparaison des temps d'exécution")

    for i, v in enumerate(execution_times):
        ax1.text(i, v + 0.1, f"{v:.2f}s", ha="center")

    # Graphique 2: Similarités des sources
    if "sources" in results["original"] and "sources" in results["enhanced"]:
        original_similarities = [
            s.get("similarity", 0) for s in results["original"]["sources"]
        ]
        enhanced_similarities = [
            s.get("similarity", 0) for s in results["enhanced"]["sources"]
        ]

        if (
            enhanced_similarities
            and "rerank_score" in results["enhanced"]["sources"][0]
        ):
            enhanced_rerank = [
                s.get("rerank_score", 0)
                for s in results["enhanced"]["sources"]
                if "rerank_score" in s
            ]

            # Graphique avec trois séries
            x = np.arange(
                min(
                    len(original_similarities),
                    len(enhanced_similarities),
                    len(enhanced_rerank),
                )
            )

            if len(x) > 0:
                width = 0.25

                # Ajuster les séries à la même longueur
                original_similarities = original_similarities[: len(x)]
                enhanced_similarities = enhanced_similarities[: len(x)]
                enhanced_rerank = enhanced_rerank[: len(x)]

                ax2.bar(
                    x - width,
                    original_similarities,
                    width,
                    label="Original Sim.",
                    color="blue",
                )
                ax2.bar(
                    x,
                    enhanced_similarities,
                    width,
                    label="Amélioré Sim.",
                    color="green",
                )
                ax2.bar(
                    x + width, enhanced_rerank, width, label="Rerank Score", color="red"
                )

                ax2.set_xlabel("Source #")
                ax2.set_ylabel("Score")
                ax2.set_title("Scores de similarité et reranking")
                ax2.set_xticks(x)
                ax2.legend()
        else:
            # Graphique avec deux séries
            x = np.arange(min(len(original_similarities), len(enhanced_similarities)))

            if len(x) > 0:
                width = 0.35

                # Ajuster les séries à la même longueur
                original_similarities = original_similarities[: len(x)]
                enhanced_similarities = enhanced_similarities[: len(x)]

                ax2.bar(
                    x - width / 2,
                    original_similarities,
                    width,
                    label="Original",
                    color="blue",
                )
                ax2.bar(
                    x + width / 2,
                    enhanced_similarities,
                    width,
                    label="Amélioré",
                    color="green",
                )

                ax2.set_xlabel("Source #")
                ax2.set_ylabel("Similarité")
                ax2.set_title("Scores de similarité")
                ax2.set_xticks(x)
                ax2.legend()

    # Ajuster la mise en page et sauvegarder
    plt.tight_layout()
    plt.savefig("comparison_results.png")
    print("Graphique de comparaison sauvegardé dans comparison_results.png")

    try:
        plt.show()
    except Exception:
        pass  # Ignorer si l'affichage n'est pas disponible


if __name__ == "__main__":
    args = parser_arguments()
    results = run_test(args)

    if results:
        try:
            plot_comparison(results)
        except Exception as e:
            print(f"Erreur lors de la génération du graphique: {e}")
