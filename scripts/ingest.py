#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script d'ingestion de données pour le système RAG multimodal
----------------------------------------------------------
Ce script permet d'ajouter des documents (PDF, images, textes)
au système RAG multimodal.
"""

import os
import argparse
import glob
from typing import List, Optional
from tqdm import tqdm
import sys

# Ajouter le répertoire parent au chemin Python
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Importer le système RAG
from core.rag import RAGSystem


def parse_args():
    """Parse les arguments en ligne de commande."""
    parser = argparse.ArgumentParser(
        description="Script d'ingestion de données pour le système RAG multimodal"
    )

    parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="Chemin vers un fichier ou un répertoire à ingérer",
    )

    parser.add_argument(
        "--recursive",
        "-r",
        action="store_true",
        help="Ingérer les fichiers récursivement (pour les répertoires)",
    )

    parser.add_argument(
        "--pattern",
        "-p",
        default="*.*",
        help="Pattern de fichiers à ingérer (ex: '*.pdf', '*.jpg')",
    )

    parser.add_argument(
        "--description",
        "-d",
        help="Description des fichiers (pour les images uniquement)",
    )

    parser.add_argument(
        "--gpu", action="store_true", help="Utiliser le GPU si disponible"
    )

    return parser.parse_args()


def get_file_list(input_path: str, pattern: str, recursive: bool) -> List[str]:
    """
    Obtient la liste des fichiers à ingérer.

    Args:
        input_path: Chemin vers un fichier ou répertoire
        pattern: Pattern de fichiers à ingérer
        recursive: Ingérer récursivement

    Returns:
        List[str]: Liste des chemins de fichiers
    """
    if os.path.isfile(input_path):
        return [input_path]

    if os.path.isdir(input_path):
        if recursive:
            pattern_path = os.path.join(input_path, "**", pattern)
            return glob.glob(pattern_path, recursive=True)
        else:
            pattern_path = os.path.join(input_path, pattern)
            return glob.glob(pattern_path)

    return []


def main():
    """Fonction principale d'ingestion."""
    args = parse_args()

    # Vérifier que le chemin d'entrée existe
    if not os.path.exists(args.input):
        print(f"Erreur: le chemin {args.input} n'existe pas")
        return 1

    # Obtenir la liste des fichiers
    files = get_file_list(args.input, args.pattern, args.recursive)

    if not files:
        print(f"Aucun fichier trouvé correspondant au pattern '{args.pattern}'")
        return 1

    print(f"Trouvé {len(files)} fichier(s) à ingérer")

    # Initialiser le système RAG
    rag_system = RAGSystem(use_gpu=args.gpu)

    # Statistiques
    total_elements_added = 0

    # Ingérer chaque fichier
    for file_path in tqdm(files, desc="Ingestion"):
        try:
            ids = rag_system.add_document(file_path, description=args.description)
            total_elements_added += len(ids)
            print(f"✅ {file_path}: {len(ids)} élément(s) ajouté(s)")
        except Exception as e:
            print(f"❌ Erreur lors de l'ingestion de {file_path}: {str(e)}")

    print(
        f"\nIngestion terminée. {total_elements_added} élément(s) ajouté(s) au total."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
