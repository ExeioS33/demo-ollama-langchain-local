#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module de configuration pour le système RAG
------------------------------------------
Ce module gère le chargement des variables d'environnement depuis
le fichier .env et fournit une configuration centralisée pour
l'application de RAG avec chunking intelligent.
"""

import os
from typing import Dict, Any
from dotenv import load_dotenv
import torch

# Charger d'abord le fichier .env
load_dotenv()

# Puis surcharger avec .env.local s'il existe (prioritaire)
load_dotenv(".env.local", override=True)


class Config:
    """Classe de configuration pour le système RAG."""

    # Modèle LLM
    LLM_MODEL = os.getenv("RAG_LLM_MODEL", "qwen2.5:3b")

    # Chemin de stockage des vecteurs
    VECTOR_STORE_PATH = os.getenv("RAG_VECTOR_STORE_PATH", "data/vectors")

    # Paramètres GPU
    USE_GPU = os.getenv("RAG_USE_GPU", "false").lower() == "true"
    HAS_GPU = torch.cuda.is_available()

    # Paramètres LLM
    TEMPERATURE = float(os.getenv("RAG_TEMPERATURE", "0.1"))
    SIMILARITY_THRESHOLD = float(os.getenv("RAG_SIMILARITY_THRESHOLD", "0.2"))

    # Paramètres de chunking
    CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", "200"))

    # Paramètres HTTP
    REQUEST_TIMEOUT = int(os.getenv("RAG_REQUEST_TIMEOUT", "300"))
    RETRY_COUNT = int(os.getenv("RAG_RETRY_COUNT", "3"))

    # URL Ollama
    OLLAMA_HOST = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    @classmethod
    def get_all(cls) -> Dict[str, Any]:
        """Récupère toutes les variables de configuration sous forme de dictionnaire."""
        return {
            "llm_model": cls.LLM_MODEL,
            "vector_store_path": cls.VECTOR_STORE_PATH,
            "use_gpu": cls.USE_GPU and cls.HAS_GPU,
            "temperature": cls.TEMPERATURE,
            "similarity_threshold": cls.SIMILARITY_THRESHOLD,
            "chunk_size": cls.CHUNK_SIZE,
            "chunk_overlap": cls.CHUNK_OVERLAP,
            "request_timeout": cls.REQUEST_TIMEOUT,
            "retry_count": cls.RETRY_COUNT,
            "ollama_host": cls.OLLAMA_HOST,
        }

    @classmethod
    def print_config(cls) -> None:
        """Affiche la configuration courante."""
        print("=== Configuration du système RAG ===")
        print(f"Modèle LLM: {cls.LLM_MODEL}")
        print(f"Chemin de stockage: {cls.VECTOR_STORE_PATH}")
        print(f"Utilisation GPU: {cls.USE_GPU} (disponible: {cls.HAS_GPU})")
        print(f"Température LLM: {cls.TEMPERATURE}")
        print(f"Seuil de similarité: {cls.SIMILARITY_THRESHOLD}")
        print(f"Taille de chunk: {cls.CHUNK_SIZE}")
        print(f"Chevauchement de chunk: {cls.CHUNK_OVERLAP}")
        print(f"Timeout HTTP: {cls.REQUEST_TIMEOUT}s")
        print(f"Tentatives HTTP: {cls.RETRY_COUNT}")
        print(f"Hôte Ollama: {cls.OLLAMA_HOST}")
        print("===================================")


# Exécution standalone pour test
if __name__ == "__main__":
    Config.print_config()
