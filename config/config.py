#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Configuration du système RAG multimodal
-------------------------------------
Paramètres centralisés pour les différents composants du système.
"""

import os
from typing import Dict, Any


# Répertoires principaux
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
VECTOR_STORE_DIR = os.path.join(DATA_DIR, "vectors")
MODELS_DIR = os.path.join(DATA_DIR, "models")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")

# Configuration par défaut
DEFAULT_CONFIG = {
    # Paramètres généraux
    "use_gpu": False,
    "debug": True,
    # Paramètres LLM
    "llm": {
        "model_name": "llava:7b-v1.6-vicuna-q8_0",
        "temperature": 0.1,
        "max_tokens": 1024,
        "ollama_host": "http://localhost:11434",
    },
    # Paramètres Embeddings
    "embeddings": {
        "model_name": "openai/clip-vit-base-patch32",
        "batch_size": 16,
        "text_weight": 0.6,
        "image_weight": 0.4,
    },
    # Paramètres Vector Store
    "vector_store": {
        "collection_name": "multimodal_collection",
        "persist_directory": VECTOR_STORE_DIR,
        "similarity_threshold": 0.2,
    },
    # Paramètres API
    "api": {"host": "0.0.0.0", "port": 8000, "debug": True},
}


def get_config() -> Dict[str, Any]:
    """
    Récupère la configuration complète avec les valeurs des variables d'environnement.

    Returns:
        Dict[str, Any]: Configuration complète
    """
    config = DEFAULT_CONFIG.copy()

    # Override avec les variables d'environnement
    if os.environ.get("USE_GPU"):
        config["use_gpu"] = os.environ.get("USE_GPU").lower() in ("true", "1", "yes")

    if os.environ.get("LLM_MODEL"):
        config["llm"]["model_name"] = os.environ.get("LLM_MODEL")

    if os.environ.get("LLM_TEMPERATURE"):
        config["llm"]["temperature"] = float(os.environ.get("LLM_TEMPERATURE"))

    if os.environ.get("OLLAMA_HOST"):
        config["llm"]["ollama_host"] = os.environ.get("OLLAMA_HOST")

    if os.environ.get("EMBEDDING_MODEL"):
        config["embeddings"]["model_name"] = os.environ.get("EMBEDDING_MODEL")

    if os.environ.get("API_PORT"):
        config["api"]["port"] = int(os.environ.get("API_PORT"))

    return config


# Configuration active
CONFIG = get_config()
