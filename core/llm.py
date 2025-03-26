#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module d'intégration LLM pour LLaVA
-----------------------------------
Intégration avec le modèle LLaVA via Ollama pour la génération
de réponses basées sur le contexte.
"""

from typing import Optional, Dict, List, Any
from PIL import Image
import requests
import json
import os


class LLaVA:
    """Intégration avec le modèle LLaVA via Ollama."""

    def __init__(
        self,
        model_name: str = "llava:7b-v1.6-vicuna-q8_0",
        temperature: float = 0.1,
        max_tokens: int = 1024,
        ollama_host: str = "http://localhost:11434",
    ):
        """
        Initialise l'intégration avec LLaVA.

        Args:
            model_name: Nom du modèle LLaVA
            temperature: Température pour la génération
            max_tokens: Nombre maximum de tokens à générer
            ollama_host: URL de l'API Ollama
        """
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.ollama_host = ollama_host

        # Vérifier si le modèle est multimodal
        self.is_multimodal = "llava" in model_name.lower()

        print(f"Initialisation du modèle LLM: {model_name}")
        if self.is_multimodal:
            print("Modèle multimodal détecté (LLaVA)")

    def generate(
        self,
        query: str,
        context: str,
        is_image_query: bool = False,
        image_path: Optional[str] = None,
    ) -> str:
        """
        Génère une réponse basée sur la requête et le contexte.

        Args:
            query: Requête de l'utilisateur
            context: Contexte pour la réponse
            is_image_query: Si True, la requête concerne une image
            image_path: Chemin vers l'image (pour requêtes image)

        Returns:
            str: Réponse générée
        """
        # Construire le prompt
        system_prompt = """Tu es un assistant spécialisé dans l'analyse de documents multimodaux (texte et images).
Utilise uniquement le contexte fourni pour répondre à la question. 
Si tu ne trouves pas l'information dans le contexte, indique-le clairement.
Réponds de manière concise et précise."""

        # Préparer le prompt utilisateur
        formatted_context = f"Contexte :\n{context}\n\n"
        formatted_query = f"Question : {query}\n\nRéponse :"

        user_prompt = formatted_context + formatted_query

        # Cas spécial pour les requêtes image avec LLaVA
        if is_image_query and self.is_multimodal and image_path:
            return self._generate_with_image(user_prompt, image_path)

        # Cas standard (texte uniquement)
        return self._generate_text_only(system_prompt, user_prompt)

    def _generate_text_only(self, system_prompt: str, user_prompt: str) -> str:
        """Génère une réponse pour une requête textuelle."""
        try:
            # Appel à l'API Ollama
            response = requests.post(
                f"{self.ollama_host}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": user_prompt,
                    "system": system_prompt,
                    "temperature": self.temperature,
                    "num_predict": self.max_tokens,
                    "stream": False,
                },
                timeout=60,  # Timeout plus long pour les requêtes complexes
            )

            if response.status_code == 200:
                result = response.json()
                return result.get("response", "")
            else:
                error_msg = (
                    f"Erreur API Ollama ({response.status_code}): {response.text}"
                )
                print(error_msg)
                return f"Erreur lors de la génération: {error_msg}"

        except Exception as e:
            print(f"Erreur lors de l'appel à Ollama: {e}")
            return f"Erreur technique: {str(e)}"

    def _generate_with_image(self, prompt: str, image_path: str) -> str:
        """
        Génère une réponse pour une requête incluant une image.
        Utilise l'API spécifique à Ollama pour les modèles multimodaux.
        """
        try:
            # Vérifier que l'image existe
            if not os.path.exists(image_path):
                return f"Image introuvable: {image_path}"

            # Lire l'image en base64
            import base64

            with open(image_path, "rb") as image_file:
                image_data = base64.b64encode(image_file.read()).decode("utf-8")

            # Appel à l'API Ollama avec l'image
            response = requests.post(
                f"{self.ollama_host}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "images": [image_data],  # Format attendu par Ollama pour LLaVA
                    "temperature": self.temperature,
                    "num_predict": self.max_tokens,
                    "stream": False,
                },
                timeout=120,  # Plus long pour le traitement d'images
            )

            if response.status_code == 200:
                result = response.json()
                return result.get("response", "")
            else:
                error_msg = (
                    f"Erreur API Ollama ({response.status_code}): {response.text}"
                )
                print(error_msg)
                return f"Erreur lors de la génération avec image: {error_msg}"

        except Exception as e:
            print(f"Erreur lors de l'appel à Ollama avec image: {e}")
            return f"Erreur technique avec traitement d'image: {str(e)}"
