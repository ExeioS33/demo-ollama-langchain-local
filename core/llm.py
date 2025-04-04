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
import time

# Import de la configuration
from core.config import Config


class LLaVA:
    """Intégration avec le modèle LLaVA via Ollama."""

    def __init__(
        self,
        model_name: str = None,
        temperature: float = None,
        max_tokens: int = 1024,
        ollama_host: str = None,
        timeout: int = None,
        retry_count: int = None,
    ):
        """
        Initialise l'intégration avec LLaVA.

        Args:
            model_name: Nom du modèle LLaVA
            temperature: Température pour la génération
            max_tokens: Nombre maximum de tokens à générer
            ollama_host: URL de l'API Ollama
            timeout: Timeout en secondes pour les requêtes HTTP
            retry_count: Nombre de tentatives en cas d'échec
        """
        # Utiliser les valeurs de la configuration ou les valeurs par défaut
        self.model_name = model_name or Config.LLM_MODEL
        self.temperature = temperature or Config.TEMPERATURE
        self.max_tokens = max_tokens
        self.ollama_host = ollama_host or Config.OLLAMA_HOST
        self.timeout = timeout or Config.REQUEST_TIMEOUT
        self.retry_count = retry_count or Config.RETRY_COUNT

        # Vérifier si le modèle est multimodal
        self.is_multimodal = "llava" in self.model_name.lower()

        print(f"Initialisation du modèle LLM: {self.model_name}")
        if self.is_multimodal:
            print("Modèle multimodal détecté (LLaVA)")

        # Vérification que le modèle est disponible
        self._verify_model_availability()

    def _verify_model_availability(self):
        """Vérifie que le modèle est disponible dans Ollama."""
        try:
            response = requests.get(f"{self.ollama_host}/api/tags", timeout=10)
            if response.status_code == 200:
                available_models = [
                    model["name"] for model in response.json().get("models", [])
                ]
                if self.model_name not in available_models:
                    print(
                        f"⚠️ Avertissement: Le modèle {self.model_name} n'est pas trouvé dans Ollama."
                    )
                    print(f"Modèles disponibles: {available_models}")
                    # Essayer de trouver un modèle similaire
                    prefix = self.model_name.split(":")[0]
                    similar_models = [
                        m for m in available_models if m.startswith(prefix)
                    ]
                    if similar_models:
                        print(f"Modèles similaires disponibles: {similar_models}")
        except Exception as e:
            print(f"⚠️ Impossible de vérifier la disponibilité du modèle: {e}")

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
        system_prompt = """Tu es un assistant IA spécialisé dans les questions Ressources Humaines (RH), capable d'analyser des documents texte et images.
Je peux répondre à des questions générales, mais mon expertise principale concerne les RH.
Lorsque ta question contient des termes RH (par exemple: salaire, congé, performance, contrat, recrutement, politique interne), je prioriserai l'analyse du contexte documentaire fourni pour te donner la réponse la plus précise possible.
Si l'information n'est pas dans le contexte, je te l'indiquerai.
Je m'efforcerai de répondre de manière concise et précise."""

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
        for attempt in range(self.retry_count):
            try:
                print(
                    f"Envoi de la requête à Ollama (tentative {attempt + 1}/{self.retry_count})..."
                )

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
                    timeout=self.timeout,  # Utilisation du timeout configurable
                )

                if response.status_code == 200:
                    result = response.json()
                    return result.get("response", "")
                else:
                    error_msg = (
                        f"Erreur API Ollama ({response.status_code}): {response.text}"
                    )
                    print(error_msg)

                    # Si c'est une erreur 404 (modèle non trouvé), inutile de réessayer
                    if response.status_code == 404:
                        return (
                            f"Erreur: Modèle {self.model_name} non trouvé dans Ollama."
                        )

                    # Si ce n'est pas la dernière tentative, attendre avant de réessayer
                    if attempt < self.retry_count - 1:
                        wait_time = 2**attempt  # Attente exponentielle
                        print(f"Nouvel essai dans {wait_time} secondes...")
                        time.sleep(wait_time)
                    else:
                        return f"Erreur lors de la génération: {error_msg}"

            except requests.exceptions.Timeout:
                print(
                    f"Timeout lors de l'appel à Ollama (tentative {attempt + 1}/{self.retry_count})"
                )

                # Si ce n'est pas la dernière tentative, attendre avant de réessayer
                if attempt < self.retry_count - 1:
                    wait_time = 2**attempt  # Attente exponentielle
                    print(f"Nouvel essai dans {wait_time} secondes...")
                    time.sleep(wait_time)
                else:
                    return "Erreur: Timeout lors de la connexion à Ollama. Le modèle est peut-être trop grand ou Ollama manque de ressources."

            except Exception as e:
                print(
                    f"Erreur lors de l'appel à Ollama (tentative {attempt + 1}/{self.retry_count}): {e}"
                )

                # Si ce n'est pas la dernière tentative, attendre avant de réessayer
                if attempt < self.retry_count - 1:
                    wait_time = 2**attempt  # Attente exponentielle
                    print(f"Nouvel essai dans {wait_time} secondes...")
                    time.sleep(wait_time)
                else:
                    return f"Erreur technique: {str(e)}"

    def _generate_with_image(self, prompt: str, image_path: str) -> str:
        """
        Génère une réponse pour une requête incluant une image.
        Utilise l'API spécifique à Ollama pour les modèles multimodaux.
        """
        for attempt in range(self.retry_count):
            try:
                # Vérifier que l'image existe
                if not os.path.exists(image_path):
                    return f"Image introuvable: {image_path}"

                # Lire l'image en base64
                import base64

                with open(image_path, "rb") as image_file:
                    image_data = base64.b64encode(image_file.read()).decode("utf-8")

                print(
                    f"Envoi de la requête image à Ollama (tentative {attempt + 1}/{self.retry_count})..."
                )

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
                    timeout=self.timeout,  # Même timeout que pour le texte
                )

                if response.status_code == 200:
                    result = response.json()
                    return result.get("response", "")
                else:
                    error_msg = (
                        f"Erreur API Ollama ({response.status_code}): {response.text}"
                    )
                    print(error_msg)

                    # Si c'est une erreur 404 (modèle non trouvé), inutile de réessayer
                    if response.status_code == 404:
                        return (
                            f"Erreur: Modèle {self.model_name} non trouvé dans Ollama."
                        )

                    # Si ce n'est pas la dernière tentative, attendre avant de réessayer
                    if attempt < self.retry_count - 1:
                        wait_time = 2**attempt  # Attente exponentielle
                        print(f"Nouvel essai dans {wait_time} secondes...")
                        time.sleep(wait_time)
                    else:
                        return f"Erreur lors de la génération avec image: {error_msg}"

            except requests.exceptions.Timeout:
                print(
                    f"Timeout lors de l'appel à Ollama avec image (tentative {attempt + 1}/{self.retry_count})"
                )

                # Si ce n'est pas la dernière tentative, attendre avant de réessayer
                if attempt < self.retry_count - 1:
                    wait_time = 2**attempt  # Attente exponentielle
                    print(f"Nouvel essai dans {wait_time} secondes...")
                    time.sleep(wait_time)
                else:
                    return "Erreur: Timeout lors de la connexion à Ollama pour le traitement d'image. Le modèle est peut-être trop grand ou Ollama manque de ressources."

            except Exception as e:
                print(
                    f"Erreur lors de l'appel à Ollama avec image (tentative {attempt + 1}/{self.retry_count}): {e}"
                )

                # Si ce n'est pas la dernière tentative, attendre avant de réessayer
                if attempt < self.retry_count - 1:
                    wait_time = 2**attempt  # Attente exponentielle
                    print(f"Nouvel essai dans {wait_time} secondes...")
                    time.sleep(wait_time)
                else:
                    return f"Erreur technique avec traitement d'image: {str(e)}"
