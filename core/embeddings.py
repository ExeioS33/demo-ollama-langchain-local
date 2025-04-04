#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module de Gestion des Embeddings
--------------------------------
Implémentation simplifiée de la génération d'embeddings
pour texte et images en utilisant CLIP.
"""

from typing import Union, List, Optional
import numpy as np
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel


class MultimodalEmbedder:
    """
    Classe qui génère des embeddings pour texte et images
    en utilisant le modèle CLIP (Contrastive Language-Image Pretraining).
    """

    def __init__(
        self, model_name: str = "openai/clip-vit-base-patch32", use_gpu: bool = False
    ):
        """
        Initialise l'embedder avec le modèle CLIP.

        Args:
            model_name: Nom du modèle CLIP à utiliser
            use_gpu: Utiliser le GPU si disponible
        """
        self.model_name = model_name

        # Déterminer le device (CPU/GPU)
        self.device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        if self.device == "cuda":
            print(
                f"Utilisation du GPU pour les embeddings ({torch.cuda.get_device_name(0)})"
            )
        else:
            print("Utilisation du CPU pour les embeddings")

        # Charger le modèle CLIP et son processeur
        print(f"Chargement du modèle CLIP {model_name}...")
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)

        # Dimension des embeddings
        self.embedding_dim = self.model.config.projection_dim
        # Alias for backward compatibility
        self.dimension = self.embedding_dim
        print(f"Dimension des embeddings: {self.embedding_dim}")

    def embed_text(self, text: str) -> np.ndarray:
        """
        Génère un embedding pour un texte.

        Args:
            text: Texte à convertir en embedding

        Returns:
            np.ndarray: Vecteur d'embedding normalisé
        """
        with torch.no_grad():
            # Prétraiter le texte avec le processeur CLIP
            inputs = self.processor(
                text=text, return_tensors="pt", padding=True, truncation=True
            )

            # Transférer les entrées sur le device approprié
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Obtenir l'embedding textuel
            text_features = self.model.get_text_features(**inputs)

            # Normaliser l'embedding
            text_features = text_features / text_features.norm(dim=1, keepdim=True)

            # Convertir en numpy
            embedding = text_features.cpu().numpy()[0]

            return embedding

    def embed_image(self, image: Image.Image) -> np.ndarray:
        """
        Génère un embedding pour une image.

        Args:
            image: Image à convertir en embedding

        Returns:
            np.ndarray: Vecteur d'embedding normalisé
        """
        with torch.no_grad():
            # Prétraiter l'image avec le processeur CLIP
            inputs = self.processor(images=image, return_tensors="pt")

            # Transférer les entrées sur le device approprié
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Obtenir l'embedding de l'image
            image_features = self.model.get_image_features(**inputs)

            # Normaliser l'embedding
            image_features = image_features / image_features.norm(dim=1, keepdim=True)

            # Convertir en numpy
            embedding = image_features.cpu().numpy()[0]

            return embedding

    def embed_text_and_image(self, text: str, image: Image.Image) -> np.ndarray:
        """
        Génère un embedding combiné pour un texte et une image.

        Args:
            text: Texte à combiner
            image: Image à combiner

        Returns:
            np.ndarray: Vecteur d'embedding combiné
        """
        # Obtenir les embeddings individuels
        text_embedding = self.embed_text(text)
        image_embedding = self.embed_image(image)

        # Pondération adaptative (favorise légèrement le texte)
        text_weight = 0.6
        image_weight = 0.4

        # Combinaison pondérée
        combined = (text_weight * text_embedding) + (image_weight * image_embedding)

        # Renormaliser
        norm = np.linalg.norm(combined)
        if norm > 0:
            combined = combined / norm

        return combined

    def embed(self, query: Union[str, Image.Image]) -> np.ndarray:
        """
        Génère un embedding pour une requête (texte ou image).

        Args:
            query: Requête textuelle ou image

        Returns:
            np.ndarray: Vecteur d'embedding
        """
        if isinstance(query, str):
            return self.embed_text(query)
        elif isinstance(query, Image.Image):
            return self.embed_image(query)
        else:
            raise TypeError(
                "La requête doit être une chaîne de caractères ou une image"
            )
