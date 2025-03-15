#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module d'embeddings multimodaux amélioré
----------------------------------------
Ce module implémente une classe améliorée pour générer des embeddings
textuels et visuels en utilisant les modèles CLIP via Hugging Face Transformers,
offrant une meilleure robustesse, compatibilité et flexibilité.
"""

import torch
import numpy as np
from typing import List, Union, Optional
from PIL import Image
from transformers import CLIPProcessor, CLIPModel


class EnhancedEmbedder:
    """
    Classe qui encapsule les fonctionnalités d'embeddings multimodaux en utilisant
    les modèles CLIP (Contrastive Language-Image Pretraining) de Hugging Face Transformers.
    """

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        use_gpu: bool = False,
        batch_size: int = 16,
    ):
        """
        Initialise l'embedder multimodal avec le modèle CLIP spécifié.

        Args:
            model_name (str): Nom du modèle CLIP à utiliser
            use_gpu (bool): Utiliser le GPU pour le modèle CLIP si disponible
            batch_size (int): Taille des lots pour le traitement par lots
        """
        self.model_name = model_name
        self.batch_size = batch_size

        # Détermination du device
        self.device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        if self.device == "cuda":
            print(
                f"Utilisation du GPU pour EnhancedEmbedder ({torch.cuda.get_device_name(0)})"
            )
        else:
            print("Utilisation du CPU pour EnhancedEmbedder")

        # Chargement du modèle CLIP et du processeur
        print(f"Chargement du modèle CLIP {model_name}...")
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)

        # Dimension des embeddings
        self.embedding_dimension = self.model.config.projection_dim
        print(f"Dimension des embeddings: {self.embedding_dimension}")

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Génère des embeddings pour une liste de textes.

        Args:
            texts (List[str]): Liste de textes à convertir en embeddings

        Returns:
            np.ndarray: Tableau NumPy de dimension [len(texts), embedding_dim] avec les embeddings normalisés
        """
        if not texts:
            raise ValueError("La liste de textes ne peut pas être vide")

        embeddings = []

        # Traitement par lots pour optimiser les performances
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]

            with torch.no_grad():
                # Utilisation du processeur CLIP pour prétraiter les textes
                inputs = self.processor(
                    text=batch, return_tensors="pt", padding=True, truncation=True
                )

                # Transfert des entrées sur le dispositif approprié (CPU/GPU)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                # Obtention des caractéristiques textuelles
                text_features = self.model.get_text_features(**inputs)

                # Normalisation des embeddings
                text_features = text_features / text_features.norm(dim=1, keepdim=True)

                # Transfert des embeddings vers CPU pour retour en numpy
                batch_embeddings = text_features.cpu().numpy()
                embeddings.append(batch_embeddings)

        # Concaténation de tous les embeddings de tous les lots
        all_embeddings = np.vstack(embeddings)
        return all_embeddings

    def embed_images(self, images: List[Image.Image]) -> np.ndarray:
        """
        Génère des embeddings pour une liste d'images.

        Args:
            images (List[Image.Image]): Liste d'objets PIL.Image à convertir en embeddings

        Returns:
            np.ndarray: Tableau NumPy de dimension [len(images), embedding_dim] avec les embeddings normalisés
        """
        if not images:
            raise ValueError("La liste d'images ne peut pas être vide")

        # Vérification que toutes les images sont des objets PIL.Image
        for img in images:
            if not isinstance(img, Image.Image):
                raise ValueError(
                    f"Toutes les images doivent être des objets PIL.Image, reçu {type(img)}"
                )

        embeddings = []

        # Traitement par lots pour optimiser les performances
        for i in range(0, len(images), self.batch_size):
            batch = images[i : i + self.batch_size]

            with torch.no_grad():
                # Utilisation du processeur CLIP pour prétraiter les images
                inputs = self.processor(images=batch, return_tensors="pt")

                # Transfert des entrées sur le dispositif approprié (CPU/GPU)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                # Obtention des caractéristiques des images
                image_features = self.model.get_image_features(**inputs)

                # Normalisation des embeddings
                image_features = image_features / image_features.norm(
                    dim=1, keepdim=True
                )

                # Transfert des embeddings vers CPU pour retour en numpy
                batch_embeddings = image_features.cpu().numpy()
                embeddings.append(batch_embeddings)

        # Concaténation de tous les embeddings de tous les lots
        all_embeddings = np.vstack(embeddings)
        return all_embeddings

    def embed_text(self, text: str) -> np.ndarray:
        """
        Génère un embedding pour un seul texte.

        Args:
            text (str): Texte à convertir en embedding

        Returns:
            np.ndarray: Tableau NumPy de dimension [embedding_dim] avec l'embedding normalisé
        """
        return self.embed_texts([text])[0]

    def embed_image(self, image: Image.Image) -> np.ndarray:
        """
        Génère un embedding pour une seule image.

        Args:
            image (Image.Image): Image à convertir en embedding

        Returns:
            np.ndarray: Tableau NumPy de dimension [embedding_dim] avec l'embedding normalisé
        """
        return self.embed_images([image])[0]

    def get_embedding_dimension(self) -> int:
        """
        Retourne la dimension des embeddings générés par le modèle.

        Returns:
            int: Dimension des embeddings
        """
        return self.embedding_dimension

    def embed_text_and_image(self, text: str, image: Image.Image) -> np.ndarray:
        """
        Génère un embedding combiné pour un texte et une image.

        Cette méthode crée un embedding unique qui représente à la fois
        le texte et l'image, permettant des requêtes multimodales combinées.
        L'embedding résultant est la moyenne des embeddings normalisés du texte et de l'image.

        Args:
            text (str): Texte à combiner
            image (Image.Image): Image à combiner

        Returns:
            np.ndarray: Tableau NumPy de dimension [embedding_dim] avec l'embedding combiné normalisé
        """
        # Obtenir les embeddings individuels
        text_embedding = self.embed_text(text)
        image_embedding = self.embed_image(image)

        # Combiner les embeddings (moyenne simple)
        combined_embedding = (text_embedding + image_embedding) / 2.0

        # Re-normaliser l'embedding combiné
        norm = np.linalg.norm(combined_embedding)
        if norm > 0:
            combined_embedding = combined_embedding / norm

        return combined_embedding
