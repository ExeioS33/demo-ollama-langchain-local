#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module de gestion de l'index vectoriel FAISS
-------------------------------------------
Implémentation de l'indexation et de la recherche
vectorielle avec FAISS pour le système RAG.
"""

import os
import pickle
import faiss
import numpy as np
from typing import List, Dict, Union, Optional, Any
from PIL import Image
import shutil
from pathlib import Path
import fitz  # PyMuPDF


class FAISS:
    """Gestionnaire d'index vectoriel FAISS."""

    def __init__(
        self,
        embedder,
        collection_name: str = "multimodal_collection",
        persist_directory: str = "data/vectors",
        use_gpu: bool = False,
    ):
        """
        Initialise l'index vectoriel FAISS.

        Args:
            embedder: Instance du générateur d'embeddings
            collection_name: Nom de la collection
            persist_directory: Répertoire de persistance
            use_gpu: Utiliser le GPU si disponible
        """
        self.embedder = embedder
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.use_gpu = use_gpu

        # Dimension des embeddings
        self.dimension = embedder.embedding_dim

        # Créer le répertoire de persistance
        os.makedirs(persist_directory, exist_ok=True)

        # Chemins des fichiers
        self.index_path = os.path.join(
            persist_directory, f"{collection_name}_index.bin"
        )
        self.metadata_path = os.path.join(
            persist_directory, f"{collection_name}_metadata.pkl"
        )

        # Initialiser l'index et les métadonnées
        self.index = self._create_index()
        self.metadata = []

        # Sauvegarder l'index initial
        self._save()

    def _create_index(self) -> faiss.Index:
        """Crée un nouvel index FAISS optimisé."""
        # Créer un index de base optimisé pour la similarité cosinus
        index = faiss.IndexFlatIP(self.dimension)

        # Pour les grands volumes (>1M vecteurs), un index IVF serait préférable
        # index = faiss.IndexIVFFlat(quantizer, self.dimension, n_clusters, faiss.METRIC_INNER_PRODUCT)

        # Si GPU disponible, utiliser la version GPU de l'index
        if self.use_gpu and faiss.get_num_gpus() > 0:
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index)
            print("Utilisation du GPU pour FAISS")

        return index

    def _save(self):
        """Sauvegarde l'index et les métadonnées."""
        # Sauvegarder l'index
        if self.use_gpu:
            # Convertir en index CPU pour sauvegarde
            cpu_index = faiss.index_gpu_to_cpu(self.index)
            faiss.write_index(cpu_index, self.index_path)
        else:
            faiss.write_index(self.index, self.index_path)

        # Sauvegarder les métadonnées
        with open(self.metadata_path, "wb") as f:
            pickle.dump(self.metadata, f)

        print(f"Index et métadonnées sauvegardés: {len(self.metadata)} éléments")

    @classmethod
    def load(cls, persist_directory: str, embedder) -> "FAISS":
        """
        Charge un index FAISS existant.

        Args:
            persist_directory: Répertoire où se trouve l'index
            embedder: Instance du générateur d'embeddings

        Returns:
            FAISS: Instance initialisée avec l'index chargé
        """
        # Trouver le fichier d'index
        index_files = list(Path(persist_directory).glob("*_index.bin"))
        if not index_files:
            raise FileNotFoundError(f"Aucun index trouvé dans {persist_directory}")

        index_path = str(index_files[0])
        collection_name = os.path.basename(index_path).replace("_index.bin", "")
        metadata_path = os.path.join(
            persist_directory, f"{collection_name}_metadata.pkl"
        )

        # Vérifier si les métadonnées existent
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Métadonnées manquantes: {metadata_path}")

        # Créer une instance
        instance = cls(
            embedder=embedder,
            collection_name=collection_name,
            persist_directory=persist_directory,
            use_gpu=False,  # Sera corrigé après chargement
        )

        # Charger l'index
        instance.index = faiss.read_index(index_path)

        # Charger les métadonnées
        with open(metadata_path, "rb") as f:
            instance.metadata = pickle.load(f)

        print(f"Index chargé avec {len(instance.metadata)} éléments")
        return instance

    def add_texts(
        self, texts: List[str], metadatas: Optional[List[Dict]] = None
    ) -> List[str]:
        """
        Ajoute des textes à l'index.

        Args:
            texts: Liste de textes
            metadatas: Métadonnées optionnelles

        Returns:
            List[str]: IDs des textes ajoutés
        """
        ids = []

        # Générer les IDs et préparer les métadonnées
        if metadatas is None:
            metadatas = [{} for _ in texts]

        # Générer les embeddings
        embeddings = []
        for i, (text, metadata) in enumerate(zip(texts, metadatas)):
            # Générer un ID
            text_id = f"text_{len(self.metadata) + i}"
            ids.append(text_id)

            # Générer l'embedding
            embedding = self.embedder.embed_text(text)
            embeddings.append(embedding)

            # Ajouter les métadonnées
            self.metadata.append(
                {
                    "id": text_id,
                    "content": text,
                    "metadata": metadata,
                    "is_image": False,
                }
            )

        # Ajouter les embeddings à l'index
        if embeddings:
            embeddings_array = np.array(embeddings).astype("float32")
            self.index.add(embeddings_array)

            # Sauvegarder
            self._save()

        return ids

    def add_images(
        self,
        images: List[str],
        descriptions: Optional[List[str]] = None,
        metadatas: Optional[List[Dict]] = None,
    ) -> List[str]:
        """
        Ajoute des images à l'index.

        Args:
            images: Liste de chemins d'images
            descriptions: Descriptions optionnelles
            metadatas: Métadonnées optionnelles

        Returns:
            List[str]: IDs des images ajoutées
        """
        ids = []

        # Préparer les descriptions et métadonnées
        if descriptions is None:
            descriptions = [f"Image {i + 1}" for i in range(len(images))]

        if metadatas is None:
            metadatas = [{} for _ in images]

        # Générer les embeddings
        embeddings = []
        for i, (image_path, description, metadata) in enumerate(
            zip(images, descriptions, metadatas)
        ):
            try:
                # Charger l'image
                image = Image.open(image_path).convert("RGB")

                # Générer un ID
                image_id = f"image_{len(self.metadata) + i}"
                ids.append(image_id)

                # Générer l'embedding
                embedding = self.embedder.embed_image(image)
                embeddings.append(embedding)

                # Mettre à jour les métadonnées
                metadata["path"] = image_path
                metadata["filename"] = os.path.basename(image_path)

                # Ajouter les métadonnées
                self.metadata.append(
                    {
                        "id": image_id,
                        "content": description,
                        "metadata": metadata,
                        "is_image": True,
                    }
                )

            except Exception as e:
                print(f"Erreur lors du traitement de l'image {image_path}: {e}")

        # Ajouter les embeddings à l'index
        if embeddings:
            embeddings_array = np.array(embeddings).astype("float32")
            self.index.add(embeddings_array)

            # Sauvegarder
            self._save()

        return ids

    def add_pdf(self, pdf_path: str, metadatas: Optional[Dict] = None) -> List[str]:
        """
        Ajoute un document PDF à l'index.

        Args:
            pdf_path: Chemin vers le PDF
            metadatas: Métadonnées de base

        Returns:
            List[str]: IDs des éléments ajoutés
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF introuvable: {pdf_path}")

        if metadatas is None:
            metadatas = {}

        all_ids = []

        # Métadonnées de base communes
        base_metadata = {
            "source": pdf_path,
            "filename": os.path.basename(pdf_path),
            **metadatas,
        }

        # Ouvrir le document
        doc = fitz.open(pdf_path)

        # Créer un répertoire temporaire pour les images
        temp_dir = os.path.join(self.persist_directory, "temp_images")
        os.makedirs(temp_dir, exist_ok=True)

        # Traiter chaque page
        for page_num, page in enumerate(doc):
            # Extraire le texte
            text = page.get_text()

            if text.strip():
                # Ajouter le texte
                page_metadata = {**base_metadata, "page": page_num + 1, "type": "text"}
                text_ids = self.add_texts([text], [page_metadata])
                all_ids.extend(text_ids)

            # Extraire les images
            image_list = page.get_images(full=True)

            for img_index, img_info in enumerate(image_list):
                try:
                    xref = img_info[0]
                    base_img = doc.extract_image(xref)
                    img_bytes = base_img["image"]
                    img_ext = base_img["ext"]

                    # Sauvegarder temporairement
                    img_filename = f"page{page_num + 1}_img{img_index + 1}.{img_ext}"
                    img_path = os.path.join(temp_dir, img_filename)

                    with open(img_path, "wb") as img_file:
                        img_file.write(img_bytes)

                    # Ajouter l'image
                    img_metadata = {
                        **base_metadata,
                        "page": page_num + 1,
                        "image_index": img_index,
                        "type": "image",
                    }
                    description = f"Image {img_index + 1} de la page {page_num + 1} du document {os.path.basename(pdf_path)}"

                    img_ids = self.add_images([img_path], [description], [img_metadata])
                    all_ids.extend(img_ids)

                except Exception as e:
                    print(
                        f"Erreur lors de l'extraction de l'image {img_index} page {page_num + 1}: {e}"
                    )

        # Nettoyer
        shutil.rmtree(temp_dir, ignore_errors=True)

        return all_ids

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        filter_metadata: Optional[Dict] = None,
    ) -> List[Dict]:
        """
        Recherche les éléments les plus proches.

        Args:
            query_embedding: Embedding de la requête
            top_k: Nombre de résultats
            filter_metadata: Filtrage optionnel

        Returns:
            List[Dict]: Résultats avec métadonnées et scores
        """
        if len(self.metadata) == 0:
            return []

        # Préparer la requête
        q = np.array([query_embedding]).astype("float32")

        # Effectuer la recherche
        distances, indices = self.index.search(q, min(top_k * 2, len(self.metadata)))

        # Convertir en scores de similarité (produit scalaire normalisé)
        similarities = distances[0]
        indices = indices[0]

        # Préparer les résultats
        results = []

        for idx, similarity in zip(indices, similarities):
            # Récupérer les métadonnées
            meta_entry = self.metadata[idx]

            # Appliquer le filtre si spécifié
            if filter_metadata:
                skip = False
                for key, value in filter_metadata.items():
                    meta_value = meta_entry["metadata"].get(key)
                    if meta_value != value:
                        skip = True
                        break
                if skip:
                    continue

            # Ajouter aux résultats
            results.append(
                {
                    "id": meta_entry["id"],
                    "content": meta_entry["content"],
                    "metadata": meta_entry["metadata"],
                    "is_image": meta_entry["is_image"],
                    "similarity": float(similarity),
                }
            )

        # Trier par score et limiter
        results = sorted(results, key=lambda x: x["similarity"], reverse=True)

        return results[:top_k]
