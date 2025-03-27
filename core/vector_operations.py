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
from typing import List, Dict, Union, Optional, Any, Tuple
from PIL import Image
import shutil
from pathlib import Path
import fitz  # PyMuPDF
import re
from langchain.text_splitter import RecursiveCharacterTextSplitter
import uuid


class TextSplitter:
    """
    Classe pour découper intelligemment du texte en chunks.
    Utilise RecursiveCharacterTextSplitter de LangChain.
    """

    def __init__(self, chunk_size=1000, chunk_overlap=200):
        """
        Initialise le TextSplitter avec les paramètres spécifiés.

        Args:
            chunk_size: Taille des chunks en caractères
            chunk_overlap: Chevauchement entre chunks en caractères
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = ["\n\n", "\n", ". ", ", ", " ", ""]

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=self.separators,
        )

    def update_splitter(self, chunk_size=None, chunk_overlap=None):
        """
        Met à jour les paramètres du splitter et récrée l'instance.

        Args:
            chunk_size: Nouvelle taille des chunks
            chunk_overlap: Nouveau chevauchement entre chunks
        """
        if chunk_size is not None:
            self.chunk_size = chunk_size
        if chunk_overlap is not None:
            self.chunk_overlap = chunk_overlap

        # Recréer le splitter avec les nouveaux paramètres
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=self.separators,
        )
        return self

    def __str__(self):
        """Représentation sous forme de chaîne."""
        return f"TextSplitter(chunk_size={self.chunk_size}, chunk_overlap={self.chunk_overlap})"

    def split_text(self, text, metadata=None):
        """
        Découpe un texte en chunks avec métadonnées.

        Args:
            text: Texte à découper
            metadata: Métadonnées de base à enrichir

        Returns:
            List[Dict]: Liste de dicts avec 'content' et 'metadata'
        """
        if metadata is None:
            metadata = {}

        # Extraire un titre si possible
        title = self._extract_title(text)
        if title and "title" not in metadata:
            metadata["title"] = title

        # Découper le texte
        raw_chunks = self.splitter.split_text(text)

        # Créer une liste de chunks avec métadonnées
        chunks = []
        for i, chunk_text in enumerate(raw_chunks):
            # Copier les métadonnées de base et ajouter des informations sur le chunk
            chunk_metadata = metadata.copy()
            chunk_metadata.update(
                {
                    "chunk_index": i,
                    "total_chunks": len(raw_chunks),
                    "chunk_size": len(chunk_text),
                    "extracted_title": title is not None,
                }
            )

            chunks.append({"content": chunk_text, "metadata": chunk_metadata})

        return chunks

    def split_pdf_page(self, page_text, page_num, total_pages, metadata=None):
        """
        Découpe une page de PDF en chunks avec métadonnées.

        Args:
            page_text: Texte de la page
            page_num: Numéro de la page
            total_pages: Nombre total de pages
            metadata: Métadonnées de base à enrichir

        Returns:
            List[Dict]: Liste de dicts avec 'content' et 'metadata'
        """
        if metadata is None:
            metadata = {}

        # Ajouter les informations de page aux métadonnées
        page_metadata = metadata.copy()
        page_metadata.update(
            {
                "page": page_num,
                "total_pages": total_pages,
            }
        )

        # Utiliser la méthode split_text pour découper le texte de la page
        page_chunks = self.split_text(page_text, page_metadata)

        return page_chunks

    def _extract_title(self, text):
        """
        Tente d'extraire un titre significatif du texte.

        Args:
            text: Texte à analyser

        Returns:
            str or None: Titre extrait ou None si aucun titre trouvé
        """
        # Essayer d'extraire un titre de type Markdown (##, ###)
        lines = text.split("\n")
        for line in lines[:5]:  # Examiner uniquement les 5 premières lignes
            line = line.strip()
            if line.startswith("# "):  # Titre principal
                return line[2:].strip()
            elif line.startswith("## "):  # Sous-titre
                return line[3:].strip()

        # Si aucun titre markdown, prendre la première ligne non vide
        for line in lines[:5]:
            line = line.strip()
            if line and len(line) < 100:  # Une ligne courte non vide
                return line

        return None


class FAISS:
    """Classe pour gérer les opérations vectorielles avec FAISS."""

    def __init__(self, embedder, persist_directory="vector_store"):
        """
        Initialise un nouvel index FAISS.

        Args:
            embedder: Instance de MultimodalEmbedder pour générer les embeddings
            persist_directory: Répertoire pour persister l'index
        """
        self.embedder = embedder
        self.persist_directory = persist_directory
        self.metadata = []
        self.text_splitter = TextSplitter()

        # Créer le répertoire de persistance s'il n'existe pas
        os.makedirs(persist_directory, exist_ok=True)

        # Initialiser l'index FAISS
        self.dimension = self.embedder.dimension
        self.index = faiss.IndexFlatL2(self.dimension)

        # Charger l'index s'il existe
        index_path = os.path.join(persist_directory, "index.faiss")
        metadata_path = os.path.join(persist_directory, "metadata.json")

        if os.path.exists(index_path) and os.path.exists(metadata_path):
            self.load_index(index_path, metadata_path)

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
        # Créer le répertoire de persistance s'il n'existe pas
        os.makedirs(self.persist_directory, exist_ok=True)

        # Chemins des fichiers
        index_path = os.path.join(self.persist_directory, "index.faiss")
        metadata_path = os.path.join(self.persist_directory, "metadata.json")

        # Sauvegarder l'index FAISS
        faiss.write_index(self.index, index_path)

        # Sauvegarder les métadonnées au format JSON pour faciliter le débogage
        import json

        with open(metadata_path, "w", encoding="utf-8") as f:
            # Convertir les métadonnées en format JSON-compatible
            json_metadata = []
            for entry in self.metadata:
                # Créer une copie pour éviter de modifier l'original
                json_entry = {
                    "id": entry["id"],
                    "content": entry["content"],
                    "metadata": entry["metadata"],
                    "is_image": entry["is_image"],
                }
                json_metadata.append(json_entry)

            json.dump(json_metadata, f, ensure_ascii=False, indent=2)

        print(
            f"Index et métadonnées sauvegardés: {len(self.metadata)} éléments dans {self.persist_directory}"
        )

    def load_index(self, index_path, metadata_path):
        """
        Charge un index FAISS existant et ses métadonnées.

        Args:
            index_path: Chemin vers le fichier d'index FAISS
            metadata_path: Chemin vers le fichier de métadonnées
        """
        try:
            # Charger l'index FAISS
            self.index = faiss.read_index(index_path)

            # Charger les métadonnées
            import json

            with open(metadata_path, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)

            print(
                f"Index chargé avec {len(self.metadata)} éléments depuis {index_path}"
            )

        except Exception as e:
            print(f"Erreur lors du chargement de l'index: {e}")
            # Initialiser un nouvel index vide
            self.index = faiss.IndexFlatL2(self.dimension)
            self.metadata = []

    @classmethod
    def load(cls, persist_directory, embedder):
        """
        Charge un index FAISS existant.

        Args:
            persist_directory: Répertoire de persistance
            embedder: Instance de l'embedder à utiliser

        Returns:
            FAISS: Instance avec l'index chargé
        """
        instance = cls(embedder=embedder, persist_directory=persist_directory)

        # Les fichiers sont déjà chargés dans l'initialisation si présents
        return instance

    def add_texts(self, texts, metadatas=None, chunk_size=None, chunk_overlap=None):
        """
        Ajoute des textes à l'index vectoriel.

        Args:
            texts: Liste de textes à ajouter
            metadatas: Liste de métadonnées associées
            chunk_size: Taille des chunks (optional)
            chunk_overlap: Chevauchement entre chunks (optional)

        Returns:
            List[str]: IDs des documents ajoutés
        """
        if metadatas is None:
            metadatas = [{} for _ in texts]

        # Mise à jour des paramètres de chunking si spécifiés
        if chunk_size is not None or chunk_overlap is not None:
            self.text_splitter.update_splitter(chunk_size, chunk_overlap)

        added_ids = []

        # Traiter chaque texte
        for i, (text, metadata) in enumerate(zip(texts, metadatas)):
            # Chunking intelligent du texte
            chunks = self.text_splitter.split_text(text, metadata)

            # Ajouter chaque chunk à l'index
            for chunk in chunks:
                chunk_content = chunk["content"]
                chunk_metadata = chunk["metadata"]

                try:
                    # Générer l'embedding
                    embedding = self.embedder.embed(chunk_content)

                    if embedding is None or not embedding.size:
                        print(
                            f"AVERTISSEMENT: Embedding vide ou null pour le chunk {i}"
                        )
                        continue

                    # Générer un ID unique
                    doc_id = str(uuid.uuid4())

                    # Ajouter à l'index FAISS
                    self.index.add(np.array([embedding]))

                    # Stocker les métadonnées
                    self.metadata.append(
                        {
                            "id": doc_id,
                            "content": chunk_content,
                            "metadata": chunk_metadata,
                            "is_image": False,
                        }
                    )

                    added_ids.append(doc_id)

                except Exception as e:
                    print(f"Erreur lors de l'ajout du chunk {i}: {str(e)}")
                    continue

        # Persister l'index
        self._save()

        print(f"{len(added_ids)} chunks ajoutés à l'index vectoriel.")
        return added_ids

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
        added_ids = []

        # Préparer les descriptions et métadonnées
        if descriptions is None:
            descriptions = [f"Image {i + 1}" for i in range(len(images))]

        if metadatas is None:
            metadatas = [{} for _ in images]

        # Traiter chaque image
        for i, (image_path, description, metadata) in enumerate(
            zip(images, descriptions, metadatas)
        ):
            try:
                # Charger l'image
                image = Image.open(image_path).convert("RGB")

                # Générer un ID unique
                image_id = str(uuid.uuid4())
                added_ids.append(image_id)

                # Générer l'embedding
                embedding = self.embedder.embed(image)

                if embedding is None or not embedding.size:
                    print(f"AVERTISSEMENT: Embedding vide pour l'image {image_path}")
                    continue

                # Ajouter à l'index FAISS
                self.index.add(np.array([embedding]))

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

        # Persister l'index
        self._save()

        print(f"{len(added_ids)} images ajoutées à l'index vectoriel.")
        return added_ids

    def add_pdf(self, pdf_path, metadata=None, chunk_size=None, chunk_overlap=None):
        """
        Ajoute un document PDF à l'index vectoriel.

        Args:
            pdf_path: Chemin vers le fichier PDF
            metadata: Métadonnées du document
            chunk_size: Taille des chunks (optional)
            chunk_overlap: Chevauchement entre chunks (optional)

        Returns:
            List[str]: IDs des pages ajoutées
        """
        if metadata is None:
            metadata = {}

        # Mise à jour des paramètres de chunking si spécifiés
        if chunk_size is not None or chunk_overlap is not None:
            self.text_splitter.update_splitter(chunk_size, chunk_overlap)

        try:
            import PyPDF2
        except ImportError:
            print("PyPDF2 n'est pas installé. Exécutez 'pip install PyPDF2'.")
            return []

        added_ids = []

        try:
            with open(pdf_path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                total_pages = len(reader.pages)

                # Ajouter le nombre total de pages aux métadonnées
                metadata.update({"total_pages": total_pages})

                # Traiter chaque page
                for page_num, page in enumerate(reader.pages):
                    page_text = page.extract_text()

                    if not page_text.strip():
                        print(f"Page {page_num + 1} vide, ignorée.")
                        continue

                    # Chunking intelligent de la page
                    page_chunks = self.text_splitter.split_pdf_page(
                        page_text, page_num + 1, total_pages, metadata
                    )

                    # Ajouter chaque chunk de page à l'index
                    for chunk in page_chunks:
                        chunk_content = chunk["content"]
                        chunk_metadata = chunk["metadata"]

                        try:
                            # Générer l'embedding
                            embedding = self.embedder.embed(chunk_content)

                            if embedding is None or not embedding.size:
                                print(
                                    f"AVERTISSEMENT: Embedding vide pour la page {page_num + 1}"
                                )
                                continue

                            # Générer un ID unique
                            doc_id = str(uuid.uuid4())

                            # Ajouter à l'index FAISS
                            self.index.add(np.array([embedding]))

                            # Stocker les métadonnées
                            self.metadata.append(
                                {
                                    "id": doc_id,
                                    "content": chunk_content,
                                    "metadata": chunk_metadata,
                                    "is_image": False,
                                }
                            )

                            added_ids.append(doc_id)

                        except Exception as e:
                            print(
                                f"Erreur lors de l'ajout du chunk de la page {page_num + 1}: {str(e)}"
                            )
                            continue

        except Exception as e:
            print(f"Erreur lors du traitement du PDF {pdf_path}: {str(e)}")
            return []

        # Persister l'index
        self._save()

        print(f"{len(added_ids)} chunks ajoutés à partir du PDF.")
        return added_ids

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

        # Déduplication basée sur le contenu (pour éviter les chunks trop similaires)
        deduplicated_results = []
        seen_content_starts = set()

        for result in results:
            # Pour les textes, utiliser le début du chunk pour déduplication
            if not result["is_image"]:
                # Prendre les premiers 100 caractères comme signature
                content_start = result["content"][:100].strip()
                # Si déjà vu un chunk similaire, sauter
                if content_start in seen_content_starts:
                    continue
                seen_content_starts.add(content_start)

            deduplicated_results.append(result)
            # Limiter au nombre demandé
            if len(deduplicated_results) >= top_k:
                break

        return deduplicated_results
