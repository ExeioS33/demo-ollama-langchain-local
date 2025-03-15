#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module de magasin vectoriel amélioré pour un système RAG multimodal
------------------------------------------------------------------
Ce module implémente un magasin vectoriel basé sur FAISS pour
la recherche approximative de plus proches voisins et des techniques
de reranking avancées pour améliorer la précision des résultats.
"""

import os
import json
import shutil
import pickle
import numpy as np
from typing import List, Dict, Union, Optional, Tuple, Any
from pathlib import Path
import torch
from PIL import Image
import faiss
import chromadb
from tqdm import tqdm
from sentence_transformers import CrossEncoder
from transformers import CLIPProcessor, CLIPModel

# Pour traitement des PDFs
import fitz  # PyMuPDF

# Constantes
DEFAULT_CLIP_MODEL = "ViT-B/32"


class EnhancedVectorStore:
    """
    Magasin vectoriel amélioré utilisant FAISS pour le stockage et la recherche
    de vecteurs texte et image, avec capacités de reranking avancées.
    """

    def __init__(
        self,
        collection_name: str = "enhanced_multimodal_collection",
        persist_directory: str = "enhanced_vector_store",
        use_gpu: bool = False,
        clip_model_name: str = DEFAULT_CLIP_MODEL,
        reranking_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ):
        """
        Initialise le magasin vectoriel amélioré.

        Args:
            collection_name (str): Nom de la collection
            persist_directory (str): Répertoire où stocker l'index FAISS
            use_gpu (bool): Si True, utilise GPU pour FAISS (si disponible)
            clip_model_name (str): Nom du modèle CLIP à utiliser
            reranking_model (str): Modèle de reranking à utiliser
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.use_gpu = use_gpu and torch.cuda.is_available()

        # Créer le répertoire pour stocker l'index s'il n'existe pas
        self.index_dir = os.path.join(persist_directory, collection_name)
        os.makedirs(self.index_dir, exist_ok=True)

        # Chemins vers les fichiers
        self.index_path = os.path.join(self.index_dir, "faiss_index.bin")
        self.metadata_path = os.path.join(self.index_dir, "metadata.pkl")
        self.ids_path = os.path.join(self.index_dir, "ids.pkl")

        # Chargement du modèle CLIP
        self.device = "cuda" if self.use_gpu else "cpu"
        print(f"Chargement du modèle CLIP {clip_model_name} sur {self.device}...")
        # Convertit les noms de modèles anciennes versions vers les noms de transformers
        if clip_model_name == "ViT-B/32":
            transformers_model_name = "openai/clip-vit-base-patch32"
        elif clip_model_name == "ViT-L/14":
            transformers_model_name = "openai/clip-vit-large-patch14"
        else:
            transformers_model_name = clip_model_name

        self.clip_model = CLIPModel.from_pretrained(transformers_model_name).to(
            self.device
        )
        self.clip_processor = CLIPProcessor.from_pretrained(transformers_model_name)
        self.embedding_dim = self.clip_model.config.projection_dim

        # Chargement du modèle de reranking
        if reranking_model:
            print(f"Chargement du modèle de reranking {reranking_model}...")
            self.reranker = CrossEncoder(reranking_model, device=self.device)
        else:
            self.reranker = None

        # Initialiser ou charger l'index FAISS
        if os.path.exists(self.index_path):
            self._load_index()
        else:
            self._create_index()

        # Initialiser ou charger les métadonnées
        self.metadata = []
        self.ids = []

        if os.path.exists(self.metadata_path) and os.path.exists(self.ids_path):
            self._load_metadata()
        else:
            self._save_metadata()

    def _create_index(self):
        """Crée un nouvel index FAISS."""
        # Créer l'index avec la dimension appropriée
        index = faiss.IndexFlatIP(
            self.embedding_dim
        )  # Inner product (cosine sim pour vecteurs normalisés)

        # Optionnellement, utiliser un index plus complexe pour grandes bases
        if self.use_gpu:
            # Utiliser un index IVF pour grande bases avec GPU
            nlist = max(4096, 4 * int(np.sqrt(1000)))  # Règle empirique
            quantizer = faiss.IndexFlatIP(self.embedding_dim)
            index = faiss.IndexIVFFlat(
                quantizer, self.embedding_dim, nlist, faiss.METRIC_INNER_PRODUCT
            )

            # Convertir pour GPU
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index)

            # Pour IVF, il faut entraîner avec des données
            print(
                "Création d'un index IVF vide (nécessitera un entraînement avant utilisation)"
            )
            self.needs_training = True
        else:
            # Pour CPU, utiliser HNSW pour un bon compromis vitesse/précision
            index = faiss.IndexHNSWFlat(
                self.embedding_dim, 32, faiss.METRIC_INNER_PRODUCT
            )
            self.needs_training = False

        self.index = index
        self._save_index()
        print(f"Nouvel index FAISS créé avec dimension {self.embedding_dim}")

    def _save_index(self):
        """Sauvegarde l'index FAISS."""
        # Convertir en index CPU si sur GPU
        if self.use_gpu:
            index_cpu = faiss.index_gpu_to_cpu(self.index)
            faiss.write_index(index_cpu, self.index_path)
        else:
            faiss.write_index(self.index, self.index_path)

    def _load_index(self):
        """Charge l'index FAISS depuis le disque."""
        index = faiss.read_index(self.index_path)

        if self.use_gpu:
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index)

        self.index = index
        self.needs_training = False
        print(f"Index FAISS chargé depuis {self.index_path}")

    def _save_metadata(self):
        """Sauvegarde les métadonnées et les IDs."""
        with open(self.metadata_path, "wb") as f:
            pickle.dump(self.metadata, f)

        with open(self.ids_path, "wb") as f:
            pickle.dump(self.ids, f)

    def _load_metadata(self):
        """Charge les métadonnées et les IDs depuis le disque."""
        with open(self.metadata_path, "rb") as f:
            self.metadata = pickle.load(f)

        with open(self.ids_path, "rb") as f:
            self.ids = pickle.load(f)

        print(f"Métadonnées chargées: {len(self.metadata)} éléments")

    def _get_text_embedding(self, text: str) -> np.ndarray:
        """
        Génère un embedding pour du texte avec CLIP.

        Args:
            text (str): Texte à encoder

        Returns:
            np.ndarray: Vecteur d'embedding normalisé
        """
        with torch.no_grad():
            inputs = self.clip_processor(
                text=[text], return_tensors="pt", padding=True, truncation=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            text_features = self.clip_model.get_text_features(**inputs)
            embedding = text_features / text_features.norm(dim=1, keepdim=True)

        return embedding.cpu().numpy()[0]

    def _get_image_embedding(self, image: Union[str, Image.Image]) -> np.ndarray:
        """
        Génère un embedding pour une image avec CLIP.

        Args:
            image (Union[str, Image.Image]): Chemin vers l'image ou objet PIL Image

        Returns:
            np.ndarray: Vecteur d'embedding normalisé
        """
        # Charger l'image si c'est un chemin
        if isinstance(image, str):
            try:
                image = Image.open(image).convert("RGB")
            except Exception as e:
                print(f"Erreur lors du chargement de l'image {image}: {e}")
                return None

        # Prétraiter l'image et générer l'embedding
        with torch.no_grad():
            inputs = self.clip_processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            image_features = self.clip_model.get_image_features(**inputs)
            embedding = image_features / image_features.norm(dim=1, keepdim=True)

        return embedding.cpu().numpy()[0]

    def add_texts(
        self, texts: List[str], metadatas: Optional[List[Dict]] = None
    ) -> List[str]:
        """
        Ajoute des textes à l'index.

        Args:
            texts (List[str]): Liste de textes à ajouter
            metadatas (Optional[List[Dict]]): Métadonnées pour chaque texte

        Returns:
            List[str]: Liste des IDs générés
        """
        if metadatas is None:
            metadatas = [{} for _ in texts]

        ids = []
        embeddings = []

        print(f"Génération des embeddings pour {len(texts)} textes...")
        for text, meta in zip(texts, metadatas):
            # Générer un ID unique
            text_id = f"txt_{len(self.ids)}"
            ids.append(text_id)

            # Générer l'embedding
            embedding = self._get_text_embedding(text)
            embeddings.append(embedding)

            # Préparer les métadonnées
            meta_entry = {
                "id": text_id,
                "content": text,
                "metadata": meta,
                "is_image": False,
            }
            self.metadata.append(meta_entry)
            self.ids.append(text_id)

        # Ajouter les embeddings à l'index
        embeddings_array = np.array(embeddings).astype("float32")
        self.index.add(embeddings_array)

        # Sauvegarder
        self._save_index()
        self._save_metadata()

        return ids

    def add_images(
        self,
        images: List[Union[str, Image.Image]],
        descriptions: Optional[List[str]] = None,
        metadatas: Optional[List[Dict]] = None,
    ) -> List[str]:
        """
        Ajoute des images à l'index.

        Args:
            images (List[Union[str, Image.Image]]): Liste de chemins d'images ou objets PIL
            descriptions (Optional[List[str]]): Descriptions pour chaque image
            metadatas (Optional[List[Dict]]): Métadonnées pour chaque image

        Returns:
            List[str]: Liste des IDs générés
        """
        if descriptions is None:
            descriptions = ["" for _ in images]

        if metadatas is None:
            metadatas = [{} for _ in images]

        ids = []
        embeddings = []

        print(f"Génération des embeddings pour {len(images)} images...")
        for image, description, meta in zip(images, descriptions, metadatas):
            # Générer un ID unique
            image_id = f"img_{len(self.ids)}"
            ids.append(image_id)

            # Générer l'embedding
            embedding = self._get_image_embedding(image)

            if embedding is None:
                print(f"Échec pour l'image {image}, ignorée")
                continue

            embeddings.append(embedding)

            # Préparer les métadonnées
            if isinstance(image, str):
                # Stocker le chemin dans les métadonnées
                meta["path"] = image

            meta_entry = {
                "id": image_id,
                "content": description,
                "metadata": meta,
                "is_image": True,
            }
            self.metadata.append(meta_entry)
            self.ids.append(image_id)

        # Ajouter les embeddings à l'index
        if embeddings:
            embeddings_array = np.array(embeddings).astype("float32")
            self.index.add(embeddings_array)

            # Sauvegarder
            self._save_index()
            self._save_metadata()

        return ids

    def add_pdf(self, pdf_path: str, metadatas: Optional[Dict] = None) -> List[str]:
        """
        Ajoute un document PDF au magasin de vecteurs en extrayant texte et images.

        Args:
            pdf_path (str): Chemin vers le document PDF
            metadatas (Optional[Dict]): Métadonnées de base pour tous les éléments du PDF

        Returns:
            List[str]: Liste des IDs générés
        """
        if metadatas is None:
            metadatas = {}

        # Vérifier que le fichier existe
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"Le fichier PDF {pdf_path} n'existe pas")

        pdf_filename = os.path.basename(pdf_path)
        print(f"Traitement du PDF: {pdf_filename}")

        # Base de métadonnées commune
        base_meta = {"source": pdf_path, "filename": pdf_filename, **metadatas}

        all_ids = []

        # Ouvrir le document PDF
        doc = fitz.open(pdf_path)

        # Créer un répertoire temporaire pour les images
        temp_img_dir = os.path.join(self.persist_directory, "temp_images")
        os.makedirs(temp_img_dir, exist_ok=True)

        # Traiter chaque page
        for page_num, page in enumerate(tqdm(doc, desc="Pages")):
            # Extraire le texte
            text = page.get_text()

            if text.strip():  # Ne pas ajouter les pages vides
                # Ajouter le texte avec métadonnées spécifiques à la page
                page_meta = {**base_meta, "page": page_num + 1, "type": "text"}
                text_ids = self.add_texts([text], [page_meta])
                all_ids.extend(text_ids)

            # Extraire les images
            image_list = page.get_images(full=True)

            for img_index, img_info in enumerate(image_list):
                xref = img_info[0]  # ID interne

                try:
                    # Extraire l'image
                    base_img = doc.extract_image(xref)
                    img_bytes = base_img["image"]

                    # Sauvegarder temporairement l'image
                    img_ext = base_img["ext"]
                    img_filename = f"page{page_num + 1}_img{img_index + 1}.{img_ext}"
                    img_path = os.path.join(temp_img_dir, img_filename)

                    with open(img_path, "wb") as img_file:
                        img_file.write(img_bytes)

                    # Ajouter l'image avec métadonnées spécifiques
                    img_meta = {
                        **base_meta,
                        "page": page_num + 1,
                        "image_index": img_index,
                        "type": "image",
                    }

                    # Description automatique simple
                    description = f"Image {img_index + 1} de la page {page_num + 1} du document {pdf_filename}"

                    img_ids = self.add_images([img_path], [description], [img_meta])
                    all_ids.extend(img_ids)

                except Exception as e:
                    print(
                        f"Erreur lors de l'extraction de l'image {img_index} de la page {page_num + 1}: {e}"
                    )

        # Fermer le document
        doc.close()

        # Nettoyer les images temporaires
        shutil.rmtree(temp_img_dir, ignore_errors=True)

        return all_ids

    def query(
        self,
        query: Union[str, Image.Image],
        top_k: int = 5,
        filter_metadata: Optional[Dict] = None,
        use_reranking: bool = True,
    ) -> List[Dict]:
        """
        Interroge l'index avec une requête textuelle ou une image.

        Args:
            query (Union[str, Image.Image]): Requête textuelle ou image
            top_k (int): Nombre maximum de résultats à retourner
            filter_metadata (Optional[Dict]): Filtre à appliquer sur les métadonnées
            use_reranking (bool): Si True, utilise le reranking pour améliorer les résultats

        Returns:
            List[Dict]: Liste de résultats avec contenu, métadonnées et scores
        """
        # Vérifier que l'index n'est pas vide
        if len(self.ids) == 0:
            print("L'index est vide, aucun résultat disponible")
            return []

        # Augmenter top_k pour le reranking
        faiss_top_k = top_k * 3 if use_reranking and self.reranker else top_k

        # Générer l'embedding de la requête
        if isinstance(query, str):
            print(f"Recherche textuelle: {query}")
            query_embedding = self._get_text_embedding(query)
        else:
            print("Recherche par image")
            query_embedding = self._get_image_embedding(query)

        if query_embedding is None:
            print("Impossible de générer un embedding pour la requête")
            return []

        # Préparer l'embedding pour la recherche
        query_embedding = np.array([query_embedding]).astype("float32")

        # Effectuer la recherche dans FAISS
        distances, indices = self.index.search(
            query_embedding, min(faiss_top_k, len(self.ids))
        )

        # Convertir distances à scores de similarité (1 - distance normalisée)
        similarities = distances[0]  # Pour cosine sim, c'est directement la similarité
        indices = indices[0]

        # Préparer les résultats
        results = []

        for idx, similarity in zip(indices, similarities):
            # Obtenir les métadonnées
            meta_entry = self.metadata[idx]

            # Appliquer le filtre des métadonnées si spécifié
            if filter_metadata:
                skip = False
                for key, value in filter_metadata.items():
                    if (
                        key not in meta_entry["metadata"]
                        or meta_entry["metadata"][key] != value
                    ):
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

        # Appliquer le reranking si demandé et disponible
        if use_reranking and self.reranker and len(results) > 1:
            print("Application du reranking...")
            pairs = []

            # Préparer les paires (requête, contenu) pour le reranker
            query_text = query if isinstance(query, str) else "image query"

            for result in results:
                pairs.append([query_text, result["content"]])

            # Obtenir les scores de reranking
            rerank_scores = self.reranker.predict(pairs)

            # Appliquer les scores
            for i, score in enumerate(rerank_scores):
                results[i]["rerank_score"] = float(score)

            # Réordonner les résultats par score de reranking
            results = sorted(results, key=lambda x: x["rerank_score"], reverse=True)

        # Retourner les résultats limités à top_k
        return results[:top_k]

    def reset(self):
        """Réinitialise l'index et les métadonnées."""
        # Supprimer l'index et les métadonnées
        if os.path.exists(self.index_path):
            os.remove(self.index_path)
        if os.path.exists(self.metadata_path):
            os.remove(self.metadata_path)
        if os.path.exists(self.ids_path):
            os.remove(self.ids_path)

        # Recréer
        self._create_index()
        self.metadata = []
        self.ids = []
        self._save_metadata()

        print("Index et métadonnées réinitialisés")


def convert_chromadb_to_faiss(
    chroma_collection_name: str,
    chroma_persist_directory: str,
    output_directory: str = "enhanced_vector_store",
    use_gpu: bool = False,
    clip_model_name: str = DEFAULT_CLIP_MODEL,
    reranking_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
) -> Optional[EnhancedVectorStore]:
    """
    Convertit une collection ChromaDB existante en index FAISS.

    Args:
        chroma_collection_name (str): Nom de la collection ChromaDB à convertir
        chroma_persist_directory (str): Répertoire de persistance ChromaDB
        output_directory (str): Répertoire où stocker l'index FAISS
        use_gpu (bool): Si True, utilise GPU pour FAISS si disponible
        clip_model_name (str): Nom du modèle CLIP à utiliser
        reranking_model (str): Modèle de reranking à utiliser

    Returns:
        Optional[EnhancedVectorStore]: Le nouveau magasin de vecteurs ou None en cas d'échec
    """
    print(f"Démarrage de la conversion de ChromaDB vers FAISS...")

    try:
        # Vérifier que le répertoire ChromaDB existe
        if not os.path.exists(chroma_persist_directory):
            print(f"Répertoire ChromaDB {chroma_persist_directory} introuvable")
            return None

        # Charger la collection ChromaDB
        client = chromadb.PersistentClient(path=chroma_persist_directory)

        try:
            collection = client.get_collection(name=chroma_collection_name)
        except ValueError:
            print(f"Collection ChromaDB {chroma_collection_name} introuvable")
            return None

        # Récupérer toutes les données
        chroma_data = collection.get(include=["embeddings", "documents", "metadatas"])

        # Vérifier qu'il y a des données
        if not chroma_data["ids"]:
            print("La collection ChromaDB est vide")
            return None

        # Initialiser le nouvel EnhancedVectorStore
        new_vs = EnhancedVectorStore(
            collection_name=chroma_collection_name,
            persist_directory=output_directory,
            use_gpu=use_gpu,
            clip_model_name=clip_model_name,
            reranking_model=reranking_model,
        )

        # Réinitialiser le nouvel index (au cas où)
        new_vs.reset()

        # Traiter chaque élément pour l'ajouter au nouveau magasin
        print(f"Conversion de {len(chroma_data['ids'])} éléments...")

        text_entries = []
        text_metadatas = []
        image_paths = []
        image_descriptions = []
        image_metadatas = []

        for i, (doc_id, embedding, document, metadata) in enumerate(
            zip(
                chroma_data["ids"],
                chroma_data["embeddings"],
                chroma_data["documents"],
                chroma_data["metadatas"],
            )
        ):
            # Déterminer si c'est une image ou du texte basé sur les métadonnées
            is_image = (
                metadata.get("document_type") == "image"
                or metadata.get("type") == "image"
            )

            if is_image:
                # Pour les images, on a besoin du chemin
                path = metadata.get("path") or metadata.get("source")

                if not path or not os.path.exists(path):
                    print(f"⚠️ Chemin d'image invalide pour l'ID {doc_id}, ignoré")
                    continue

                image_paths.append(path)
                image_descriptions.append(document)  # La description de l'image
                image_metadatas.append(metadata)
            else:
                # Pour le texte, ajouter à la liste
                text_entries.append(document)
                text_metadatas.append(metadata)

        # Ajouter les textes et images au nouveau magasin
        if text_entries:
            print(f"Ajout de {len(text_entries)} entrées textuelles...")
            new_vs.add_texts(text_entries, text_metadatas)

        if image_paths:
            print(f"Ajout de {len(image_paths)} entrées d'images...")
            new_vs.add_images(image_paths, image_descriptions, image_metadatas)

        print(
            f"Conversion terminée avec succès! {len(new_vs.ids)} éléments dans le nouvel index."
        )
        return new_vs

    except Exception as e:
        print(f"Erreur lors de la conversion: {e}")
        import traceback

        traceback.print_exc()
        return None


# Exemple d'utilisation directe
if __name__ == "__main__":
    # Créer un magasin de vecteurs
    vs = EnhancedVectorStore()

    # Ajouter quelques textes
    text_ids = vs.add_texts(
        ["Ceci est un exemple de texte.", "Voici un autre texte d'exemple."]
    )

    # Ajouter une image
    # image_ids = vs.add_images(["chemin/vers/image.jpg"], ["Une description de l'image"])

    # Rechercher
    results = vs.query("exemple de texte")

    for result in results:
        print(f"Score: {result['similarity']:.4f}, Contenu: {result['content']}")
