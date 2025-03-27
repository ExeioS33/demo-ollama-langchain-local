#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Système RAG Multimodal Principal
--------------------------------
Implémentation simplifiée du pipeline RAG multimodal complet.
"""

import os
from typing import Dict, List, Union, Optional, Any
from PIL import Image
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Import des modules principaux
from core.embeddings import MultimodalEmbedder
from core.llm import LLaVA
from core.vector_operations import FAISS


class RAGSystem:
    """Système RAG principal qui orchestre le pipeline complet."""

    def __init__(
        self,
        vector_store_path: str = "data/vectors",
        use_gpu: bool = False,
        model_name: str = "llava:7b-v1.6-vicuna-q8_0",
        temperature: float = 0.1,
        similarity_threshold: float = 0.2,
    ):
        """
        Initialise le système RAG multimodal.

        Args:
            vector_store_path: Chemin vers l'index vectoriel
            use_gpu: Utiliser le GPU si disponible
            model_name: Nom du modèle LLM à utiliser
            temperature: Température pour la génération
            similarity_threshold: Seuil minimal de similarité
        """
        self.use_gpu = use_gpu and self._check_gpu_available()

        # Initialiser les composants principaux
        print(f"Initialisation du système RAG avec modèle {model_name}")

        # Composant pour les embeddings
        self.embedder = MultimodalEmbedder(use_gpu=self.use_gpu)

        # Vérifier si l'index vectoriel existe
        if os.path.exists(vector_store_path):
            print(f"Chargement de l'index vectoriel depuis {vector_store_path}")
            self.vector_db = FAISS.load(vector_store_path, self.embedder)
        else:
            print(f"Création d'un nouvel index vectoriel dans {vector_store_path}")
            self.vector_db = FAISS(
                embedder=self.embedder, persist_directory=vector_store_path
            )

        # Composant LLM
        self.llm = LLaVA(model_name=model_name, temperature=temperature)

        # Paramètres
        self.similarity_threshold = similarity_threshold

    def _check_gpu_available(self) -> bool:
        """Vérifie si un GPU est disponible."""
        try:
            import torch

            return torch.cuda.is_available()
        except ImportError:
            return False

    def query(self, query: Union[str, Image.Image], top_k: int = 3) -> Dict:
        """
        Traite une requête texte ou image et génère une réponse.

        Args:
            query: Requête textuelle ou image (chemin ou objet PIL)
            top_k: Nombre de résultats à récupérer

        Returns:
            Dict: Réponse avec contexte et sources
        """
        # 1. Déterminer le type de requête
        is_image_query = isinstance(query, Image.Image) or (
            isinstance(query, str)
            and any(
                query.lower().endswith(ext) for ext in [".jpg", ".jpeg", ".png", ".gif"]
            )
        )

        # Charger l'image si c'est un chemin
        if is_image_query and isinstance(query, str):
            try:
                query_obj = Image.open(query).convert("RGB")
                print(f"Image chargée depuis {query}")
            except Exception as e:
                print(f"Erreur lors du chargement de l'image: {e}")
                return {"answer": f"Erreur: {str(e)}", "sources": []}
        else:
            query_obj = query

        # 2. Générer l'embedding de la requête
        print("Génération de l'embedding pour la requête")
        embedding = self.embedder.embed(query_obj)

        # 3. Recherche vectorielle
        print(f"Recherche des {top_k} documents les plus pertinents")
        results = self.vector_db.search(embedding, top_k=top_k)

        # 4. Filtrer par similarité
        filtered_results = [
            r for r in results if r.get("similarity", 0) >= self.similarity_threshold
        ]

        # 5. Préparer le contexte
        context = self._prepare_context(filtered_results)

        # 6. Génération de la réponse
        print("Génération de la réponse avec le LLM")
        query_text = (
            "Décris cette image" if is_image_query and isinstance(query, str) else query
        )
        answer = self.llm.generate(
            query_text,
            context,
            is_image_query=is_image_query,
            image_path=query if is_image_query and isinstance(query, str) else None,
        )

        # 7. Formater la réponse
        return {"answer": answer, "sources": filtered_results}

    def _prepare_context(self, results: List[Dict]) -> str:
        """Prépare le contexte à partir des résultats de recherche."""
        if not results:
            return "Aucune information pertinente trouvée."

        context_pieces = []

        for i, result in enumerate(results):
            # Format différent selon le type de résultat
            if result.get("is_image", False):
                # Format pour les images
                context_pieces.append(f"[Image {i + 1}] {result['content']}")
            else:
                # Format pour le texte
                metadata = result.get("metadata", {})
                # Inclure le titre si disponible
                title = metadata.get("title", "")
                title_info = f" - {title}" if title else ""

                # Inclure la page pour les PDF
                page_info = ""
                if "page" in metadata and "total_pages" in metadata:
                    page_info = f" (Page {metadata['page']}/{metadata['total_pages']})"

                # Source du document
                source = metadata.get("source", "inconnu")

                # Formater le contexte
                context_pieces.append(
                    f"[Document {i + 1}{title_info}{page_info} - Source: {source}] {result['content']}"
                )

        return "\n\n".join(context_pieces)

    def add_document(
        self,
        document_path: str,
        description: Optional[str] = None,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ) -> List[str]:
        """
        Ajoute un document au système avec chunking intelligent.

        Args:
            document_path: Chemin vers le document
            description: Description optionnelle (pour images)
            chunk_size: Taille des chunks pour le texte (en caractères)
            chunk_overlap: Chevauchement entre chunks (en caractères)

        Returns:
            List[str]: IDs des éléments ajoutés
        """
        print(
            f"Configuration du chunking: taille={chunk_size}, chevauchement={chunk_overlap}"
        )

        # Configurer les paramètres de chunking
        if not hasattr(self.vector_db, "text_splitter"):
            print(
                "ATTENTION: Le TextSplitter n'est pas disponible dans la base vectorielle."
            )
        else:
            self.vector_db.text_splitter.chunk_size = chunk_size
            self.vector_db.text_splitter.chunk_overlap = chunk_overlap
            self.vector_db.text_splitter.update_splitter(chunk_size, chunk_overlap)
            print(f"TextSplitter configuré avec succès: {self.vector_db.text_splitter}")

        # Déterminer automatiquement le type de document
        if document_path.lower().endswith((".jpg", ".jpeg", ".png", ".gif")):
            print(f"Ajout de l'image {document_path}")
            metadata = self._prepare_image_metadata(document_path, description)
            return self.vector_db.add_images(
                [document_path], [description] if description else None, [metadata]
            )
        elif document_path.lower().endswith(".pdf"):
            print(f"Ajout du PDF {document_path}")
            metadata = self._prepare_document_metadata(document_path)
            return self.vector_db.add_pdf(
                document_path, metadata, chunk_size, chunk_overlap
            )
        else:
            # Traitement comme texte
            print(f"Ajout du document texte {document_path}")
            metadata = self._prepare_document_metadata(document_path)
            with open(document_path, "r", encoding="utf-8") as f:
                text = f.read()
            return self.vector_db.add_texts(
                [text], [metadata], chunk_size, chunk_overlap
            )

    def _prepare_document_metadata(self, document_path: str) -> Dict[str, Any]:
        """
        Prépare les métadonnées pour un document.

        Args:
            document_path: Chemin vers le document

        Returns:
            Dict: Métadonnées enrichies
        """
        filename = os.path.basename(document_path)
        file_extension = os.path.splitext(filename)[1].lower()

        # Déterminer le type de document
        doc_type = "unknown"
        if file_extension in [".txt", ".md", ".html", ".docx"]:
            doc_type = "text"
        elif file_extension == ".pdf":
            doc_type = "pdf"
        elif file_extension in [".jpg", ".jpeg", ".png", ".gif"]:
            doc_type = "image"

        # Déterminer la catégorie basée sur le chemin
        category = self._extract_category_from_path(document_path)

        return {
            "source": document_path,
            "filename": filename,
            "document_type": doc_type,
            "category": category,
            "date_added": self._get_current_date(),
        }

    def _prepare_image_metadata(
        self, image_path: str, description: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Prépare les métadonnées pour une image.

        Args:
            image_path: Chemin vers l'image
            description: Description optionnelle

        Returns:
            Dict: Métadonnées enrichies
        """
        # Obtenir les métadonnées de base
        metadata = self._prepare_document_metadata(image_path)

        # Ajouter des métadonnées spécifiques aux images
        try:
            with Image.open(image_path) as img:
                metadata.update(
                    {
                        "width": img.width,
                        "height": img.height,
                        "format": img.format,
                        "mode": img.mode,
                        "has_description": description is not None,
                    }
                )
        except Exception as e:
            print(
                f"Erreur lors de l'extraction des métadonnées de l'image {image_path}: {e}"
            )

        return metadata

    def _extract_category_from_path(self, file_path: str) -> str:
        """
        Extrait une catégorie du chemin du fichier.

        Args:
            file_path: Chemin du fichier

        Returns:
            str: Catégorie déduite du chemin
        """
        # Normaliser le chemin
        path = os.path.normpath(file_path)
        parts = path.split(os.sep)

        # Si le fichier est dans un sous-dossier, utiliser le nom du dossier comme catégorie
        if len(parts) > 1:
            # Chercher le premier dossier significatif comme catégorie
            for part in reversed(parts[:-1]):  # Ignorer le nom du fichier
                if part and part not in [
                    ".",
                    "..",
                    "data",
                    "documents",
                    "images",
                    "pdf",
                ]:
                    return part

        return "général"  # Catégorie par défaut

    def _get_current_date(self) -> str:
        """Obtient la date actuelle au format YYYY-MM-DD."""
        from datetime import datetime

        return datetime.now().strftime("%Y-%m-%d")
