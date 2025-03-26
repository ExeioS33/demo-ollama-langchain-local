#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Système de Retrieval Augmented Generation (RAG) Multimodal
----------------------------------------------------------
Ce script implémente un système RAG qui utilise les embeddings CLIP
pour représenter à la fois les images et le texte dans le même espace vectoriel.
Cela permet de faire des requêtes textuelles qui peuvent retourner des images pertinentes
et vice versa, sans avoir à maintenir des systèmes séparés pour chaque modalité.
"""

import os
import uuid
import numpy as np
import torch
from PIL import Image
import fitz  # PyMuPDF pour traiter les PDF
import requests
from io import BytesIO
import chromadb
from chromadb.utils import embedding_functions
from typing import List, Dict, Union, Tuple, Optional, Any
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.chains import LLMChain
from transformers import CLIPProcessor, CLIPModel
import base64
from langchain_community.llms import Ollama


class MultimodalEmbedder:
    """
    Classe qui génère des embeddings pour le texte et les images en utilisant CLIP.
    """

    def __init__(self, model_name: str = "openai/clip-vit-base-patch32"):
        """
        Initialise l'embedder multimodal avec le modèle CLIP.

        Args:
            model_name (str): Nom ou chemin du modèle CLIP à utiliser
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")

        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)

        # Dimensionnalité de l'espace d'embedding
        self.embedding_dim = self.model.config.projection_dim
        print(f"Embedding dimension: {self.embedding_dim}")

    def embed_text(self, texts: List[str]) -> np.ndarray:
        """
        Génère des embeddings pour une liste de textes.

        Args:
            texts (List[str]): Liste de textes à encoder

        Returns:
            np.ndarray: Matrice d'embeddings normalisés (N x embedding_dim)
        """
        with torch.no_grad():
            inputs = self.processor(
                text=texts, return_tensors="pt", padding=True, truncation=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            text_features = self.model.get_text_features(**inputs)
            # Normalisation L2 pour la similarité cosinus
            text_embeddings = text_features / text_features.norm(dim=1, keepdim=True)

        return text_embeddings.cpu().numpy()

    def embed_image(self, images: List[Union[str, Image.Image]]) -> np.ndarray:
        """
        Génère des embeddings pour une liste d'images.

        Args:
            images (List[Union[str, Image.Image]]): Liste de chemins d'images ou objets PIL.Image

        Returns:
            np.ndarray: Matrice d'embeddings normalisés (N x embedding_dim)
        """
        processed_images = []

        for img in images:
            if isinstance(img, str):
                if img.startswith("http"):
                    # Image depuis URL
                    response = requests.get(img)
                    img = Image.open(BytesIO(response.content))
                else:
                    # Image depuis chemin local
                    img = Image.open(img)

            # S'assurer que l'image est en mode RGB (convertir si nécessaire)
            if img.mode != "RGB":
                img = img.convert("RGB")

            processed_images.append(img)

        with torch.no_grad():
            inputs = self.processor(
                images=processed_images, return_tensors="pt", padding=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            image_features = self.model.get_image_features(**inputs)
            # Normalisation L2 pour la similarité cosinus
            image_embeddings = image_features / image_features.norm(dim=1, keepdim=True)

        return image_embeddings.cpu().numpy()


class MultimodalVectorStore:
    """
    Classe qui gère le stockage et l'interrogation des embeddings multimodaux.
    """

    def __init__(
        self,
        collection_name: str = "multimodal_collection",
        persist_directory: str = "chroma_db",
    ):
        """
        Initialise le magasin de vecteurs multimodal.

        Args:
            collection_name (str): Nom de la collection ChromaDB
            persist_directory (str): Répertoire où stocker la base ChromaDB
        """
        self.embedder = MultimodalEmbedder()
        self.embedding_dim = self.embedder.embedding_dim

        # Créer un client ChromaDB persistant pour conserver les données entre les sessions
        os.makedirs(persist_directory, exist_ok=True)
        self.client = chromadb.PersistentClient(path=persist_directory)
        print(f"ChromaDB client initialized with persistence at: {persist_directory}")

        # Fonction d'embedding personnalisée pour ChromaDB
        class ClipEmbeddingFunction(embedding_functions.EmbeddingFunction):
            def __init__(self, embedder):
                self.embedder = embedder

            def __call__(self, texts):
                return self.embedder.embed_text(texts).tolist()

        # Vérifier si la collection existe déjà
        try:
            self.collection = self.client.get_collection(
                name=collection_name,
                embedding_function=ClipEmbeddingFunction(self.embedder),
            )
            collection_count = self.collection.count()
            print(
                f"Found existing collection '{collection_name}' with {collection_count} documents"
            )
        except:
            # Créer une nouvelle collection
            self.collection = self.client.create_collection(
                name=collection_name,
                embedding_function=ClipEmbeddingFunction(self.embedder),
                metadata={"hnsw:space": "cosine"},
            )
            print(f"Created new collection '{collection_name}'")

        # Pour suivre les documents qui sont des images
        self.image_ids = set()
        # Stockage de métadonnées supplémentaires
        self.metadata_store = {}

    def add_texts(
        self, texts: List[str], metadatas: Optional[List[Dict]] = None
    ) -> List[str]:
        """
        Ajoute des textes au magasin de vecteurs.

        Args:
            texts (List[str]): Liste de textes à ajouter
            metadatas (Optional[List[Dict]]): Métadonnées associées à chaque texte

        Returns:
            List[str]: Liste des identifiants générés
        """
        if not texts:
            return []

        # Générer des identifiants uniques
        ids = [str(uuid.uuid4()) for _ in range(len(texts))]

        # S'assurer que metadatas a la bonne longueur
        if metadatas is None:
            metadatas = [{"type": "text"} for _ in range(len(texts))]
        else:
            # Ajouter le type si absent
            for i in range(len(metadatas)):
                if "type" not in metadatas[i]:
                    metadatas[i]["type"] = "text"

        # Calculer les embeddings
        embeddings = self.embedder.embed_text(texts).tolist()

        # Ajouter à la collection
        self.collection.add(
            embeddings=embeddings, documents=texts, metadatas=metadatas, ids=ids
        )

        # Stocker les métadonnées supplémentaires
        for i, id in enumerate(ids):
            self.metadata_store[id] = {
                "content": texts[i],
                "is_image": False,
                **metadatas[i],
            }

        return ids

    def add_images(
        self,
        images: List[Union[str, Image.Image]],
        descriptions: Optional[List[str]] = None,
        metadatas: Optional[List[Dict]] = None,
    ) -> List[str]:
        """
        Ajoute des images au magasin de vecteurs.

        Args:
            images (List[Union[str, Image.Image]]): Liste de chemins d'images ou objets PIL.Image
            descriptions (Optional[List[str]]): Descriptions textuelles des images
            metadatas (Optional[List[Dict]]): Métadonnées associées à chaque image

        Returns:
            List[str]: Liste des identifiants générés
        """
        if not images:
            return []

        # Générer des identifiants uniques
        ids = [str(uuid.uuid4()) for _ in range(len(images))]

        # Préparer les objets Image pour l'embedding
        processed_images = []
        image_paths = []

        for i, img in enumerate(images):
            if isinstance(img, str):
                image_paths.append(img)
                if img.startswith("http"):
                    # Image depuis URL
                    response = requests.get(img)
                    processed_images.append(Image.open(BytesIO(response.content)))
                else:
                    # Image depuis chemin local
                    processed_images.append(Image.open(img))
            else:
                # C'est déjà un objet PIL.Image
                processed_images.append(img)
                image_paths.append(f"image_{i}.jpg")  # Chemin générique

        # Calculer les embeddings
        embeddings = self.embedder.embed_image(processed_images).tolist()

        # Préparation des métadonnées
        if descriptions is None:
            # Utiliser des descriptions génériques si aucune n'est fournie
            descriptions = [f"Image {i + 1}" for i in range(len(images))]

        if metadatas is None:
            metadatas = [{"type": "image", "path": path} for path in image_paths]
        else:
            # Ajouter le type et le chemin si absents
            for i, meta in enumerate(metadatas):
                if "type" not in meta:
                    meta["type"] = "image"
                if "path" not in meta:
                    meta["path"] = image_paths[i]

        # Ajouter à la collection avec les descriptions comme documents
        self.collection.add(
            embeddings=embeddings, documents=descriptions, metadatas=metadatas, ids=ids
        )

        # Marquer ces IDs comme étant des images
        self.image_ids.update(ids)

        # Stocker les métadonnées supplémentaires
        for i, id in enumerate(ids):
            self.metadata_store[id] = {
                "content": descriptions[i],
                "is_image": True,
                "path": metadatas[i].get("path", image_paths[i]),
                **metadatas[i],
            }

        return ids

    def add_pdf(
        self,
        pdf_path: str,
        extract_images: bool = True,
        page_overlap: int = 0,
        metadatas: Optional[Dict] = None,
    ) -> List[str]:
        """
        Ajoute un document PDF au magasin de vecteurs.
        Extrait le texte page par page et optionnellement les images.

        Args:
            pdf_path (str): Chemin vers le fichier PDF
            extract_images (bool): Si True, extrait aussi les images du PDF
            page_overlap (int): Nombre de lignes qui se chevauchent entre les pages
            metadatas (Optional[Dict]): Métadonnées de base à associer à tous les éléments du PDF

        Returns:
            List[str]: Liste des identifiants générés
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"Le fichier PDF {pdf_path} n'existe pas.")

        # Ouvrir le document PDF
        pdf_document = fitz.open(pdf_path)
        pdf_name = os.path.basename(pdf_path)

        # Initialiser la liste des IDs générés
        all_ids = []

        # Métadonnées de base si non fournies
        if metadatas is None:
            metadatas = {}

        # Métadonnées communes pour tous les éléments de ce PDF
        base_metadata = {"source": pdf_path, "filename": pdf_name, **metadatas}

        # Extraire le texte page par page
        for page_num, page in enumerate(pdf_document):
            text = page.get_text()
            if text.strip():
                page_metadata = {
                    **base_metadata,
                    "page": page_num + 1,
                    "type": "pdf_text",
                }

                ids = self.add_texts([text], [page_metadata])
                all_ids.extend(ids)

        # Extraire les images si demandé
        if extract_images:
            # Créer un dossier temporaire pour les images extraites
            import tempfile

            with tempfile.TemporaryDirectory() as temp_dir:
                image_paths = []
                image_descriptions = []
                image_metadatas = []

                # Extraire les images de chaque page
                for page_num, page in enumerate(pdf_document):
                    # Obtenir les images de la page
                    image_list = page.get_images(full=True)

                    for img_idx, img_info in enumerate(image_list):
                        img_idx_abs = img_info[0]  # Identifiant de l'image
                        base_img = pdf_document.extract_image(img_idx_abs)
                        image_bytes = base_img["image"]

                        # Créer un chemin temporaire pour cette image
                        temp_img_path = os.path.join(
                            temp_dir,
                            f"page{page_num + 1}_img{img_idx + 1}.{base_img['ext']}",
                        )

                        # Enregistrer l'image
                        with open(temp_img_path, "wb") as img_file:
                            img_file.write(image_bytes)

                        # Préparer les informations pour l'ajout
                        image_paths.append(temp_img_path)
                        image_descriptions.append(
                            f"Image de la page {page_num + 1} du document {pdf_name}"
                        )
                        image_metadatas.append(
                            {
                                **base_metadata,
                                "page": page_num + 1,
                                "image_index": img_idx + 1,
                                "type": "pdf_image",
                            }
                        )

                # Ajouter toutes les images extraites
                if image_paths:
                    img_ids = self.add_images(
                        image_paths, image_descriptions, image_metadatas
                    )
                    all_ids.extend(img_ids)

        # Fermer le document
        pdf_document.close()

        return all_ids

    def query(
        self,
        query: Union[str, Image.Image],
        top_k: int = 5,
        filter_metadata: Optional[Dict] = None,
    ) -> List[Dict]:
        """
        Interroge le magasin de vecteurs avec une requête textuelle ou une image.

        Args:
            query (Union[str, Image.Image]): Requête textuelle ou image
            top_k (int): Nombre de résultats à retourner
            filter_metadata (Optional[Dict]): Filtre à appliquer sur les métadonnées

        Returns:
            List[Dict]: Liste des résultats avec leurs métadonnées et scores
        """
        print(
            f"\nRecherche avec la requête: {query if isinstance(query, str) else 'image'}"
        )
        # Vérifier si la collection contient des documents
        collection_count = self.collection.count()
        if collection_count == 0:
            print(
                "Avertissement: La collection est vide. Aucun résultat ne sera retourné."
            )
            return []
        else:
            print(f"Recherche parmi {collection_count} documents dans la collection")

        # Déterminer si la requête est une image ou du texte
        if isinstance(query, Image.Image) or (
            isinstance(query, str)
            and (
                query.endswith(".jpg")
                or query.endswith(".jpeg")
                or query.endswith(".png")
                or query.endswith(".gif")
            )
        ):
            # Si c'est un chemin d'image, charger l'image
            if isinstance(query, str):
                print(f"Chargement de l'image depuis le chemin: {query}")
                query_image = Image.open(query)
            else:
                query_image = query

            # Obtenir l'embedding de l'image
            query_embedding = self.embedder.embed_image([query_image])[0].tolist()
            print("Embedding d'image généré")

        else:
            # C'est une requête textuelle
            print(f"Génération de l'embedding pour le texte: {query}")
            query_embedding = self.embedder.embed_text([query])[0].tolist()

        # Effectuer la recherche avec un nombre de résultats plus élevé
        # pour augmenter les chances de trouver du contenu pertinent
        search_k = min(top_k * 2, collection_count)  # Recherche plus large
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=search_k,
            where=filter_metadata,
            include=["documents", "metadatas", "distances"],
        )

        # Formater les résultats
        formatted_results = []
        if results["ids"] and results["ids"][0]:  # Vérifier qu'il y a des résultats
            print(f"Nombre de résultats trouvés: {len(results['ids'][0])}")

            for i, doc_id in enumerate(results["ids"][0]):
                # Obtenir le document et les métadonnées
                document = results["documents"][0][i]
                metadata = results["metadatas"][0][i]
                distance = (
                    results["distances"][0][i] if "distances" in results else None
                )

                # Calculer un score de similarité (1 - distance cosinus)
                similarity_score = 1 - distance if distance is not None else None

                # N'inclure que les résultats avec une similarité suffisante
                MIN_SIMILARITY = 0.2  # Seuil de similarité minimum
                if similarity_score and similarity_score < MIN_SIMILARITY:
                    continue

                # Enrichir avec nos métadonnées supplémentaires
                extended_metadata = self.metadata_store.get(doc_id, {})

                result = {
                    "id": doc_id,
                    "content": document,
                    "metadata": {**metadata, **extended_metadata},
                    "similarity": similarity_score,
                    "is_image": doc_id in self.image_ids,
                }
                formatted_results.append(result)

                # Afficher des informations de débogage
                print(
                    f"  Résultat {i + 1}: {'Image' if doc_id in self.image_ids else 'Texte'}"
                )
                print(f"    Similarité: {similarity_score:.4f}")
                print(
                    f"    Contenu: {document[:100]}..."
                    if len(document) > 100
                    else f"    Contenu: {document}"
                )
                if "source" in metadata:
                    print(f"    Source: {metadata['source']}")

        # Retourner les résultats les plus pertinents
        return formatted_results[:top_k]


class MultimodalRAG:
    """
    Classe qui intègre le magasin de vecteurs multimodal avec un LLM pour répondre aux requêtes.
    """

    def __init__(
        self,
        llm_name: str = "qwen2.5:3b",
        collection_name: str = "multimodal_collection",
        temperature: float = 0.2,
        max_tokens: int = 1000,
        persist_directory: str = "chroma_db",
    ):
        """
        Initialise le système RAG multimodal.

        Args:
            llm_name (str): Nom du modèle LLM à utiliser via Ollama
            collection_name (str): Nom de la collection pour le magasin de vecteurs
            temperature (float): Température pour le LLM
            max_tokens (int): Nombre maximum de tokens dans la réponse
            persist_directory (str): Répertoire où stocker la base ChromaDB
        """
        # Initialiser le magasin de vecteurs
        self.vector_store = MultimodalVectorStore(
            collection_name=collection_name, persist_directory=persist_directory
        )

        # Initialiser le LLM
        self.llm = Ollama(
            model=llm_name,
            temperature=temperature,
            num_ctx=4096,  # Contexte maximal
            num_predict=max_tokens,
        )

        # Template pour les requêtes avec contexte
        self.prompt_template = PromptTemplate(
            input_variables=["context", "query"],
            template="""
Tu es un assistant spécialisé dans l'analyse de documents multimodaux (texte et images).
Utilise le contexte fourni pour répondre à la question de l'utilisateur.
Si l'information demandée ne se trouve pas explicitement dans le contexte, indique-le clairement.
Ne fabrique pas de réponse si l'information n'est pas présente dans le contexte.

Contexte:
{context}

Question de l'utilisateur:
{query}

Réponse:
""",
        )

        # Chaîne LLM
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt_template)

    def add_document(
        self,
        document_path: str,
        document_type: str = "auto",
        description: Optional[str] = None,
        metadata: Optional[Dict] = None,
    ) -> List[str]:
        """
        Ajoute un document au système RAG.

        Args:
            document_path (str): Chemin vers le document
            document_type (str): Type de document ("pdf", "image", "text" ou "auto")
            description (Optional[str]): Description du document (pour les images)
            metadata (Optional[Dict]): Métadonnées supplémentaires

        Returns:
            List[str]: Liste des identifiants générés
        """
        # Déterminer automatiquement le type de document si 'auto'
        if document_type == "auto":
            if document_path.lower().endswith((".pdf")):
                document_type = "pdf"
            elif document_path.lower().endswith((".jpg", ".jpeg", ".png", ".gif")):
                document_type = "image"
            elif document_path.lower().endswith((".txt", ".md", ".html", ".htm")):
                document_type = "text"
            else:
                raise ValueError(
                    f"Impossible de déterminer automatiquement le type du document: {document_path}"
                )

        print(f"Ajout du document {document_path} de type {document_type}")

        # Préparer les métadonnées
        if metadata is None:
            metadata = {}

        metadata.update({"source": document_path, "document_type": document_type})

        # Ajouter le document selon son type
        if document_type == "pdf":
            return self.vector_store.add_pdf(document_path, metadatas=metadata)

        elif document_type == "image":
            descriptions = [description] if description else None
            return self.vector_store.add_images(
                [document_path], descriptions, [metadata]
            )

        elif document_type == "text":
            with open(document_path, "r", encoding="utf-8") as f:
                text = f.read()
            return self.vector_store.add_texts([text], [metadata])

        else:
            raise ValueError(f"Type de document non pris en charge: {document_type}")

    def query(
        self,
        query: Union[str, Image.Image],
        top_k: int = 5,
        filter_metadata: Optional[Dict] = None,
    ) -> Dict:
        """
        Interroge le système RAG avec une requête textuelle ou une image.

        Args:
            query (Union[str, Image.Image]): Requête textuelle ou image
            top_k (int): Nombre de résultats à récupérer pour le contexte
            filter_metadata (Optional[Dict]): Filtre à appliquer sur les métadonnées

        Returns:
            Dict: Réponse du système RAG avec les sources utilisées
        """
        # Récupérer les résultats pertinents du magasin de vecteurs
        results = self.vector_store.query(
            query, top_k=top_k, filter_metadata=filter_metadata
        )

        # Préparer le contexte pour le LLM
        context_pieces = []
        sources = []

        for i, result in enumerate(results):
            # Vérifier si le score de similarité est suffisant
            similarity = result.get("similarity", 0)
            if similarity < 0.2:  # Ignorer les résultats avec une faible similarité
                print(f"Ignoring result with low similarity: {similarity:.4f}")
                continue

            # Formater différemment selon qu'il s'agit d'une image ou d'un texte
            if result["is_image"]:
                # Pour une image, inclure sa description et son chemin
                metadata = result["metadata"]
                path = metadata.get("path", "")
                page_info = f" (page {metadata['page']})" if "page" in metadata else ""
                source_info = f" de {metadata.get('filename', metadata.get('source', 'source inconnue'))}"

                context_pieces.append(
                    f"[Image {i + 1}{page_info}{source_info}] Description: {result['content']}"
                )
                sources.append(
                    {
                        "type": "image",
                        "path": path,
                        "description": result["content"],
                        "metadata": metadata,
                        "similarity": result["similarity"],
                    }
                )
            else:
                # Pour du texte, inclure le contenu
                metadata = result["metadata"]
                content = result["content"]
                source = metadata.get("source", "source inconnue")
                page_info = f" (page {metadata['page']})" if "page" in metadata else ""

                context_pieces.append(f"[Document {i + 1}{page_info}] {content}")
                sources.append(
                    {
                        "type": "text",
                        "content": content,
                        "metadata": metadata,
                        "similarity": result["similarity"],
                    }
                )

        # S'il n'y a pas de résultats, informer le LLM
        if not context_pieces:
            context = (
                "Aucune information pertinente trouvée dans la base de connaissances."
            )
            print("⚠️ Aucun contexte pertinent trouvé pour la requête.")
        else:
            context = "\n\n".join(context_pieces)
            print(
                f"✅ {len(context_pieces)} éléments de contexte trouvés pour la requête."
            )

        # Préparer les inputs pour le LLM
        query_str = query if isinstance(query, str) else "Décris cette image en détail."

        # Formater le prompt
        formatted_prompt = self.prompt_template.format(context=context, query=query_str)

        print(f"Envoi de la requête au modèle LLM: {self.llm.model}")
        # Appeler le LLM directement
        answer = self.llm.invoke(formatted_prompt)
        print("Réponse obtenue du LLM")

        # Formater la réponse finale
        return {"answer": answer, "sources": sources}


# Exemple d'utilisation
if __name__ == "__main__":
    # Initialiser le système RAG multimodal
    rag = MultimodalRAG()

    # Ajouter des documents
    pdf_ids = rag.add_document("chemin/vers/document.pdf")
    image_ids = rag.add_document(
        "chemin/vers/image.jpg", description="Une description de l'image"
    )

    # Interroger avec du texte
    result = rag.query("Quelle information contient le document ?")
    print(f"Réponse: {result['answer']}")

    # Interroger avec une image
    result = rag.query("chemin/vers/image_requete.jpg")
    print(f"Réponse: {result['answer']}")
