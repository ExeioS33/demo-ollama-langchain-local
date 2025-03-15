#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Système de Retrieval Augmented Generation (RAG) Multimodal Amélioré
-------------------------------------------------------------------
Ce module implémente une version améliorée du système RAG multimodal
en utilisant FAISS pour la recherche approximative de plus proches voisins
et des techniques de reranking avancées pour améliorer la précision des résultats.
"""

import os
from typing import List, Dict, Union, Optional, Any
from PIL import Image
import torch
import numpy as np
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.llms import Ollama

# Importer le magasin de vecteurs amélioré
from enhanced_vector_store import EnhancedVectorStore, convert_chromadb_to_faiss


class EnhancedMultimodalRAG:
    """
    Classe améliorée qui intègre le magasin de vecteurs amélioré avec un LLM
    pour répondre aux requêtes avec une meilleure précision des résultats.
    """

    def __init__(
        self,
        llm_name: str = "qwen2.5:3b",
        collection_name: str = "enhanced_multimodal_collection",
        temperature: float = 0.2,
        max_tokens: int = 1000,
        persist_directory: str = "enhanced_vector_store",
        use_gpu: bool = False,
        reranking_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        similarity_threshold: float = 0.2,
        convert_from_chromadb: Optional[Dict] = None,
    ):
        """
        Initialise le système RAG multimodal amélioré.

        Args:
            llm_name (str): Nom du modèle LLM à utiliser via Ollama
            collection_name (str): Nom de la collection pour le magasin de vecteurs
            temperature (float): Température pour le LLM
            max_tokens (int): Nombre maximum de tokens dans la réponse
            persist_directory (str): Répertoire où stocker l'index FAISS
            use_gpu (bool): Si True, utilise GPU pour FAISS (si disponible)
            reranking_model (str): Modèle de reranking à utiliser
            similarity_threshold (float): Seuil minimal de similarité pour inclure un résultat
            convert_from_chromadb (Optional[Dict]): Si spécifié, convertit une collection ChromaDB existante
                                                  doit contenir 'name' et 'directory'
        """
        # Convertir une collection ChromaDB existante si demandé
        if convert_from_chromadb:
            print(
                f"Conversion de la collection ChromaDB {convert_from_chromadb['name']} vers FAISS..."
            )
            self.vector_store = convert_chromadb_to_faiss(
                chroma_collection_name=convert_from_chromadb["name"],
                chroma_persist_directory=convert_from_chromadb["directory"],
                output_directory=persist_directory,
                use_gpu=use_gpu,
            )
            if not self.vector_store:
                print("Erreur lors de la conversion. Création d'un nouvel index...")
                self.vector_store = EnhancedVectorStore(
                    collection_name=collection_name,
                    persist_directory=persist_directory,
                    use_gpu=use_gpu,
                    reranking_model=reranking_model,
                )
        else:
            # Initialiser le magasin de vecteurs amélioré
            self.vector_store = EnhancedVectorStore(
                collection_name=collection_name,
                persist_directory=persist_directory,
                use_gpu=use_gpu,
                reranking_model=reranking_model,
            )

        # Seuil de similarité
        self.similarity_threshold = similarity_threshold

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
Organise ta réponse de manière claire et structurée.

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
        use_reranking: bool = True,
    ) -> Dict:
        """
        Interroge le système RAG avec une requête textuelle ou une image.
        Utilise FAISS pour la recherche approximative et optionnellement
        un reranker pour améliorer la précision.

        Args:
            query (Union[str, Image.Image]): Requête textuelle ou image
            top_k (int): Nombre de résultats à récupérer pour le contexte
            filter_metadata (Optional[Dict]): Filtre à appliquer sur les métadonnées
            use_reranking (bool): Si True, utilise le reranking avancé

        Returns:
            Dict: Réponse du système RAG avec les sources utilisées
        """
        # Récupérer les résultats pertinents du magasin de vecteurs
        results = self.vector_store.query(
            query,
            top_k=top_k,
            filter_metadata=filter_metadata,
            use_reranking=use_reranking,
        )

        # Filtrer les résultats par score de similarité
        results = [
            r for r in results if r.get("similarity", 0) >= self.similarity_threshold
        ]

        if use_reranking and "rerank_score" in results[0]:
            print(f"Résultats reordonnés en utilisant le reranker")
            # Déjà classés par le reranker dans la méthode query() du magasin vectoriel
        else:
            # Classer par similarité décroissante
            results = sorted(
                results, key=lambda x: x.get("similarity", 0), reverse=True
            )

        # Préparer le contexte pour le LLM
        context_pieces = []
        sources = []

        for i, result in enumerate(results):
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
                        "rerank_score": result.get("rerank_score", None),
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
                        "rerank_score": result.get("rerank_score", None),
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


# Fonction pour migrer d'une base existante
def migrate_from_original_rag(
    original_collection_name: str,
    original_persist_directory: str,
    new_persist_directory: str,
    llm_name: str = "qwen2.5:3b",
):
    """
    Migre une base ChromaDB existante vers le nouveau système RAG amélioré.

    Args:
        original_collection_name (str): Nom de la collection originale
        original_persist_directory (str): Répertoire de la base ChromaDB originale
        new_persist_directory (str): Répertoire où stocker la nouvelle base FAISS
        llm_name (str): Nom du modèle LLM à utiliser

    Returns:
        EnhancedMultimodalRAG: Instance du système RAG amélioré avec les données migrées
    """
    # Créer le système RAG amélioré avec conversion de la base existante
    enhanced_rag = EnhancedMultimodalRAG(
        llm_name=llm_name,
        collection_name=original_collection_name,
        persist_directory=new_persist_directory,
        convert_from_chromadb={
            "name": original_collection_name,
            "directory": original_persist_directory,
        },
    )

    return enhanced_rag


# Exemple d'utilisation
if __name__ == "__main__":
    # Initialiser le système RAG multimodal amélioré
    rag = EnhancedMultimodalRAG()

    # Option pour migrer d'une base existante
    # rag = migrate_from_original_rag(
    #     original_collection_name="multimodal_collection",
    #     original_persist_directory="chroma_db",
    #     new_persist_directory="enhanced_vector_store"
    # )

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
