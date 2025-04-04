#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
API pour le système RAG multimodal
---------------------------------
Serveur FastAPI qui expose les fonctionnalités du système RAG.
"""

import os
import tempfile
from typing import Optional, List, Dict
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# Import du système RAG
from core.rag import RAGSystem


# Initialiser FastAPI
app = FastAPI(
    title="API RAG Multimodal",
    description="API pour l'interrogation du système RAG multimodal",
    version="0.1.0",
)

# Configurer CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # A ajuster en production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialiser le système RAG
rag_system = RAGSystem(vector_store_path="data/vectors", use_gpu=False)


# Modèles de données
class QueryRequest(BaseModel):
    query: str
    top_k: Optional[int] = 3


class QueryResponse(BaseModel):
    answer: str
    sources: List[Dict]


@app.get("/")
async def root():
    """Point d'entrée de l'API."""
    return {"message": "API RAG Multimodal v0.1.0"}


@app.post("/query/text", response_model=QueryResponse)
async def text_query(request: QueryRequest):
    """
    Interroge le système RAG avec une requête textuelle.

    Args:
        request: Requête avec le texte et les paramètres

    Returns:
        QueryResponse: Réponse avec contexte
    """
    try:
        # Détection des salutations/questions générales qui ne nécessitent pas le RAG
        greetings = ["bonjour", "salut", "hello", "hi", "hey", "coucou", "salutations"]
        query_lower = request.query.lower().strip()

        # Si c'est une simple salutation sans autre contenu
        if query_lower in greetings or any(
            query_lower.startswith(g + " ") for g in greetings
        ):
            return {
                "answer": f"Bonjour ! Je suis votre assistant RH. Comment puis-je vous aider aujourd'hui ?",
                "sources": [],
            }

        result = rag_system.query(request.query, top_k=request.top_k)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query/image")
async def image_query(
    query: str = Form("Décris cette image"),
    image: UploadFile = File(...),
    top_k: int = Form(3),
):
    """
    Interroge le système RAG avec une image.

    Args:
        query: Texte de la requête (optionnel)
        image: Fichier image
        top_k: Nombre de résultats à récupérer

    Returns:
        QueryResponse: Réponse avec contexte
    """
    try:
        # Sauvegarder temporairement l'image
        suffix = os.path.splitext(image.filename)[1].lower()
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp:
            temp.write(await image.read())
            temp_path = temp.name

        # Interroger le système RAG
        result = rag_system.query(temp_path, top_k=top_k)

        # Supprimer le fichier temporaire
        os.unlink(temp_path)

        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/add/document")
async def add_document(
    document: UploadFile = File(...), description: Optional[str] = Form(None)
):
    """
    Ajoute un document au système RAG.

    Args:
        document: Fichier document (PDF, image, texte)
        description: Description du document (pour les images)

    Returns:
        dict: IDs des éléments ajoutés
    """
    try:
        # Sauvegarder temporairement le document
        suffix = os.path.splitext(document.filename)[1].lower()
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp:
            temp.write(await document.read())
            temp_path = temp.name

        # Ajouter au système RAG
        ids = rag_system.add_document(temp_path, description=description)

        # Supprimer le fichier temporaire
        os.unlink(temp_path)

        return {"message": f"{len(ids)} éléments ajoutés", "ids": ids}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Point d'entrée pour lancer l'API
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
