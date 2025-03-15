#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Test des importations avec uv run
"""

print("Test des importations avec uv...")

try:
    from transformers import CLIPProcessor, CLIPModel

    print("✅ CLIP via transformers importé avec succès")
except ImportError as e:
    print(f"❌ Erreur d'import CLIP via transformers: {e}")

try:
    import torch

    print(f"✅ PyTorch importé avec succès (version {torch.__version__})")
except ImportError as e:
    print(f"❌ Erreur d'import PyTorch: {e}")

try:
    import faiss

    print(f"✅ FAISS importé avec succès (version {faiss.__version__})")
except ImportError as e:
    print(f"❌ Erreur d'import FAISS: {e}")

try:
    from sentence_transformers import CrossEncoder

    print("✅ CrossEncoder importé avec succès")
except ImportError as e:
    print(f"❌ Erreur d'import CrossEncoder: {e}")

print("\nTest terminé avec uv.")
