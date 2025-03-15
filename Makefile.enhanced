# Makefile pour le système RAG Multimodal Amélioré avec FAISS
#
# Ce Makefile fournit des commandes pour faciliter l'utilisation
# du système RAG multimodal optimisé avec FAISS et reranking
# qui permet d'analyser des textes, images et PDFs avec une
# meilleure précision et performance.

# Configuration
PYTHON = uv run python
PIP = uv pip
UV_RUN = uv run
MODEL = llava:latest
DB_PATH = ./enhanced_vector_store
COLLECTION = enhanced_multimodal_collection
CHROMA_PATH = ./chroma_db
CHROMA_COLLECTION = multimodal_collection
GPU = false
SIMILARITY = 0.2
RERANKER = cross-encoder/ms-marco-MiniLM-L-6-v2

# Vérifier si nous sommes dans un environnement virtuel
VENV_CHECK := $(shell $(PYTHON) -c "import sys; print(hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix))" 2>/dev/null)

# Vérifier si les compilateurs nécessaires sont disponibles
HAS_GCC := $(shell which g++ >/dev/null 2>&1 && echo true || echo false)
HAS_CLANG := $(shell which clang++ >/dev/null 2>&1 && echo true || echo false)

# Installation des dépendances
install:
	@echo "Installation des dépendances avec UV..."
	@echo "Vérification des outils de compilation..."
	@if [ "$(HAS_GCC)" = "false" ] && [ "$(HAS_CLANG)" = "false" ]; then \
		echo "⚠️ Aucun compilateur C++ (g++ ou clang++) n'a été trouvé."; \
		echo "  Les packages comme FAISS nécessitent un compilateur pour être installés."; \
		echo "  Vous pouvez installer les outils de compilation avec:"; \
		echo "    Ubuntu/Debian: sudo apt install build-essential"; \
		echo "    CentOS/RHEL: sudo yum groupinstall 'Development Tools'"; \
		echo "    Fedora: sudo dnf groupinstall 'Development Tools'"; \
		echo "    macOS: xcode-select --install"; \
		echo "  OU utiliser l'option d'installation sans packages nécessitant une compilation:"; \
		echo "    make install-no-compile"; \
		read -p "Essayer d'installer quand même? (y/n) " choice; \
		if [ "$$choice" != "y" ]; then \
			echo "Installation annulée."; \
			exit 1; \
		fi; \
	fi
	@echo "Installation des dépendances avec UV..."
	$(PIP) install -r requirements.txt || (echo "⚠️ Erreur lors de l'installation; essayez 'make install-no-compile'"; exit 1)
	@echo "✅ Dépendances installées"
	@echo "Vérification de transformers et CLIP..."
	@echo "import transformers; from transformers import CLIPProcessor, CLIPModel; print('✅ CLIP via transformers est correctement installé')" > check_clip.py && $(UV_RUN) check_clip.py || echo "⚠️ Problème avec l'installation de transformers/CLIP"

# Installation sans packages nécessitant une compilation
install-no-compile:
	@echo "Installation des dépendances sans packages nécessitant une compilation..."
	$(PIP) install torch transformers sentence-transformers
	$(PIP) install langchain langchain-core langchain-community langchain-ollama
	$(PIP) install --no-build-isolation faiss-cpu
	$(PIP) install pydantic pillow matplotlib pymupdf requests numpy tqdm
	$(PIP) install -r requirements.txt --no-deps
	@echo "✅ Dépendances essentielles installées (sans compilation)"

# Vérifier que faiss et transformers sont bien installés
check-deps:
	@echo "Vérification des dépendances principales..."
	@echo "import faiss; print('✅ FAISS est correctement installé (version', faiss.__version__, ')')" > check_faiss.py && $(UV_RUN) check_faiss.py || \
	(echo "❌ FAISS n'est pas installé correctement. Essayez:"; \
	 echo "$(PIP) install faiss-cpu --no-build-isolation"; exit 1)
	@echo "from transformers import CLIPProcessor, CLIPModel; print('✅ CLIP via transformers est correctement installé')" > check_clip.py && $(UV_RUN) check_clip.py || \
	(echo "❌ CLIP n'est pas installé correctement. Essayez:"; \
	 echo "$(PIP) install transformers"; exit 1)
	@echo "from sentence_transformers import CrossEncoder; print('✅ CrossEncoder est disponible pour le reranking')" > check_cross.py && $(UV_RUN) check_cross.py || \
	(echo "⚠️ CrossEncoder n'est pas disponible. Le reranking sera désactivé."; \
	 echo "Pour l'installer: $(PIP) install sentence-transformers")
	@rm -f check_faiss.py check_clip.py check_cross.py

# Vérifier si Ollama est en cours d'exécution
check-ollama:
	@curl -s http://localhost:11434/api/tags > /dev/null && echo "✅ Ollama est en cours d'exécution" || (echo "❌ Ollama n'est pas en cours d'exécution. Lancez Ollama avec 'make start-ollama'"; exit 1)

# Démarrer Ollama (en arrière-plan)
start-ollama:
	@echo "Démarrage d'Ollama..."
	@ollama serve > ollama.log 2>&1 &
	@echo "✅ Ollama démarré en arrière-plan (logs dans ollama.log)"

# Télécharger les modèles Ollama nécessaires
download-models:
	@echo "Téléchargement des modèles..."
	@ollama pull qwen2.5:3b
	@ollama pull llava
	@echo "✅ Modèles téléchargés"

# Réinitialiser la base de données FAISS
reset-db: check-deps
	$(UV_RUN) enhanced_multimodal_rag_demo.py --reset --db-path $(DB_PATH) --collection $(COLLECTION) --model $(MODEL)
	@echo "✅ Base de données FAISS réinitialisée"

# Migrer depuis une base ChromaDB existante
migrate: check-deps
	$(UV_RUN) enhanced_multimodal_rag_demo.py --migrate --chroma-path $(CHROMA_PATH) --chroma-collection $(CHROMA_COLLECTION) --db-path $(DB_PATH) --model $(MODEL) $(if $(filter true,$(GPU)),--use-gpu,)
	@echo "✅ Migration depuis ChromaDB vers FAISS terminée"

# Ajouter un document (auto-détection du type)
add-document: check-deps
	@read -p "Chemin du document (PDF, image, texte): " path; \
	read -p "Description (optionnelle, pour les images): " desc; \
	if [ -z "$$desc" ]; then \
		$(UV_RUN) enhanced_multimodal_rag_demo.py --add-document $$path --model $(MODEL) --db-path $(DB_PATH) --collection $(COLLECTION) $(if $(filter true,$(GPU)),--use-gpu,) --similarity-threshold $(SIMILARITY) --reranker $(RERANKER); \
	else \
		$(UV_RUN) enhanced_multimodal_rag_demo.py --add-document $$path --description "$$desc" --model $(MODEL) --db-path $(DB_PATH) --collection $(COLLECTION) $(if $(filter true,$(GPU)),--use-gpu,) --similarity-threshold $(SIMILARITY) --reranker $(RERANKER); \
	fi

# Faire une requête textuelle
query: check-ollama check-deps
	@read -p "Votre question: " question; \
	$(UV_RUN) enhanced_multimodal_rag_demo.py --query "$$question" --model $(MODEL) --db-path $(DB_PATH) --collection $(COLLECTION) $(if $(filter true,$(GPU)),--use-gpu,) --similarity-threshold $(SIMILARITY) --reranker $(RERANKER)

# Faire une requête avec une image
image-query: check-ollama check-deps
	@read -p "Chemin de l'image de requête: " path; \
	$(UV_RUN) enhanced_multimodal_rag_demo.py --image-query $$path --model $(MODEL) --db-path $(DB_PATH) --collection $(COLLECTION) $(if $(filter true,$(GPU)),--use-gpu,) --similarity-threshold $(SIMILARITY) --reranker $(RERANKER)

# Faire une requête sans reranking (pour comparer)
query-no-rerank: check-ollama check-deps
	@read -p "Votre question: " question; \
	$(UV_RUN) enhanced_multimodal_rag_demo.py --query "$$question" --model $(MODEL) --db-path $(DB_PATH) --collection $(COLLECTION) $(if $(filter true,$(GPU)),--use-gpu,) --similarity-threshold $(SIMILARITY) --no-reranking

# Changer le modèle LLM
set-model:
	@read -p "Nouveau modèle (ex: llava, qwen2.5:3b): " model; \
	echo "MODEL = $$model" > .model_config; \
	echo "✅ Modèle changé à: $$model"
	@echo "Utilisez 'make' avec les autres commandes pour appliquer ce changement"

# Activer/désactiver le GPU
toggle-gpu:
	@if [ "$(GPU)" = "true" ]; then \
		echo "GPU = false" > .gpu_config; \
		echo "✅ Mode GPU désactivé"; \
	else \
		echo "GPU = true" > .gpu_config; \
		echo "✅ Mode GPU activé"; \
	fi
	@echo "Utilisez 'make' avec les autres commandes pour appliquer ce changement"

# Modifier le seuil de similarité
set-similarity:
	@read -p "Nouveau seuil de similarité (0.0-1.0, recommandé: 0.2-0.4): " sim; \
	echo "SIMILARITY = $$sim" > .similarity_config; \
	echo "✅ Seuil de similarité changé à: $$sim"
	@echo "Utilisez 'make' avec les autres commandes pour appliquer ce changement"

# Changer le modèle de reranking
set-reranker:
	@read -p "Nouveau modèle de reranking: " reranker; \
	echo "RERANKER = $$reranker" > .reranker_config; \
	echo "✅ Modèle de reranking changé à: $$reranker"
	@echo "Utilisez 'make' avec les autres commandes pour appliquer ce changement"

# Obtenir des informations sur la configuration
config-info:
	@echo "Configuration actuelle:"
	@echo "  Modèle LLM: $(MODEL)"
	@echo "  Base de données FAISS: $(DB_PATH)"
	@echo "  Collection: $(COLLECTION)"
	@echo "  Utilisation GPU: $(GPU)"
	@echo "  Seuil de similarité: $(SIMILARITY)"
	@echo "  Modèle de reranking: $(RERANKER)"
	@echo ""
	@echo "Environnement:"
	@echo "  Python via UV: $(shell $(PYTHON) -c "import sys; print(sys.version.split()[0])" 2>/dev/null || echo "Non disponible")"
	@echo "  Compilateur C++: $(if $(filter true,$(HAS_GCC)),GCC disponible,$(if $(filter true,$(HAS_CLANG)),Clang disponible,Non disponible))"
	@echo "  FAISS: $(shell $(PYTHON) -c "import faiss; print(faiss.__version__)" 2>/dev/null || echo "Non installé")"
	@echo "  Transformers: $(shell $(PYTHON) -c "import transformers; print(transformers.__version__)" 2>/dev/null || echo "Non installé")"
	@echo "  Torch: $(shell $(PYTHON) -c "import torch; print(torch.__version__)" 2>/dev/null || echo "Non installé")"
	@echo "  CUDA disponible: $(shell $(PYTHON) -c "import torch; print(torch.cuda.is_available())" 2>/dev/null || echo "Non installé")"

# Compare performance (ancienne vs nouvelle implémentation)
compare-performance: check-ollama check-deps
	@read -p "Votre question pour la comparaison: " question; \
	echo "\n=== PERFORMANCE AVEC CHROMADB (ORIGINAL) ==="; \
	time $(UV_RUN) multimodal_rag_demo.py --query "$$question" --model $(MODEL) --db_path $(CHROMA_PATH) --collection $(CHROMA_COLLECTION); \
	echo "\n=== PERFORMANCE AVEC FAISS (AMÉLIORÉ) ==="; \
	time $(UV_RUN) enhanced_multimodal_rag_demo.py --query "$$question" --model $(MODEL) --db-path $(DB_PATH) --collection $(COLLECTION) $(if $(filter true,$(GPU)),--use-gpu,)

# Installer les dépendances système (pour Ubuntu/Debian)
install-system-deps-debian:
	@echo "Installation des dépendances système nécessaires (Ubuntu/Debian)..."
	sudo apt update
	sudo apt install -y build-essential python3-dev
	@echo "✅ Dépendances système installées"

# Installer les dépendances système (pour CentOS/RHEL/Fedora)
install-system-deps-redhat:
	@echo "Installation des dépendances système nécessaires (CentOS/RHEL/Fedora)..."
	sudo $(if command -v dnf >/dev/null 2>&1; then echo "dnf"; else echo "yum"; fi) groupinstall -y "Development Tools"
	sudo $(if command -v dnf >/dev/null 2>&1; then echo "dnf"; else echo "yum"; fi) install -y python3-devel
	@echo "✅ Dépendances système installées"

# Créer un script de test pour vérifier l'installation
create-test-script:
	@echo "#!/usr/bin/env python3" > test_installation.py
	@echo "# -*- coding: utf-8 -*-" >> test_installation.py
	@echo "" >> test_installation.py
	@echo "print(\"Test des importations avec UV...\")" >> test_installation.py
	@echo "try:" >> test_installation.py
	@echo "    from transformers import CLIPProcessor, CLIPModel" >> test_installation.py
	@echo "    print(\"✅ CLIP via transformers importé avec succès\")" >> test_installation.py
	@echo "except ImportError as e:" >> test_installation.py
	@echo "    print(f\"❌ Erreur d'import CLIP via transformers: {e}\")" >> test_installation.py
	@echo "try:" >> test_installation.py
	@echo "    import torch" >> test_installation.py
	@echo "    print(f\"✅ PyTorch importé avec succès (version {torch.__version__})\")" >> test_installation.py
	@echo "except ImportError as e:" >> test_installation.py
	@echo "    print(f\"❌ Erreur d'import PyTorch: {e}\")" >> test_installation.py
	@echo "try:" >> test_installation.py
	@echo "    import faiss" >> test_installation.py
	@echo "    print(f\"✅ FAISS importé avec succès (version {faiss.__version__})\")" >> test_installation.py
	@echo "except ImportError as e:" >> test_installation.py
	@echo "    print(f\"❌ Erreur d'import FAISS: {e}\")" >> test_installation.py
	@echo "print(\"\\nTest terminé avec UV.\")" >> test_installation.py
	@echo "✅ Script de test créé: test_installation.py"

# Exécuter le script de test
test-installation: create-test-script
	@echo "Exécution du test d'installation avec UV..."
	$(UV_RUN) test_installation.py

# Démo complète
demo: check-deps
	@echo "\n======= DÉMO DU SYSTÈME RAG MULTIMODAL AMÉLIORÉ =======\n"
	@echo "Configuration: Model=$(MODEL), GPU=$(GPU), Similarité=$(SIMILARITY)"
	@echo "\n1. Ajout d'un document texte de test..."
	@echo "Ceci est un document de test pour démontrer le système RAG multimodal amélioré avec FAISS et reranking." > test_document.txt
	$(UV_RUN) enhanced_multimodal_rag_demo.py --add-document test_document.txt --model $(MODEL) --db-path $(DB_PATH) --collection $(COLLECTION) $(if $(filter true,$(GPU)),--use-gpu,) --similarity-threshold $(SIMILARITY)
	
	@echo "\n2. Interrogation du système..."
	$(UV_RUN) enhanced_multimodal_rag_demo.py --query "Explique ce qu'est le système RAG?" --model $(MODEL) --db-path $(DB_PATH) --collection $(COLLECTION) $(if $(filter true,$(GPU)),--use-gpu,) --similarity-threshold $(SIMILARITY)
	
	@echo "\n3. Comparaison avec/sans reranking..."
	@echo "\n=== AVEC RERANKING ==="
	$(UV_RUN) enhanced_multimodal_rag_demo.py --query "Que démontre ce document?" --model $(MODEL) --db-path $(DB_PATH) --collection $(COLLECTION) $(if $(filter true,$(GPU)),--use-gpu,) --similarity-threshold $(SIMILARITY)
	@echo "\n=== SANS RERANKING ==="
	$(UV_RUN) enhanced_multimodal_rag_demo.py --query "Que démontre ce document?" --model $(MODEL) --db-path $(DB_PATH) --collection $(COLLECTION) $(if $(filter true,$(GPU)),--use-gpu,) --similarity-threshold $(SIMILARITY) --no-reranking
	
	@echo "\n✅ Démo terminée - Nettoyage..."
	rm test_document.txt
	@echo "✅ Fichier de test supprimé"

# Aide
help:
	@echo "Commandes disponibles pour le système RAG multimodal amélioré (avec UV):"
	@echo "  make install             : Installer les dépendances avec UV"
	@echo "  make install-no-compile  : Installer les dépendances sans compilation"
	@echo "  make check-deps          : Vérifier que les dépendances principales sont installées"
	@echo "  make test-installation   : Créer et exécuter un script de test pour vérifier l'installation"
	@echo "  make install-system-deps-debian : Installer les dépendances système (Ubuntu/Debian)"
	@echo "  make install-system-deps-redhat : Installer les dépendances système (CentOS/RHEL/Fedora)"
	@echo "  make start-ollama        : Démarrer Ollama en arrière-plan"
	@echo "  make download-models     : Télécharger les modèles nécessaires"
	@echo "  make reset-db            : Réinitialiser la base de données FAISS"
	@echo "  make migrate             : Migrer depuis une base ChromaDB existante"
	@echo "  make add-document        : Ajouter un document (PDF, image, texte)"
	@echo "  make query               : Faire une requête textuelle"
	@echo "  make image-query         : Faire une requête avec une image"
	@echo "  make query-no-rerank     : Faire une requête sans reranking"
	@echo "  make set-model           : Changer le modèle LLM"
	@echo "  make toggle-gpu          : Activer/désactiver le mode GPU"
	@echo "  make set-similarity      : Modifier le seuil de similarité"
	@echo "  make set-reranker        : Changer le modèle de reranking"
	@echo "  make config-info         : Afficher la configuration actuelle"
	@echo "  make compare-performance : Comparer les performances (original vs amélioré)"
	@echo "  make demo                : Exécuter une démonstration complète"
	@echo ""
	@echo "Configuration actuelle:"
	@echo "  Modèle LLM: $(MODEL)"
	@echo "  Base de données FAISS: $(DB_PATH)"
	@echo "  Collection: $(COLLECTION)"
	@echo "  Utilisation GPU: $(GPU)"
	@echo "  Seuil de similarité: $(SIMILARITY)"
	@echo "  Modèle de reranking: $(RERANKER)"
	@echo ""
	@echo "Vous pouvez également spécifier ces paramètres directement:"
	@echo "  make query MODEL=llava GPU=true SIMILARITY=0.3"

# Inclure les fichiers de configuration si disponibles
-include .model_config
-include .gpu_config
-include .similarity_config
-include .reranker_config

.PHONY: install install-no-compile check-deps check-ollama start-ollama download-models reset-db migrate add-document query image-query query-no-rerank set-model toggle-gpu set-similarity set-reranker config-info compare-performance install-system-deps-debian install-system-deps-redhat demo help create-test-script test-installation 