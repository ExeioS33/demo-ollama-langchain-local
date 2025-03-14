# Makefile pour le système RAG Multimodal avec CLIP
#
# Ce Makefile fournit des commandes pour faciliter l'utilisation
# du système RAG multimodal qui permet d'analyser des textes, images
# et PDFs avec des embeddings unifiés.

# Configuration
PYTHON = python3
MODEL = qwen2.5:3b
DB_PATH = ./chroma_db
COLLECTION = documents

# Convertir le JSON en notebook Jupyter
notebook:
	$(PYTHON) -c "import json; import nbformat; f = open('multimodal_rag_demo.json'); j = json.load(f); nbf = nbformat.from_dict(j); open('multimodal_rag_demo.ipynb', 'w').write(nbformat.writes(nbf))"
	@echo "✅ Notebook Jupyter créé: multimodal_rag_demo.ipynb"

# Installation des dépendances
install:
	pip install -r requirements.txt
	@echo "✅ Dépendances installées"

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

# Réinitialiser la base de données
reset-db:
	$(PYTHON) multimodal_rag_demo.py --reset --db_path $(DB_PATH) --collection $(COLLECTION)
	@echo "✅ Base de données réinitialisée"

# Ajouter un document texte
add-text: check-ollama
	@read -p "Chemin du fichier texte: " path; \
	$(PYTHON) multimodal_rag_demo.py --add $$path --document_type text --model $(MODEL) --db_path $(DB_PATH) --collection $(COLLECTION)

# Ajouter une image
add-image: check-ollama
	@read -p "Chemin de l'image: " path; \
	read -p "Description (optionnelle): " desc; \
	if [ -z "$$desc" ]; then \
		$(PYTHON) multimodal_rag_demo.py --add $$path --document_type image --model $(MODEL) --db_path $(DB_PATH) --collection $(COLLECTION); \
	else \
		$(PYTHON) multimodal_rag_demo.py --add $$path --document_type image --description "$$desc" --model $(MODEL) --db_path $(DB_PATH) --collection $(COLLECTION); \
	fi

# Ajouter un PDF
add-pdf: check-ollama
	@read -p "Chemin du PDF: " path; \
	$(PYTHON) multimodal_rag_demo.py --add $$path --model $(MODEL) --db_path $(DB_PATH) --collection $(COLLECTION)

# Faire une requête textuelle
query: check-ollama
	@read -p "Votre question: " question; \
	$(PYTHON) multimodal_rag_demo.py --query "$$question" --model $(MODEL) --db_path $(DB_PATH) --collection $(COLLECTION)

# Faire une requête avec une image
image-query: check-ollama
	@read -p "Chemin de l'image: " path; \
	read -p "Votre question sur l'image: " question; \
	$(PYTHON) multimodal_rag_demo.py --image_query $$path --query "$$question" --model $(MODEL) --db_path $(DB_PATH) --collection $(COLLECTION)

# Obtenir des informations sur la base de données
db-info:
	@echo "Base de données: $(DB_PATH)"
	@echo "Collection: $(COLLECTION)"
	@echo "Nombre de documents: $$(ls -la $(DB_PATH)/$(COLLECTION) 2>/dev/null | wc -l)"

# Lancer le notebook Jupyter
jupyter:
	jupyter notebook multimodal_rag_demo.ipynb

# Aide
help:
	@echo "Commandes disponibles:"
	@echo "  make install             : Installer les dépendances"
	@echo "  make notebook            : Convertir multimodal_rag_demo.json en notebook Jupyter"
	@echo "  make start-ollama        : Démarrer Ollama en arrière-plan"
	@echo "  make download-models     : Télécharger les modèles nécessaires"
	@echo "  make reset-db            : Réinitialiser la base de données"
	@echo "  make add-text            : Ajouter un document texte"
	@echo "  make add-image           : Ajouter une image avec description optionnelle"
	@echo "  make add-pdf             : Ajouter un document PDF"
	@echo "  make query               : Faire une requête textuelle"
	@echo "  make image-query         : Faire une requête avec une image"
	@echo "  make db-info             : Afficher des informations sur la base de données"
	@echo "  make jupyter             : Lancer le notebook Jupyter"
	@echo ""
	@echo "Configuration:"
	@echo "  Modèle: $(MODEL)"
	@echo "  Base de données: $(DB_PATH)"
	@echo "  Collection: $(COLLECTION)"
	@echo ""
	@echo "Pour utiliser un autre modèle: make query MODEL=llava"
	@echo "Pour utiliser une autre base de données: make query DB_PATH=./autre_db"

.PHONY: install notebook check-ollama start-ollama download-models reset-db add-text add-image add-pdf query image-query db-info jupyter help 