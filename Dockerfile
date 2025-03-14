# Base image avec Python 3.10
FROM python:3.10-slim

# Définir le répertoire de travail
WORKDIR /app

# Installation des dépendances système nécessaires
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copie du fichier requirements et installation des dépendances
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Configuration pour éviter les problèmes de version
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Configuration pour Jupyter
RUN pip install jupyter-lab

# Exposer le port Jupyter
EXPOSE 8888

# Commande de démarrage - Jupyter Lab accessible depuis l'extérieur
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=''", "--NotebookApp.password=''"]