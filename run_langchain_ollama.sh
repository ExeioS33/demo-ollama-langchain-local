#!/bin/bash

echo "==== Nettoyage des conteneurs existants ===="
docker ps -a | grep langchain-ollama && docker stop $(docker ps -a | grep langchain-ollama | awk '{print $1}')
docker ps -a | grep langchain-ollama && docker rm $(docker ps -a | grep langchain-ollama | awk '{print $1}')

echo "==== Construction de l'image Docker ===="
docker build -t langchain-ollama .

echo "==== Vérification de l'état d'Ollama sur l'hôte ===="
ollama ps
ollama list

echo "==== Démarrage du conteneur avec accès réseau à l'hôte ===="
docker run -d --name langchain-ollama \
  --network host \
  -v "$(pwd):/app" \
  langchain-ollama

echo "==== Container en cours d'exécution ===="
docker ps

echo "==== Accédez à Jupyter Lab via: http://localhost:8888 ===="
echo "==== Pour tester, exécutez dans le conteneur: python ollama_example.py ===="
echo "docker exec -it langchain-ollama python ollama_example.py"

# Optionnel: tester directement
# docker exec -it langchain-ollama python ollama_example.py 