# LangChain with Ollama Project

This repository contains a Python project for working with LangChain and Ollama, using UV as the package manager. This README provides detailed instructions on setting up and running this project.

## Introduction

This project allows you to interact with large language models (LLMs) using LangChain combined with Ollama, which provides local access to open-source models like Llama2. 

## Prerequisites

Before you begin, make sure you have the following installed:

1. **Linux** (This project was tested on Ubuntu with Linux kernel 5.15.0)
2. **Zsh** or Bash shell
3. **UV Package Manager**: Follow the [official installation guide](https://github.com/astral-sh/uv) if you don't have it installed:
   ```bash
   curl -fsSL https://astral.sh/uv/install.sh | bash
   ```
4. **Ollama**: Install and run Ollama by following the [official guide](https://ollama.ai/download):
   ```bash
   curl -fsSL https://ollama.ai/install.sh | bash
   ```

## Project Setup

### 1. Clone the Repository

```bash
git clone <repository-url>
cd langchain-course
```

### 2. Setup Python 3.12 with UV

UV can manage Python versions for you. Install Python 3.12 and set it as the default for this project:

```bash
# Install Python 3.12
uv python install 3.12 --preview

# Pin Python 3.12 as the default for this project
uv python pin 3.12
```

### 3. Create a Virtual Environment

Create and activate a virtual environment using UV:

```bash
# Create a virtual environment using Python 3.12
uv venv

# Activate the virtual environment
source .venv/bin/activate
```

### 4. Install Dependencies

Install the required dependencies using UV:

```bash
# Install dependencies from requirements.txt
uv pip install -r requirements.txt
```

The project's dependencies include:
- `langchain` - Framework for LLM applications
- `langchain-core` - Core abstractions for LangChain
- `langchain-community` - Community-maintained integrations
- `langchain-ollama` - Integration with Ollama
- `gpt4all` - Local embeddings model (used for GPT4AllEmbeddings)
- `chromadb` - Vector database for storing embeddings
- `sentence-transformers` - For Hugging Face embeddings (preferred over GPT4All due to compatibility)
- `pillow` - Python Imaging Library for image processing
- `matplotlib` - For displaying images
- Other utility packages like `pydantic`, `numpy`, `pandas`, etc.

### 5. Install Ollama Models

Before running the application, make sure to pull the necessary models using Ollama:

```bash
# Pull the Llama2 model for text generation
ollama pull llama2:7b-chat-q4_0

# Pull the Llava model for vision tasks
ollama pull llava
```

## Running the Application

To run the example script:

```bash
# Make sure your virtual environment is activated
uv run example.py
```

For image analysis using the Llava vision model:

```bash
# Analyze an image
uv run image_query_llava.py path/to/image.jpg "What can you see in this image?"
```

Or explore the provided Jupyter notebook:

```bash
jupyter notebook image_analysis_demo.ipynb
```

## Project Structure

- `example.py` - Simple example script demonstrating LangChain with Ollama
- `image_query_llava.py` - Module for querying the Llava vision model with images
- `image_analysis_demo.ipynb` - Jupyter notebook showcasing vision model capabilities
- `requirements.txt` - Project dependencies
- `pyproject.toml` - Project metadata and configuration

## Features

### Text-based LLM Querying

The project includes examples of using Ollama-based LLMs for text generation via LangChain.

### Image Analysis with Vision Language Models

The `image_query_llava.py` module provides a robust interface for interacting with the Llava vision language model:

- **Query images with natural language questions**
- **Process both local images and remote images from URLs**
- **Batch process multiple questions for the same image**
- **Handle both file paths and PIL Image objects**

Example usage:

```python
from image_query_llava import analyze_image

# Quick analysis
result = analyze_image("path/to/image.jpg", "What's in this image?")
print(result)

# For more advanced usage
from image_query_llava import ImageQuery

# Initialize the image querier
querier = ImageQuery(model="llava")

# Ask a question about an image
response = querier.query_image("path/to/image.jpg", "Describe this image in detail.")
print(response)

# Ask multiple questions about the same image
questions = [
    "What objects can you see?",
    "What colors are dominant?",
    "Is there any text visible?"
]
responses = querier.batch_query("path/to/image.jpg", questions)
for q, r in zip(questions, responses):
    print(f"Q: {q}\nA: {r}\n")
```

## Troubleshooting

### Common Issues:

1. **Python Interpreter Issues**: If you encounter any issues with the Python interpreter, use `uv run` instead of calling Python directly:
   ```bash
   uv run your_script.py
   ```

2. **Dependency Conflicts**: If you encounter dependency conflicts, you may need to adjust version constraints in `requirements.txt`.

3. **Ollama Connection Issues**: Make sure Ollama is running in the background:
   ```bash
   ollama serve
   ```

4. **Model Not Found**: If the model is not found, make sure to pull it first:
   ```bash
   ollama pull llama2:7b-chat-q4_0
   ollama pull llava
   ```

5. **Missing Embeddings Libraries**: If you encounter errors about missing embeddings libraries like `gpt4all`, make sure they are installed:
   ```bash
   uv pip install gpt4all chromadb
   ```

6. **GLIBC Version Issues**: If you encounter errors with GLIBC version compatibility when using GPT4All embeddings, use sentence-transformers instead as a more compatible alternative:
   ```bash
   uv pip install sentence-transformers
   ```
   And then replace your embeddings code with HuggingFaceEmbeddings (see the embedding examples section).

7. **Image Processing Issues**: For issues with image processing, ensure you have the required libraries:
   ```bash
   uv pip install pillow matplotlib requests
   ```

## Embedding Examples

### Using Sentence Transformers (Hugging Face)
```python
from langchain_community.embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma.from_documents(documents=all_splits, embedding=embeddings)
```

### Using Ollama Embeddings
```python
from langchain_community.embeddings import OllamaEmbeddings

embeddings = OllamaEmbeddings(model="nomic-embed-text")
vectorstore = Chroma.from_documents(documents=all_splits, embedding=embeddings)
```

## Additional Resources

- [UV Documentation](https://docs.astral.sh/uv/)
- [LangChain Documentation](https://python.langchain.com/docs/get_started/introduction)
- [Ollama Documentation](https://ollama.ai/docs)
- [GPT4All Documentation](https://docs.gpt4all.io/)
- [ChromaDB Documentation](https://docs.trychroma.com/)
- [Sentence Transformers Documentation](https://www.sbert.net/)
- [Llava Model Information](https://ollama.com/library/llava)

## License

This project is licensed under the terms of the MIT license. 