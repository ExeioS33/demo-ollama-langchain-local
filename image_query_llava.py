#!/usr/bin/env python3
"""
Image Query with Llava Vision Language Model

This module provides functionality to query a local Ollama instance running the llava
vision language model with questions about images.
"""

import os
import base64
import json
from typing import Union, List, Optional
from pathlib import Path
import requests
from PIL import Image
import io


class ImageQuery:
    """
    A class for querying the Llava vision language model with images.

    This class provides methods to send images along with text queries to
    the llava model running on a local Ollama instance.
    """

    def __init__(self, base_url: str = "http://localhost:11434", model: str = "llava"):
        """
        Initialize the ImageQuery class.

        Args:
            base_url: URL of the Ollama API. Defaults to "http://localhost:11434".
            model: Name of the model to use. Defaults to "llava".
        """
        self.base_url = base_url
        self.model = model
        self.api_url = f"{self.base_url}/api/generate"

        # Check if the model is available
        self._check_model_available()

    def _check_model_available(self) -> None:
        """
        Check if the specified model is available in Ollama.
        If not, provide instructions to pull it.
        """
        try:
            response = requests.get(f"{self.base_url}/api/tags")
            if response.status_code == 200:
                models = response.json().get("models", [])
                model_names = [model["name"] for model in models]
                if self.model not in model_names:
                    print(
                        f"Model '{self.model}' not found in Ollama. Pulling it now..."
                    )
                    self._pull_model()
            else:
                print(f"Error connecting to Ollama API: {response.status_code}")
                print("Make sure Ollama is running with 'ollama serve'")
        except requests.exceptions.ConnectionError:
            print(
                "Could not connect to Ollama. Make sure it's running with 'ollama serve'"
            )

    def _pull_model(self) -> None:
        """Pull the model from Ollama if it's not available."""
        try:
            os.system(f"ollama pull {self.model}")
            print(f"Model '{self.model}' has been pulled.")
        except Exception as e:
            print(f"Error pulling model: {e}")
            print("You may need to manually run: ollama pull llava")

    def _encode_image(self, image_path: Union[str, Path]) -> str:
        """
        Encode an image to base64.

        Args:
            image_path: Path to the image file.

        Returns:
            Base64 encoded string of the image.
        """
        if isinstance(image_path, str):
            image_path = Path(image_path)

        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")

        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode("utf-8")

    def _encode_pil_image(self, image: Image.Image, format: str = "PNG") -> str:
        """
        Encode a PIL Image to base64.

        Args:
            image: PIL Image object.
            format: Image format for saving. Defaults to "PNG".

        Returns:
            Base64 encoded string of the image.
        """
        buffered = io.BytesIO()
        image.save(buffered, format=format)
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    def query_image(
        self,
        image: Union[str, Path, Image.Image],
        prompt: str,
        temperature: float = 0.7,
        stream: bool = False,
    ) -> str:
        """
        Query the model with an image and prompt.

        Args:
            image: Path to the image file or PIL Image object.
            prompt: Text prompt to send with the image.
            temperature: Sampling temperature. Higher values make output more random.
            stream: Whether to stream the response.

        Returns:
            The model's response as a string.
        """
        # Encode the image
        if isinstance(image, (str, Path)):
            image_b64 = self._encode_image(image)
        elif isinstance(image, Image.Image):
            image_b64 = self._encode_pil_image(image)
        else:
            raise TypeError("Image must be a string path, Path object, or PIL Image")

        # Prepare the payload
        payload = {
            "model": self.model,
            "prompt": prompt,
            "images": [image_b64],
            "stream": stream,
            "temperature": temperature,
        }

        # Make the API request
        try:
            response = requests.post(self.api_url, json=payload)
            if response.status_code == 200:
                if stream:
                    # Return a generator for streamed responses
                    return self._process_stream(response)
                else:
                    # Return the complete response
                    return response.json().get("response", "")
            else:
                error_msg = f"Error: {response.status_code} - {response.text}"
                print(error_msg)
                return error_msg
        except Exception as e:
            error_msg = f"Exception occurred: {str(e)}"
            print(error_msg)
            return error_msg

    def _process_stream(self, response):
        """Process a streaming response from the API."""
        for line in response.iter_lines():
            if line:
                data = line.decode("utf-8")
                if data.startswith("data:"):
                    json_str = data[5:].strip()
                    if json_str:
                        try:
                            chunk = json.loads(json_str)
                            if "response" in chunk:
                                yield chunk["response"]
                        except json.JSONDecodeError:
                            continue

    def batch_query(
        self,
        image: Union[str, Path, Image.Image],
        prompts: List[str],
        temperature: float = 0.7,
    ) -> List[str]:
        """
        Send multiple queries for the same image.

        Args:
            image: Path to the image file or PIL Image object.
            prompts: List of text prompts to send with the image.
            temperature: Sampling temperature.

        Returns:
            List of model responses.
        """
        return [self.query_image(image, prompt, temperature) for prompt in prompts]


def analyze_image(
    image_path: Union[str, Path],
    question: str,
    model: str = "llava",
    base_url: str = "http://localhost:11434",
) -> str:
    """
    Analyze an image by asking a question about it.

    A utility function that creates an ImageQuery instance and queries the model.

    Args:
        image_path: Path to the image file.
        question: Question to ask about the image.
        model: Name of the model to use.
        base_url: URL of the Ollama API.

    Returns:
        The model's response as a string.
    """
    querier = ImageQuery(base_url=base_url, model=model)
    return querier.query_image(image_path, question)


# Example usage
if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Query an image with the LLaVA model")
    parser.add_argument("image", help="Path to the image file")
    parser.add_argument("question", help="Question to ask about the image")
    parser.add_argument("--model", default="llava", help="Model name (default: llava)")
    parser.add_argument(
        "--url", default="http://localhost:11434", help="Ollama API URL"
    )

    args = parser.parse_args()

    print(f"Analyzing image: {args.image}")
    print(f"Question: {args.question}")

    response = analyze_image(args.image, args.question, args.model, args.url)
    print("\nResponse:")
    print(response)
