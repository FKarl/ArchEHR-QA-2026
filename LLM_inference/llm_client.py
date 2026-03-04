"""
LLM Client - Unified interface for Ollama and MLX-LM backends

This module provides a simple interface to interact with local LLM servers.
Supports both Ollama and MLX-LM backends with OpenAI-compatible APIs.
"""

import requests
import json
import time
from typing import Generator, Optional, Dict, Any, List
from dataclasses import dataclass


@dataclass
class InferenceResult:
    """Result of an LLM inference call."""

    text: str
    model: str
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    inference_time: float = 0.0
    tokens_per_second: float = 0.0


class LLMClient:
    """
    Unified LLM client for local inference servers.

    Supports:
    - Ollama (default, recommended for Mac Studio)
    - MLX-LM server

    Usage:
        client = LLMClient(backend="ollama", model="llama3.2:3b")
        result = client.generate("What is AI?")
        print(result.text)
    """

    def __init__(
        self,
        backend: str = "ollama",
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialize the LLM client.

        Args:
            backend: "ollama" or "mlx"
            model: Model name/identifier
            base_url: API base URL (auto-detected if not provided)
            **kwargs: Additional parameters passed to generation
        """
        self.backend = backend.lower()
        self.kwargs = kwargs

        # Set defaults based on backend
        if self.backend == "ollama":
            from config import OLLAMA_BASE_URL, OLLAMA_MODEL

            self.base_url = base_url or OLLAMA_BASE_URL
            self.model = model or OLLAMA_MODEL
        elif self.backend == "mlx":
            from config import MLX_BASE_URL, MLX_MODEL

            self.base_url = base_url or MLX_BASE_URL
            # MLX server uses whatever model was loaded at startup
            # "default" or "default_model" tells the server to use that model
            self.model = (
                model if model and model != "default" else "default_model"
            )
        else:
            raise ValueError(
                f"Unknown backend: {backend}. Use 'ollama' or 'mlx'"
            )

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        stream: bool = False,
        **kwargs,
    ) -> InferenceResult:
        """
        Generate text from a prompt.

        Args:
            prompt: The user prompt
            system_prompt: Optional system prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            stream: Whether to stream the response
            **kwargs: Additional backend-specific parameters

        Returns:
            InferenceResult with generated text and metrics
        """
        from config import MAX_TOKENS, TEMPERATURE, TOP_P

        max_tokens = max_tokens or MAX_TOKENS
        temperature = temperature if temperature is not None else TEMPERATURE
        top_p = top_p if top_p is not None else TOP_P

        start_time = time.perf_counter()

        if self.backend == "ollama":
            result = self._generate_ollama(
                prompt,
                system_prompt,
                max_tokens,
                temperature,
                top_p,
                stream,
                **kwargs,
            )
        else:
            result = self._generate_mlx(
                prompt,
                system_prompt,
                max_tokens,
                temperature,
                top_p,
                stream,
                **kwargs,
            )

        end_time = time.perf_counter()
        result.inference_time = end_time - start_time

        # Calculate tokens per second
        if result.completion_tokens and result.inference_time > 0:
            result.tokens_per_second = (
                result.completion_tokens / result.inference_time
            )

        return result

    def _generate_ollama(
        self,
        prompt: str,
        system_prompt: Optional[str],
        max_tokens: int,
        temperature: float,
        top_p: float,
        stream: bool,
        **kwargs,
    ) -> InferenceResult:
        """Generate using Ollama API. Tries /api/chat first, falls back to /api/generate."""

        # Try the newer /api/chat endpoint first
        try:
            return self._generate_ollama_chat(
                prompt, system_prompt, max_tokens, temperature, top_p, stream
            )
        except RuntimeError as e:
            if "404" in str(e):
                # Fall back to older /api/generate endpoint
                return self._generate_ollama_generate(
                    prompt,
                    system_prompt,
                    max_tokens,
                    temperature,
                    top_p,
                    stream,
                )
            raise

    def _generate_ollama_chat(
        self,
        prompt: str,
        system_prompt: Optional[str],
        max_tokens: int,
        temperature: float,
        top_p: float,
        stream: bool,
    ) -> InferenceResult:
        """Generate using Ollama /api/chat endpoint (newer API)."""
        url = f"{self.base_url}/api/chat"

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": stream,
            "options": {
                "num_predict": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
            },
        }

        try:
            response = requests.post(url, json=payload, timeout=300)
            response.raise_for_status()
            data = response.json()

            return InferenceResult(
                text=data.get("message", {}).get("content", ""),
                model=self.model,
                prompt_tokens=data.get("prompt_eval_count"),
                completion_tokens=data.get("eval_count"),
                total_tokens=(data.get("prompt_eval_count", 0) or 0)
                + (data.get("eval_count", 0) or 0),
            )
        except requests.exceptions.ConnectionError:
            raise ConnectionError(
                f"Cannot connect to Ollama at {self.base_url}. "
                "Make sure Ollama is running: `ollama serve`"
            )
        except requests.exceptions.HTTPError as e:
            # Include response body for debugging
            error_body = ""
            try:
                error_body = e.response.text[:200]
            except:
                pass
            raise RuntimeError(f"Ollama API error: {e} - {error_body}")
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Ollama API error: {e}")

    def _generate_ollama_generate(
        self,
        prompt: str,
        system_prompt: Optional[str],
        max_tokens: int,
        temperature: float,
        top_p: float,
        stream: bool,
    ) -> InferenceResult:
        """Generate using Ollama /api/generate endpoint (older API)."""
        url = f"{self.base_url}/api/generate"

        # Combine system prompt with user prompt for the generate endpoint
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"

        payload = {
            "model": self.model,
            "prompt": full_prompt,
            "stream": stream,
            "options": {
                "num_predict": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
            },
        }

        try:
            response = requests.post(url, json=payload, timeout=300)
            response.raise_for_status()
            data = response.json()

            return InferenceResult(
                text=data.get("response", ""),
                model=self.model,
                prompt_tokens=data.get("prompt_eval_count"),
                completion_tokens=data.get("eval_count"),
                total_tokens=(data.get("prompt_eval_count", 0) or 0)
                + (data.get("eval_count", 0) or 0),
            )
        except requests.exceptions.ConnectionError:
            raise ConnectionError(
                f"Cannot connect to Ollama at {self.base_url}. "
                "Make sure Ollama is running: `ollama serve`"
            )
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Ollama API error: {e}")

    def _generate_mlx(
        self,
        prompt: str,
        system_prompt: Optional[str],
        max_tokens: int,
        temperature: float,
        top_p: float,
        stream: bool,
        **kwargs,
    ) -> InferenceResult:
        """Generate using MLX-LM server (OpenAI-compatible API)."""
        url = f"{self.base_url}/v1/chat/completions"

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stream": stream,
        }

        try:
            response = requests.post(url, json=payload, timeout=300)
            response.raise_for_status()
            data = response.json()

            choice = data.get("choices", [{}])[0]
            usage = data.get("usage", {})

            return InferenceResult(
                text=choice.get("message", {}).get("content", ""),
                model=self.model,
                prompt_tokens=usage.get("prompt_tokens"),
                completion_tokens=usage.get("completion_tokens"),
                total_tokens=usage.get("total_tokens"),
            )
        except requests.exceptions.ConnectionError:
            raise ConnectionError(
                f"Cannot connect to MLX server at {self.base_url}. "
                "Make sure the MLX server is running: `python mlx_server.py`"
            )
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"MLX API error: {e}")

    def chat(
        self,
        messages: List[Dict[str, str]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs,
    ) -> InferenceResult:
        """
        Multi-turn chat interface.

        Args:
            messages: List of message dicts with 'role' and 'content' keys
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Returns:
            InferenceResult with generated response
        """
        from config import MAX_TOKENS, TEMPERATURE, TOP_P

        max_tokens = max_tokens or MAX_TOKENS
        temperature = temperature if temperature is not None else TEMPERATURE

        start_time = time.perf_counter()

        if self.backend == "ollama":
            url = f"{self.base_url}/api/chat"
            payload = {
                "model": self.model,
                "messages": messages,
                "stream": False,
                "options": {
                    "num_predict": max_tokens,
                    "temperature": temperature,
                },
            }
            response = requests.post(url, json=payload, timeout=300)
            response.raise_for_status()
            data = response.json()

            result = InferenceResult(
                text=data.get("message", {}).get("content", ""),
                model=self.model,
                prompt_tokens=data.get("prompt_eval_count"),
                completion_tokens=data.get("eval_count"),
            )
        else:
            url = f"{self.base_url}/v1/chat/completions"
            payload = {
                "model": self.model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
            response = requests.post(url, json=payload, timeout=300)
            response.raise_for_status()
            data = response.json()

            choice = data.get("choices", [{}])[0]
            usage = data.get("usage", {})

            result = InferenceResult(
                text=choice.get("message", {}).get("content", ""),
                model=self.model,
                prompt_tokens=usage.get("prompt_tokens"),
                completion_tokens=usage.get("completion_tokens"),
            )

        end_time = time.perf_counter()
        result.inference_time = end_time - start_time
        if result.completion_tokens and result.inference_time > 0:
            result.tokens_per_second = (
                result.completion_tokens / result.inference_time
            )

        return result

    def list_models(self) -> List[str]:
        """List available models."""
        if self.backend == "ollama":
            url = f"{self.base_url}/api/tags"
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            data = response.json()
            return [m["name"] for m in data.get("models", [])]
        else:
            url = f"{self.base_url}/v1/models"
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            data = response.json()
            return [m["id"] for m in data.get("data", [])]

    def pull_model(self, model_name: str) -> bool:
        """
        Pull/download a model (Ollama only).

        Args:
            model_name: Name of the model to pull

        Returns:
            True if successful
        """
        if self.backend != "ollama":
            raise NotImplementedError(
                "Model pulling is only supported for Ollama"
            )

        url = f"{self.base_url}/api/pull"
        payload = {"name": model_name, "stream": False}

        print(f"Pulling model {model_name}... This may take a while.")
        response = requests.post(url, json=payload, timeout=3600)
        response.raise_for_status()
        print(f"Model {model_name} pulled successfully!")
        return True

    def is_server_running(self) -> bool:
        """Check if the inference server is running."""
        try:
            if self.backend == "ollama":
                response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            else:
                response = requests.get(
                    f"{self.base_url}/v1/models", timeout=5
                )
            return response.status_code == 200
        except:
            return False


def get_client(
    backend: Optional[str] = None, model: Optional[str] = None, **kwargs
) -> LLMClient:
    """
    Factory function to create an LLM client.

    Uses config.py defaults if not specified.

    Args:
        backend: "ollama" or "mlx"
        model: Model name
        **kwargs: Additional parameters

    Returns:
        Configured LLMClient instance
    """
    from config import BACKEND

    backend = backend or BACKEND
    return LLMClient(backend=backend, model=model, **kwargs)
