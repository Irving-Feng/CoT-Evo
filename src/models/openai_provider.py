"""
OpenAI-compatible API provider for CoT-Evo framework.

This module implements LLM providers for any API that is compatible with
the OpenAI API format, including OpenAI, Azure OpenAI, DeepSeek, Qwen, etc.
"""

import asyncio
from typing import Optional, List, Dict, Any
from openai import AsyncOpenAI, OpenAI
import logging

from .base import LLMProvider, EmbeddingProvider, GenerationConfig


logger = logging.getLogger(__name__)


class OpenAIProvider(LLMProvider):
    """
    OpenAI-compatible API provider for LLMs.

    Supports any API that follows the OpenAI API format, including:
    - OpenAI (GPT-4, GPT-3.5)
    - Azure OpenAI
    - DeepSeek
    - Qwen (via DashScope API)
    - And many others
    """

    def __init__(
        self,
        model_name: str,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: int = 120,
        max_retries: int = 3,
        **kwargs
    ):
        """
        Initialize the OpenAI-compatible provider.

        Args:
            model_name: Name of the model to use
            base_url: Base URL for the API (defaults to OpenAI's API)
            api_key: API key for authentication
            timeout: Request timeout in seconds
            max_retries: Maximum number of retries on failure
            **kwargs: Additional parameters
        """
        super().__init__(model_name, base_url, api_key, **kwargs)

        self.timeout = timeout
        self.max_retries = max_retries

        # Initialize sync and async clients
        client_kwargs = {"api_key": api_key or "dummy-key"}
        if base_url:
            client_kwargs["base_url"] = base_url

        self._sync_client = OpenAI(
            **client_kwargs,
            timeout=timeout,
            max_retries=max_retries,
        )
        self._async_client = AsyncOpenAI(
            **client_kwargs,
            timeout=timeout,
            max_retries=max_retries,
        )

    def generate(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
        messages: Optional[List[Dict[str, str]]] = None,
        **kwargs
    ) -> str:
        """
        Generate text synchronously.

        Args:
            prompt: Input prompt (used if messages is None)
            config: Generation configuration
            messages: Optional list of messages for chat-style API
            **kwargs: Additional generation parameters (overrides config)

        Returns:
            Generated text
        """
        if messages is None:
            messages = [{"role": "user", "content": prompt}]

        config = config or GenerationConfig()

        # Build API parameters, preferring kwargs over config
        api_params = {
            "model": self.model_name,
            "messages": messages,
            "temperature": kwargs.pop("temperature", config.temperature),
            "max_tokens": kwargs.pop("max_tokens", config.max_tokens),
            "top_p": kwargs.pop("top_p", config.top_p),
            "frequency_penalty": kwargs.pop("frequency_penalty", config.frequency_penalty),
            "presence_penalty": kwargs.pop("presence_penalty", config.presence_penalty),
        }

        # Handle stop parameter - prefer kwargs over config
        if "stop" in kwargs:
            api_params["stop"] = kwargs.pop("stop")
        elif config.stop_sequences:
            api_params["stop"] = config.stop_sequences

        # Add any remaining kwargs
        api_params.update(kwargs)

        try:
            response = self._sync_client.chat.completions.create(**api_params)
            return response.choices[0].message.content

        except Exception as e:
            logger.error(f"Error generating with {self.model_name}: {e}")
            raise

    async def generate_async(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
        messages: Optional[List[Dict[str, str]]] = None,
        **kwargs
    ) -> str:
        """
        Generate text asynchronously.

        Args:
            prompt: Input prompt (used if messages is None)
            config: Generation configuration
            messages: Optional list of messages for chat-style API
            **kwargs: Additional generation parameters (overrides config)

        Returns:
            Generated text
        """
        if messages is None:
            messages = [{"role": "user", "content": prompt}]

        config = config or GenerationConfig()

        # Build API parameters, preferring kwargs over config
        api_params = {
            "model": self.model_name,
            "messages": messages,
            "temperature": kwargs.pop("temperature", config.temperature),
            "max_tokens": kwargs.pop("max_tokens", config.max_tokens),
            "top_p": kwargs.pop("top_p", config.top_p),
            "frequency_penalty": kwargs.pop("frequency_penalty", config.frequency_penalty),
            "presence_penalty": kwargs.pop("presence_penalty", config.presence_penalty),
        }

        # Handle stop parameter - prefer kwargs over config
        if "stop" in kwargs:
            api_params["stop"] = kwargs.pop("stop")
        elif config.stop_sequences:
            api_params["stop"] = config.stop_sequences

        # Add any remaining kwargs
        api_params.update(kwargs)

        try:
            response = await self._async_client.chat.completions.create(**api_params)
            return response.choices[0].message.content

        except Exception as e:
            logger.error(f"Error generating async with {self.model_name}: {e}")
            raise

    @property
    def provider_type(self) -> str:
        """Get the provider type."""
        return "openai"


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """
    OpenAI-compatible API provider for embeddings.

    Supports OpenAI's embedding API and compatible services.
    """

    def __init__(
        self,
        model_name: str = "text-embedding-3-small",
        embedding_dim: int = 1536,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: int = 60,
        max_retries: int = 3,
        **kwargs
    ):
        """
        Initialize the OpenAI-compatible embedding provider.

        Args:
            model_name: Name of the embedding model
            embedding_dim: Dimension of the embedding vectors
            base_url: Base URL for the API
            api_key: API key for authentication
            timeout: Request timeout in seconds
            max_retries: Maximum number of retries on failure
            **kwargs: Additional parameters
        """
        super().__init__(model_name, embedding_dim, base_url, api_key, **kwargs)

        self.timeout = timeout
        self.max_retries = max_retries

        # Initialize clients
        client_kwargs = {"api_key": api_key or "dummy-key"}
        if base_url:
            client_kwargs["base_url"] = base_url

        self._sync_client = OpenAI(**client_kwargs, timeout=timeout, max_retries=max_retries)
        self._async_client = AsyncOpenAI(**client_kwargs, timeout=timeout, max_retries=max_retries)

    def embed(self, text: str) -> List[float]:
        """
        Generate embedding for a single text synchronously.

        Args:
            text: Input text

        Returns:
            Embedding vector as a list of floats
        """
        try:
            response = self._sync_client.embeddings.create(
                model=self.model_name,
                input=text
            )
            return response.data[0].embedding

        except Exception as e:
            logger.error(f"Error embedding with {self.model_name}: {e}")
            raise

    async def embed_async(self, text: str) -> List[float]:
        """
        Generate embedding for a single text asynchronously.

        Args:
            text: Input text

        Returns:
            Embedding vector as a list of floats
        """
        try:
            response = await self._async_client.embeddings.create(
                model=self.model_name,
                input=text
            )
            return response.data[0].embedding

        except Exception as e:
            logger.error(f"Error embedding async with {self.model_name}: {e}")
            raise

    async def embed_batch_async(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts asynchronously.

        Uses the batch embedding API for efficiency.

        Args:
            texts: List of input texts

        Returns:
            List of embedding vectors (in the same order as inputs)
        """
        try:
            response = await self._async_client.embeddings.create(
                model=self.model_name,
                input=texts
            )
            # Sort results by index to ensure order
            embeddings = [e.embedding for e in sorted(response.data, key=lambda x: x.index)]
            return embeddings

        except Exception as e:
            logger.error(f"Error batch embedding with {self.model_name}: {e}")
            raise

    @property
    def provider_type(self) -> str:
        """Get the provider type."""
        return "openai"
