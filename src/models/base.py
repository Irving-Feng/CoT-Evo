"""
Base LLM provider abstraction for CoT-Evo framework.

This module defines the abstract base class for LLM providers, allowing
the framework to work with different LLM APIs and deployment methods.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List
from dataclasses import dataclass


@dataclass
class GenerationConfig:
    """Configuration for text generation."""

    temperature: float = 0.7
    max_tokens: int = 2048
    top_p: float = 1.0
    top_k: int = 1
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    stop_sequences: Optional[List[str]] = None


class LLMProvider(ABC):
    """
    Abstract base class for LLM providers.

    This class defines the interface that all LLM providers must implement,
    allowing the CoT-Evo framework to work with different LLM backends
    (OpenAI API, local models, etc.).
    """

    def __init__(
        self,
        model_name: str,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize the LLM provider.

        Args:
            model_name: Name of the model to use
            base_url: Base URL for the API (if applicable)
            api_key: API key for authentication (if applicable)
            **kwargs: Additional provider-specific parameters
        """
        self.model_name = model_name
        self.base_url = base_url
        self.api_key = api_key
        self.extra_config = kwargs

    @abstractmethod
    def generate(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
        **kwargs
    ) -> str:
        """
        Generate text synchronously.

        Args:
            prompt: Input prompt
            config: Generation configuration
            **kwargs: Additional generation parameters

        Returns:
            Generated text
        """
        pass

    @abstractmethod
    async def generate_async(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
        **kwargs
    ) -> str:
        """
        Generate text asynchronously.

        Args:
            prompt: Input prompt
            config: Generation configuration
            **kwargs: Additional generation parameters

        Returns:
            Generated text
        """
        pass

    async def generate_batch_async(
        self,
        prompts: List[str],
        config: Optional[GenerationConfig] = None,
        **kwargs
    ) -> List[str]:
        """
        Generate text for multiple prompts asynchronously.

        Args:
            prompts: List of input prompts
            config: Generation configuration (shared across all prompts)
            **kwargs: Additional generation parameters

        Returns:
            List of generated texts (in the same order as inputs)
        """
        import asyncio

        tasks = [
            self.generate_async(prompt, config, **kwargs)
            for prompt in prompts
        ]
        return await asyncio.gather(*tasks)

    @property
    @abstractmethod
    def provider_type(self) -> str:
        """Get the provider type (e.g., 'openai', 'vllm', 'anthropic')."""
        pass

    def __repr__(self) -> str:
        """String representation of the provider."""
        return f"LLMProvider(type={self.provider_type}, model={self.model_name})"


class EmbeddingProvider(ABC):
    """
    Abstract base class for embedding providers.

    Used for generating behavioral embeddings of trajectories.
    """

    def __init__(
        self,
        model_name: str,
        embedding_dim: int,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize the embedding provider.

        Args:
            model_name: Name of the embedding model
            embedding_dim: Dimension of the embedding vectors
            base_url: Base URL for the API (if applicable)
            api_key: API key for authentication (if applicable)
            **kwargs: Additional provider-specific parameters
        """
        self.model_name = model_name
        self.embedding_dim = embedding_dim
        self.base_url = base_url
        self.api_key = api_key
        self.extra_config = kwargs

    @abstractmethod
    def embed(self, text: str) -> List[float]:
        """
        Generate embedding for a single text synchronously.

        Args:
            text: Input text

        Returns:
            Embedding vector as a list of floats
        """
        pass

    @abstractmethod
    async def embed_async(self, text: str) -> List[float]:
        """
        Generate embedding for a single text asynchronously.

        Args:
            text: Input text

        Returns:
            Embedding vector as a list of floats
        """
        pass

    async def embed_batch_async(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts asynchronously.

        Args:
            texts: List of input texts

        Returns:
            List of embedding vectors (in the same order as inputs)
        """
        import asyncio

        tasks = [self.embed_async(text) for text in texts]
        return await asyncio.gather(*tasks)

    @property
    @abstractmethod
    def provider_type(self) -> str:
        """Get the provider type."""
        pass

    def __repr__(self) -> str:
        """String representation of the provider."""
        return (
            f"EmbeddingProvider(type={self.provider_type}, "
            f"model={self.model_name}, dim={self.embedding_dim})"
        )
