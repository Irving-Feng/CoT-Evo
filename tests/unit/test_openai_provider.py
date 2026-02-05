"""
Unit tests for OpenAI-compatible providers.
"""

import pytest
from src.models.openai_provider import OpenAIProvider, OpenAIEmbeddingProvider
from src.models.base import GenerationConfig


class TestGenerationConfig:
    """Test cases for GenerationConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = GenerationConfig()

        assert config.temperature == 0.7
        assert config.max_tokens == 2048
        assert config.top_p == 1.0
        assert config.frequency_penalty == 0.0
        assert config.presence_penalty == 0.0

    def test_custom_config(self):
        """Test custom configuration values."""
        config = GenerationConfig(
            temperature=0.5,
            max_tokens=4096,
            top_p=0.9,
            frequency_penalty=0.5
        )

        assert config.temperature == 0.5
        assert config.max_tokens == 4096
        assert config.top_p == 0.9
        assert config.frequency_penalty == 0.5


class TestOpenAIProvider:
    """Test cases for OpenAIProvider."""

    def test_initialization(self):
        """Test provider initialization."""
        provider = OpenAIProvider(
            model_name="gpt-4",
            base_url="https://api.openai.com/v1",
            api_key="test-key"
        )

        assert provider.model_name == "gpt-4"
        assert provider.base_url == "https://api.openai.com/v1"
        assert provider.api_key == "test-key"
        assert provider.provider_type == "openai"

    def test_initialization_without_base_url(self):
        """Test provider initialization without base URL (uses default)."""
        provider = OpenAIProvider(
            model_name="gpt-4",
            api_key="test-key"
        )

        assert provider.model_name == "gpt-4"
        assert provider.provider_type == "openai"

    def test_provider_type(self):
        """Test provider type property."""
        provider = OpenAIProvider(
            model_name="test-model",
            base_url="https://api.test.com",
            api_key="test-key"
        )

        assert provider.provider_type == "openai"

    def test_repr(self):
        """Test string representation."""
        provider = OpenAIProvider(
            model_name="gpt-4",
            base_url="https://api.openai.com/v1",
            api_key="test-key"
        )

        repr_str = repr(provider)

        assert "LLMProvider" in repr_str
        assert "openai" in repr_str
        assert "gpt-4" in repr_str


class TestOpenAIEmbeddingProvider:
    """Test cases for OpenAIEmbeddingProvider."""

    def test_initialization(self):
        """Test embedding provider initialization."""
        provider = OpenAIEmbeddingProvider(
            model_name="text-embedding-3-small",
            embedding_dim=1536,
            base_url="https://api.openai.com/v1",
            api_key="test-key"
        )

        assert provider.model_name == "text-embedding-3-small"
        assert provider.embedding_dim == 1536
        assert provider.provider_type == "openai"

    def test_initialization_defaults(self):
        """Test embedding provider initialization with defaults."""
        provider = OpenAIEmbeddingProvider(
            model_name="text-embedding-3-small"
        )

        assert provider.model_name == "text-embedding-3-small"
        assert provider.embedding_dim == 1536

    def test_provider_type(self):
        """Test provider type property."""
        provider = OpenAIEmbeddingProvider(
            model_name="text-embedding-ada-002",
            embedding_dim=1536
        )

        assert provider.provider_type == "openai"

    def test_repr(self):
        """Test string representation."""
        provider = OpenAIEmbeddingProvider(
            model_name="text-embedding-3-small",
            embedding_dim=1536
        )

        repr_str = repr(provider)

        assert "EmbeddingProvider" in repr_str
        assert "openai" in repr_str
        assert "text-embedding-3-small" in repr_str
        assert "1536" in repr_str
