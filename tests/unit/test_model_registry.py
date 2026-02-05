"""
Unit tests for ModelRegistry class.
"""

import pytest
import tempfile
import os
from pathlib import Path
from src.models.registry import ModelRegistry
from src.models.openai_provider import OpenAIProvider


class TestModelRegistry:
    """Test cases for ModelRegistry class."""

    def test_initialization_empty(self):
        """Test creating an empty registry."""
        registry = ModelRegistry()

        assert len(registry.thinkers) == 0
        assert len(registry.models) == 0
        assert registry.selection_mode == "auto"

    def test_register_model(self):
        """Test registering a model."""
        registry = ModelRegistry()

        mock_provider = OpenAIProvider(
            model_name="test-model",
            base_url="https://api.test.com",
            api_key="test-key"
        )

        registry.register_model("test-model", mock_provider, role="thinker")

        assert "test-model" in registry.models
        assert "test-model" in registry.thinkers

    def test_register_multiple_thinkers(self):
        """Test registering multiple thinker models."""
        registry = ModelRegistry()

        for i in range(3):
            provider = OpenAIProvider(
                model_name=f"model-{i}",
                base_url="https://api.test.com",
                api_key="test-key"
            )
            registry.register_model(f"model-{i}", provider, role="thinker")

        assert len(registry.thinkers) == 3
        assert len(registry.models) == 3

    def test_get_model(self):
        """Test getting a model by name."""
        registry = ModelRegistry()

        provider = OpenAIProvider(
            model_name="test-model",
            base_url="https://api.test.com",
            api_key="test-key"
        )
        registry.register_model("test-model", provider)

        retrieved = registry.get_model("test-model")

        assert retrieved is provider
        assert retrieved.model_name == "test-model"

    def test_get_model_nonexistent(self):
        """Test getting a model that doesn't exist."""
        registry = ModelRegistry()

        retrieved = registry.get_model("nonexistent")

        assert retrieved is None

    def test_get_thinker_models(self):
        """Test getting all thinker models."""
        registry = ModelRegistry()

        # Register some thinkers
        for i in range(3):
            provider = OpenAIProvider(
                model_name=f"thinker-{i}",
                base_url="https://api.test.com",
                api_key="test-key"
            )
            registry.register_model(f"thinker-{i}", provider, role="thinker")

        # Register a non-thinker (e.g., judge)
        judge_provider = OpenAIProvider(
            model_name="judge",
            base_url="https://api.test.com",
            api_key="test-key"
        )
        registry.register_model("judge", judge_provider, role="judge")

        thinkers = registry.get_thinker_models()

        assert len(thinkers) == 3
        assert all(t.model_name.startswith("thinker-") for t in thinkers)

    def test_get_random_thinker(self):
        """Test getting a random thinker model."""
        registry = ModelRegistry()

        for i in range(3):
            provider = OpenAIProvider(
                model_name=f"model-{i}",
                base_url="https://api.test.com",
                api_key="test-key"
            )
            registry.register_model(f"model-{i}", provider, role="thinker")

        # Get random thinker multiple times
        retrieved = set()
        for _ in range(10):
            thinker = registry.get_random_thinker()
            assert thinker is not None
            retrieved.add(thinker.model_name)

        # Should have gotten at least one different model (likely)
        assert len(retrieved) >= 1

    def test_get_random_thinker_empty(self):
        """Test getting random thinker from empty registry."""
        registry = ModelRegistry()

        thinker = registry.get_random_thinker()

        assert thinker is None

    def test_set_global_operator(self):
        """Test setting global operator model."""
        registry = ModelRegistry()

        provider1 = OpenAIProvider(
            model_name="model-1",
            base_url="https://api.test.com",
            api_key="test-key"
        )
        provider2 = OpenAIProvider(
            model_name="model-2",
            base_url="https://api.test.com",
            api_key="test-key"
        )

        registry.register_model("model-1", provider1, role="thinker")
        registry.register_model("model-2", provider2, role="thinker")

        registry.set_global_operator("model-2")

        assert registry.global_operator_model is provider2
        assert registry.selection_mode == "global"

    def test_set_global_operator_nonexistent(self):
        """Test setting nonexistent model as global operator."""
        registry = ModelRegistry()

        with pytest.raises(ValueError):
            registry.set_global_operator("nonexistent")

    def test_set_auto_mode(self):
        """Test setting auto mode."""
        registry = ModelRegistry()

        provider = OpenAIProvider(
            model_name="test-model",
            base_url="https://api.test.com",
            api_key="test-key"
        )
        registry.register_model("test-model", provider)

        # First set global mode
        registry.set_global_operator("test-model")
        assert registry.selection_mode == "global"

        # Then switch to auto
        registry.set_auto_mode()

        assert registry.selection_mode == "auto"
        assert registry.global_operator_model is None

    def test_get_operator_model_auto_mode(self):
        """Test getting operator model in auto mode."""
        from src.core.trajectory import Trajectory

        registry = ModelRegistry()

        provider1 = OpenAIProvider(
            model_name="deepseek-r1",
            base_url="https://api.test.com",
            api_key="test-key"
        )
        provider2 = OpenAIProvider(
            model_name="qwen-32b",
            base_url="https://api.test.com",
            api_key="test-key"
        )

        registry.register_model("deepseek-r1", provider1, role="thinker")
        registry.register_model("qwen-32b", provider2, role="thinker")

        # Create trajectory with source model
        traj = Trajectory(
            query="Test",
            answer="Answer",
            reasoning="Reasoning",
            source_model="deepseek-r1"
        )

        # In auto mode, should return the source model
        operator = registry.get_operator_model(traj)

        assert operator is provider1
        assert operator.model_name == "deepseek-r1"

    def test_get_operator_model_global_mode(self):
        """Test getting operator model in global mode."""
        from src.core.trajectory import Trajectory

        registry = ModelRegistry()

        provider1 = OpenAIProvider(
            model_name="deepseek-r1",
            base_url="https://api.test.com",
            api_key="test-key"
        )
        provider2 = OpenAIProvider(
            model_name="qwen-32b",
            base_url="https://api.test.com",
            api_key="test-key"
        )

        registry.register_model("deepseek-r1", provider1, role="thinker")
        registry.register_model("qwen-32b", provider2, role="thinker")

        # Set global operator
        registry.set_global_operator("qwen-32b")

        # Create trajectory with different source model
        traj = Trajectory(
            query="Test",
            answer="Answer",
            reasoning="Reasoning",
            source_model="deepseek-r1"
        )

        # In global mode, should return the global operator
        operator = registry.get_operator_model(traj)

        assert operator is provider2
        assert operator.model_name == "qwen-32b"

    def test_get_operator_model_no_trajectory(self):
        """Test getting operator model without trajectory."""
        registry = ModelRegistry()

        provider = OpenAIProvider(
            model_name="test-model",
            base_url="https://api.test.com",
            api_key="test-key"
        )
        registry.register_model("test-model", provider, role="thinker")

        # No trajectory provided
        operator = registry.get_operator_model(None)

        # Should return random thinker
        assert operator is provider

    def test_substitute_env_var(self):
        """Test environment variable substitution."""
        registry = ModelRegistry()

        os.environ["TEST_VAR"] = "test_value"

        # Test ${VAR} format
        result = registry._substitute_env("${TEST_VAR}", os.environ)
        assert result == "test_value"

        # Test with prefix/suffix
        result = registry._substitute_env("prefix_${TEST_VAR}_suffix", os.environ)
        assert result == "prefix_test_value_suffix"

        # Test missing variable
        result = registry._substitute_env("${MISSING_VAR}", os.environ)
        assert result == ""

        # Clean up
        del os.environ["TEST_VAR"]

    def test_list_models(self):
        """Test listing all registered models."""
        registry = ModelRegistry()

        for i in range(3):
            provider = OpenAIProvider(
                model_name=f"model-{i}",
                base_url="https://api.test.com",
                api_key="test-key"
            )
            registry.register_model(f"model-{i}", provider)

        models = registry.list_models()

        assert len(models) == 3
        assert set(models) == {"model-0", "model-1", "model-2"}

    def test_repr(self):
        """Test string representation."""
        registry = ModelRegistry()

        provider = OpenAIProvider(
            model_name="test-model",
            base_url="https://api.test.com",
            api_key="test-key"
        )
        registry.register_model("test-model", provider, role="thinker")

        repr_str = repr(registry)

        assert "ModelRegistry" in repr_str
        assert "thinkers=1" in repr_str
        assert "auto" in repr_str
