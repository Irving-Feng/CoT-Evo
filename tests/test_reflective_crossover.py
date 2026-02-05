"""
Unit tests for reflective crossover operation module.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock
from src.variation.crossover import ReflectiveCrossover
from src.core.trajectory import Trajectory


class TestReflectiveCrossover:
    """Test suite for ReflectiveCrossover class."""

    @pytest.fixture
    def mock_registry(self):
        """Create a mock model registry."""
        registry = MagicMock()
        registry.get_operator_model = MagicMock()
        return registry

    @pytest.fixture
    def mock_recombiner(self):
        """Create a mock recombiner model."""
        model = MagicMock()
        model.model_name = "recombiner-model"
        model.generate_async = AsyncMock()
        return model

    @pytest.fixture
    def crossover(self, mock_registry):
        """Create a ReflectiveCrossover instance."""
        return ReflectiveCrossover(model_registry=mock_registry)

    @pytest.fixture
    def sample_target(self):
        """Create a sample target trajectory (incorrect answer)."""
        return Trajectory(
            query="What is the molecular weight of water?",
            answer="20.0 g/mol",  # Incorrect
            reasoning="Water is H2O. H=1, O=16, so H2O=1×2+16=18 g/mol. Answer: 20.0 g/mol",
            source_model="model-a",
            generation_method="vanilla"
        )

    @pytest.fixture
    def sample_provider(self):
        """Create a sample provider trajectory."""
        return Trajectory(
            query="What is the molecular weight of water?",
            answer="18.015 g/mol",
            reasoning="Water is H2O. Atomic weights: H=1.008, O=15.999. So H2O = 2×1.008 + 15.999 = 18.015 g/mol.",
            source_model="model-b",
            generation_method="knowledge_augmented"
        )

    @pytest.mark.asyncio
    async def test_crossover_success(self, crossover, mock_registry, mock_recombiner, sample_target, sample_provider):
        """Test successful crossover operation."""
        mock_registry.get_operator_model.return_value = mock_recombiner

        # Mock responses for the three steps
        # Step 1: Identify binding point
        mock_recombiner.generate_async.side_effect = [
            # First call: identify binding point
            "[RESULT_START]Atomic weights: H=1.008, O=15.999[RESULT_END]",
            # Second call: extract unique info
            """[RESULT_START]
* Use precise atomic weights: H=1.008, O=15.999
* Calculate: 2×1.008 + 15.999
[RESULT_END]""",
            # Third call: generate recombined
            "Water is H2O. Using precise atomic weights H=1.008, O=15.999, we calculate 2×1.008 + 15.999 = 18.015 g/mol.\nAnswer: 18.015 g/mol"
        ]

        result = await crossover.crossover(sample_target, sample_provider)

        assert isinstance(result, Trajectory)
        assert result.generation_method == "crossover"
        assert result.source_model == "recombiner-model"

    @pytest.mark.asyncio
    async def test_crossover_skip_correct_answer(self, crossover, mock_registry, sample_target, sample_provider):
        """Test that crossover is skipped when target answer is correct."""
        # Set target as correct
        sample_target._exact_match = True

        mock_registry.get_operator_model.return_value = mock_recombiner

        result = await crossover.crossover(sample_target, sample_provider)

        # Should return the original target
        assert result == sample_target

    @pytest.mark.asyncio
    async def test_identify_binding_point(self, crossover, mock_recombiner, sample_provider):
        """Test binding point identification."""
        # Mock response
        mock_recombiner.generate_async.return_value = "[RESULT_START]Atomic weights: H=1.008, O=15.999[RESULT_END]"

        binding_point = await crossover._identify_binding_point(sample_provider, mock_recombiner)

        # Should return a valid position
        assert isinstance(binding_point, int)
        assert 0 <= binding_point <= len(sample_provider.reasoning)

    @pytest.mark.asyncio
    async def test_identify_binding_point_fallback(self, crossover, mock_recombiner, sample_provider):
        """Test binding point identification fallback when pattern not found."""
        # Mock response without RESULT markers
        mock_recombiner.generate_async.return_value = "Some response without markers"

        binding_point = await crossover._identify_binding_point(sample_provider, mock_recombiner)

        # Should return 80% of reasoning length as fallback
        expected_fallback = int(len(sample_provider.reasoning) * 0.8)
        assert binding_point == expected_fallback

    @pytest.mark.asyncio
    async def test_extract_unique_info(self, crossover, mock_recombiner, sample_target, sample_provider):
        """Test unique information extraction."""
        mock_response = """[RESULT_START]
* Provider uses atomic weights H=1.008 and O=15.999
* Provider calculates 2×1.008 + 15.999
[RESULT_END]"""
        mock_recombiner.generate_async.return_value = mock_response

        unique_info = await crossover._extract_unique_info(
            sample_target, sample_provider, mock_recombiner
        )

        assert unique_info is not None
        assert "1.008" in unique_info or "15.999" in unique_info

    @pytest.mark.asyncio
    async def test_extract_unique_info_no_match(self, crossover, mock_recombiner, sample_target, sample_provider):
        """Test unique info extraction when no pattern found."""
        mock_recombiner.generate_async.return_value = "No unique info found"

        unique_info = await crossover._extract_unique_info(
            sample_target, sample_provider, mock_recombiner
        )

        assert unique_info is None

    @pytest.mark.asyncio
    async def test_generate_recombined(self, crossover, mock_recombiner, sample_provider):
        """Test recombined reasoning generation."""
        prefix = "Water is H2O."
        unique_info = "Use atomic weights: H=1.008, O=15.999"

        mock_recombiner.generate_async.return_value = (
            "Water is H2O. Using atomic weights H=1.008, O=15.999, "
            "we calculate 2×1.008 + 15.999 = 18.015 g/mol.\n"
            "Answer: 18.015 g/mol"
        )

        result = await crossover._generate_recombined(
            prefix, unique_info, mock_recombiner, sample_provider
        )

        assert "18.015" in result or "H=1.008" in result

    @pytest.mark.asyncio
    async def test_is_reasoning_model(self, crossover, mock_recombiner):
        """Test reasoning model detection."""
        # Test with reasoning model name
        mock_recombiner.model_name = "deepseek-r1"
        assert crossover._is_reasoning_model(mock_recombiner) is True

        # Test with non-reasoning model
        mock_recombiner.model_name = "gpt-4"
        assert crossover._is_reasoning_model(mock_recombiner) is False

    @pytest.mark.asyncio
    async def test_extract_answer(self, crossover):
        """Test answer extraction from reasoning."""
        reasoning = """Step 1: Calculate molecular weight
Step 2: Apply formula
The final answer is 42.5"""

        answer = crossover._extract_answer(reasoning)

        assert "42.5" in answer

    @pytest.mark.asyncio
    async def test_extract_answer_multiline(self, crossover):
        """Test answer extraction with multiple lines."""
        reasoning = """Reasoning line 1
Reasoning line 2
Final answer: 3.14159"""

        answer = crossover._extract_answer(reasoning)

        assert "3.14159" in answer

    @pytest.mark.asyncio
    async def test_crossover_exception_handling(self, crossover, mock_registry, mock_recombiner, sample_target, sample_provider):
        """Test crossover exception handling."""
        mock_registry.get_operator_model.return_value = mock_recombiner
        mock_recombiner.generate_async.side_effect = Exception("API error")

        result = await crossover.crossover(sample_target, sample_provider)

        # Should return None on error
        assert result is None


class TestReflectiveCrossoverEdgeCases:
    """Test edge cases for ReflectiveCrossover."""

    @pytest.fixture
    def crossover(self, mock_registry):
        return ReflectiveCrossover(model_registry=mock_registry)

    @pytest.fixture
    def mock_registry(self):
        return MagicMock()

    @pytest.fixture
    def mock_model(self):
        model = MagicMock()
        model.model_name = "test-model"
        model.generate_async = AsyncMock()
        return model

    @pytest.mark.asyncio
    async def test_identify_binding_point_invalid_position(self, crossover, mock_model):
        """Test binding point identification when response suggests position 0."""
        provider = Trajectory(
            query="Test",
            answer="Answer",
            reasoning="Full reasoning text",
            source_model="model"
        )

        # Mock returns empty result
        mock_model.generate_async.return_value = "[RESULT_START][RESULT_END]"

        binding_point = await crossover._identify_binding_point(provider, mock_model)

        # Should return fallback (80%)
        expected = int(len(provider.reasoning) * 0.8)
        assert binding_point == expected

    @pytest.mark.asyncio
    async def test_generate_recombined_with_reasoning_model(self, crossover, mock_model):
        """Test recombined generation with reasoning model (adds guidance)."""
        mock_model.model_name = "deepseek-r1"
        mock_model.generate_async = AsyncMock(return_value="Enhanced reasoning")

        result = await crossover._generate_recombined(
            "Prefix", "Info", mock_model, None
        )

        # Should add special guidance for reasoning models
        mock_model.generate_async.assert_called_once()
        call_args = mock_model.generate_async.call_args
        prompt = call_args[0][0]
        # Check that guidance was added
        assert "externally provided information" in prompt.lower()

    @pytest.mark.asyncio
    async def test_extract_unique_info_bullets(self, crossover, mock_model):
        """Test extraction converts bullet points to text."""
        target = Trajectory(
            query="Test", answer="A", reasoning="Target reasoning", source_model="model"
        )
        provider = Trajectory(
            query="Test", answer="B", reasoning="Provider reasoning", source_model="model"
        )

        mock_model.generate_async.return_value = """[RESULT_START]
* Point one
* Point two
* Point three
[RESULT_END]"""

        unique_info = await crossover._extract_unique_info(target, provider, mock_model)

        assert unique_info is not None
        assert "Point one" in unique_info
        assert "Point two" in unique_info
        assert "Point three" in unique_info

    @pytest.mark.asyncio
    async def test_crossover_with_empty_reasoning(self, crossover, mock_registry, mock_model):
        """Test crossover with empty provider reasoning."""
        target = Trajectory(
            query="Test",
            answer="Wrong",
            reasoning="Target reasoning",
            source_model="model"
        )
        provider = Trajectory(
            query="Test",
            answer="Correct",
            reasoning="",  # Empty reasoning
            source_model="model"
        )

        mock_registry.get_operator_model.return_value = mock_model
        mock_model.generate_async.return_value = "[RESULT_START]Some sentence[RESULT_END]"

        result = await crossover.crossover(target, provider)

        # Should handle gracefully (may return None or a new trajectory)
        assert result is None or isinstance(result, Trajectory)
