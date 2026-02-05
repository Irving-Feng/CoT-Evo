"""
Unit tests for multi-thinker generator module.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from src.initialization.generators import MultiThinkerGenerator
from src.core.population import Population
from src.core.trajectory import Trajectory


class TestMultiThinkerGenerator:
    """Test suite for MultiThinkerGenerator class."""

    @pytest.fixture
    def mock_registry(self):
        """Create a mock model registry."""
        registry = MagicMock()
        registry.get_random_thinker = MagicMock()
        return registry

    @pytest.fixture
    def mock_knowledge_augmenter(self):
        """Create a mock knowledge augmenter."""
        augmenter = MagicMock()
        augmenter.generate_knowledge = AsyncMock()
        return augmenter

    @pytest.fixture
    def mock_model(self):
        """Create a mock LLM model."""
        model = MagicMock()
        model.model_name = "test-model"
        model.generate_async = AsyncMock()
        return model

    @pytest.fixture
    def generator(self, mock_registry, mock_knowledge_augmenter):
        """Create a MultiThinkerGenerator instance."""
        return MultiThinkerGenerator(
            model_registry=mock_registry,
            knowledge_augmenter=mock_knowledge_augmenter,
            dataset_name="SciKnowEval"
        )

    @pytest.mark.asyncio
    async def test_generate_initial_pool_success(self, generator, mock_registry, mock_knowledge_augmenter, mock_model):
        """Test successful initial pool generation."""
        query = "What is the molecular weight of water?"
        ground_truth = "18.015 g/mol"

        # Setup mock models
        mock_registry.get_random_thinker.return_value = mock_model

        # Setup mock model responses
        mock_model.generate_async.return_value = "The molecular weight of H2O is 18.015 g/mol.\nAnswer: 18.015 g/mol"

        # Setup mock knowledge
        mock_knowledge_augmenter.generate_knowledge.return_value = "Water is H2O with molecular weight 18.015 g/mol"

        # Generate initial pool
        population = await generator.generate_initial_pool(
            query=query,
            ground_truth=ground_truth,
            n_vanilla=2,
            n_knowledge_augmented=1
        )

        # Verify
        assert isinstance(population, Population)
        assert population.size == 3  # 2 vanilla + 1 knowledge-augmented

    @pytest.mark.asyncio
    async def test_generate_initial_pool_vanilla_only(self, generator, mock_registry, mock_model):
        """Test generation with only vanilla trajectories."""
        query = "Test question"
        ground_truth = "Test answer"

        mock_registry.get_random_thinker.return_value = mock_model
        mock_model.generate_async.return_value = "Reasoning...\nAnswer: Test answer"

        population = await generator.generate_initial_pool(
            query=query,
            ground_truth=ground_truth,
            n_vanilla=3,
            n_knowledge_augmented=0
        )

        assert population.size == 3

    @pytest.mark.asyncio
    async def test_generate_initial_pool_knowledge_only(self, generator, mock_registry, mock_knowledge_augmenter, mock_model):
        """Test generation with only knowledge-augmented trajectories."""
        query = "Test question"
        ground_truth = "Test answer"

        mock_registry.get_random_thinker.return_value = mock_model
        mock_model.generate_async.return_value = "Reasoning...\nAnswer: Test answer"
        mock_knowledge_augmenter.generate_knowledge.return_value = "Domain knowledge"

        population = await generator.generate_initial_pool(
            query=query,
            ground_truth=ground_truth,
            n_vanilla=0,
            n_knowledge_augmented=2
        )

        assert population.size == 2

    @pytest.mark.asyncio
    async def test_generate_vanilla_trajectory(self, generator, mock_model):
        """Test _generate_vanilla method."""
        query = "What is photosynthesis?"

        mock_model.generate_async.return_value = """Photosynthesis is the process by which plants convert light energy into chemical energy.

The process involves:
1. Light absorption by chlorophyll
2. Water splitting
3. Carbon dioxide fixation
4. Glucose production

Answer: Energy conversion process in plants"""

        trajectory = await generator._generate_vanilla(mock_model, query, None)

        assert isinstance(trajectory, Trajectory)
        assert trajectory.query == query
        assert trajectory.source_model == "test-model"
        assert trajectory.generation_method == "vanilla"
        assert "photosynthesis" in trajectory.reasoning.lower()

    @pytest.mark.asyncio
    async def test_generate_knowledge_augmented_trajectory(self, generator, mock_model):
        """Test _generate_knowledge_augmented method."""
        query = "What is the molecular weight of water?"
        knowledge = "Water is H2O with atomic weights: H=1.008, O=15.999"

        mock_model.generate_async.return_value = f"""Using the knowledge that {knowledge}

We can calculate:
H2O = 2×1.008 + 15.999 = 18.015 g/mol

Answer: 18.015 g/mol"""

        trajectory = await generator._generate_knowledge_augmented(
            mock_model, query, knowledge, None
        )

        assert isinstance(trajectory, Trajectory)
        assert trajectory.query == query
        assert trajectory.knowledge == knowledge
        assert trajectory.generation_method == "knowledge_augmented"

    @pytest.mark.asyncio
    async def test_extract_answer(self, generator):
        """Test answer extraction from reasoning."""
        reasoning = """Step 1: Analyze the problem
Step 2: Calculate the result
The final answer is 42"""

        answer = generator._extract_answer(reasoning)
        assert "42" in answer

    @pytest.mark.asyncio
    async def test_extract_answer_with_prefix(self, generator):
        """Test answer extraction with common prefixes."""
        reasoning = "After calculation, the answer is: 3.14159"

        answer = generator._extract_answer(reasoning)
        assert "3.14159" in answer

    @pytest.mark.asyncio
    async def test_extract_answer_empty(self, generator):
        """Test answer extraction from empty reasoning."""
        answer = generator._extract_answer("")
        assert answer == ""

    @pytest.mark.asyncio
    async def test_knowledge_generation_failure(self, generator, mock_registry, mock_knowledge_augmenter, mock_model):
        """Test handling of knowledge generation failure."""
        mock_registry.get_random_thinker.return_value = mock_model
        mock_model.generate_async.return_value = "Reasoning...\nAnswer: Answer"
        mock_knowledge_augmenter.generate_knowledge.side_effect = Exception("Knowledge generation failed")

        population = await generator.generate_initial_pool(
            query="Test",
            ground_truth="Answer",
            n_vanilla=2,
            n_knowledge_augmented=3
        )

        # Should have only vanilla trajectories
        assert population.size == 2

    @pytest.mark.asyncio
    async def test_parallel_generation(self, generator, mock_registry, mock_knowledge_augmenter, mock_model):
        """Test that generation happens in parallel."""
        query = "Test query"
        ground_truth = "Test answer"

        # Setup to return different models
        models = [MagicMock(model_name=f"model-{i}") for i in range(5)]
        for model in models:
            model.generate_async = AsyncMock(return_value="Reasoning...\nAnswer: Answer")

        mock_registry.get_random_thinker.side_effect = models
        mock_knowledge_augmenter.generate_knowledge.return_value = "Knowledge"

        # Generate pool
        population = await generator.generate_initial_pool(
            query=query,
            ground_truth=ground_truth,
            n_vanilla=3,
            n_knowledge_augmented=2
        )

        assert population.size == 5


class TestMultiThinkerGeneratorWithCustomPrompts:
    """Test MultiThinkerGenerator with custom prompt templates."""

    @pytest.fixture
    def generator(self, mock_registry, mock_knowledge_augmenter):
        """Create generator with custom prompts."""
        return MultiThinkerGenerator(
            model_registry=mock_registry,
            knowledge_augmenter=mock_knowledge_augmenter,
            dataset_name="ChemCoTDataset"
        )

    @pytest.fixture
    def mock_registry(self):
        """Create mock registry."""
        registry = MagicMock()
        registry.get_random_thinker = MagicMock()
        return registry

    @pytest.fixture
    def mock_knowledge_augmenter(self):
        """Create mock knowledge augmenter."""
        augmenter = MagicMock()
        augmenter.generate_knowledge = AsyncMock()
        return augmenter

    @pytest.mark.asyncio
    async def test_custom_system_prompt(self, generator, mock_registry):
        """Test that dataset-specific system prompt is used."""
        # The generator should have chemistry-focused system prompt
        assert "chemistry" in generator.system_prompt.lower()

    @pytest.mark.asyncio
    async def test_custom_stop_sequences(self, generator):
        """Test that dataset-specific stop sequences are configured."""
        # ChemCoTDataset should have stop sequences configured
        assert isinstance(generator.stop_sequences, list)


class TestMultiThinkerGeneratorAnswerExtraction:
    """Test answer extraction methods for different formats."""

    @pytest.fixture
    def generator(self, mock_registry, mock_knowledge_augmenter):
        """Create generator for testing."""
        return MultiThinkerGenerator(
            model_registry=mock_registry,
            knowledge_augmenter=mock_knowledge_augmenter
        )

    @pytest.fixture
    def mock_registry(self):
        return MagicMock()

    @pytest.fixture
    def mock_knowledge_augmenter(self):
        return MagicMock()

    def test_extract_answer_json_format(self, generator):
        """Test extraction from JSON format (ChemCoT)."""
        reasoning = """Some reasoning...

{"result": "42.5"}

Final output."""

        answer = generator._extract_answer_json(reasoning)
        assert "42.5" in answer

    def test_extract_answer_json_no_match(self, generator):
        """Test JSON extraction with no JSON pattern."""
        reasoning = "No JSON here, just text: The answer is 42"

        answer = generator._extract_answer_json(reasoning)
        # Should fallback to simple extraction
        assert "42" in answer

    def test_extract_answer_bio_format(self, generator):
        """Test extraction from BioProBench format."""
        reasoning = """Reasoning steps...

[ANSWER_START]
The correct answer is mitochondria.
[ANSWER_END]"""

        answer = generator._extract_answer_bio(reasoning)
        assert "mitochondria" in answer

    def test_extract_answer_bio_no_markers(self, generator):
        """Test bio extraction without markers."""
        reasoning = "Some reasoning...\nThe answer is ATP"

        answer = generator._extract_answer_bio(reasoning)
        assert "ATP" in answer

    def test_extract_answer_bio_fallback(self, generator):
        """Test bio extraction fallback to simple extraction."""
        reasoning = "[ANSWER_START]\nPartial answer without end marker"

        answer = generator._extract_answer_bio(reasoning)
        # Should handle missing end marker
        assert len(answer) > 0
