"""
Unit tests for reflective mutation operation module.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock
from src.variation.mutation import ReflectiveMutation
from src.core.trajectory import Trajectory


class TestReflectiveMutation:
    """Test suite for ReflectiveMutation class."""

    @pytest.fixture
    def mock_registry(self):
        """Create a mock model registry."""
        registry = MagicMock()
        registry.get_operator_model = MagicMock()
        return registry

    @pytest.fixture
    def mock_mutator(self):
        """Create a mock mutator model."""
        model = MagicMock()
        model.model_name = "mutator-model"
        model.generate_async = AsyncMock()
        return model

    @pytest.fixture
    def mutation(self, mock_registry):
        """Create a ReflectiveMutation instance."""
        return ReflectiveMutation(model_registry=mock_registry)

    @pytest.fixture
    def sample_trajectory(self):
        """Create a sample trajectory."""
        return Trajectory(
            query="What is the molecular weight of water?",
            answer="18.015 g/mol",
            reasoning="Water is H2O. H=1, O=16. So molecular weight = 2×1 + 16 = 18 g/mol.",
            source_model="model-a",
            generation_method="vanilla"
        )

    @pytest.mark.asyncio
    async def test_mutate_add_mode(self, mutation, mock_registry, mock_mutator, sample_trajectory):
        """Test additive mutation."""
        mock_registry.get_operator_model.return_value = mock_mutator

        mock_response = """[RESULT_START]
Water is H2O (2 hydrogen atoms, 1 oxygen atom).
Atomic weights: H=1.008 u, O=15.999 u.
Therefore, molecular weight of H2O = 2×1.008 + 15.999 = 18.015 g/mol.
This uses more precise atomic weights for accurate calculation.
[RESULT_END]"""

        mock_mutator.generate_async.return_value = mock_response

        result = await mutation.mutate(sample_trajectory, mode="add")

        assert isinstance(result, Trajectory)
        assert result.generation_method == "mutation_add"
        assert result.source_model == "mutator-model"
        assert "18.015" in result.reasoning

    @pytest.mark.asyncio
    async def test_mutate_delete_mode(self, mutation, mock_registry, mock_mutator, sample_trajectory):
        """Test deletive mutation."""
        mock_registry.get_operator_model.return_value = mock_mutator

        mock_response = """[RESULT_START]
Water is H2O with molecular weight = 18 g/mol (2×1 + 16).
[RESULT_END]"""

        mock_mutator.generate_async.return_value = mock_response

        result = await mutation.mutate(sample_trajectory, mode="delete")

        assert isinstance(result, Trajectory)
        assert result.generation_method == "mutation_delete"
        # Should be more concise
        assert len(result.reasoning) <= len(sample_trajectory.reasoning) + 100

    @pytest.mark.asyncio
    async def test_mutate_innovate_mode(self, mutation, mock_registry, mock_mutator, sample_trajectory):
        """Test innovative mutation (two-stage process)."""
        mock_registry.get_operator_model.return_value = mock_mutator

        # Stage 1: Diagnosis
        diagnosis_response = """[RESULT_START]
* Need to use more precise atomic weights
* Current calculation uses approximate values
[RESULT_END]"""

        # Stage 2 & 3: Innovative reasoning then deletion
        innovative_response = """[RESULT_START]
Using precise atomic weights H=1.008 and O=15.999:
H2O molecular weight = 2×1.008 + 15.999 = 18.015 g/mol.
[RESULT_END]"""

        mock_mutator.generate_async.side_effect = [diagnosis_response, innovative_response]

        result = await mutation.mutate(sample_trajectory, mode="innovate")

        assert isinstance(result, Trajectory)
        assert result.generation_method == "mutation_innovate"
        assert "18.015" in result.reasoning

    @pytest.mark.asyncio
    async def test_mutate_unknown_mode(self, mutation, mock_registry, mock_mutator, sample_trajectory):
        """Test mutation with unknown mode."""
        mock_registry.get_operator_model.return_value = mock_mutator

        result = await mutation.mutate(sample_trajectory, mode="unknown")

        # Should return None for unknown mode
        assert result is None

    @pytest.mark.asyncio
    async def test_mutate_exception_handling(self, mutation, mock_registry, mock_mutator, sample_trajectory):
        """Test mutation exception handling."""
        mock_registry.get_operator_model.return_value = mock_mutator
        mock_mutator.generate_async.side_effect = Exception("API error")

        result = await mutation.mutate(sample_trajectory, mode="add")

        # Should return None on error
        assert result is None

    @pytest.mark.asyncio
    async def test_innovative_mutation_fallback(self, mutation, mock_registry, mock_mutator, sample_trajectory):
        """Test innovative mutation falls back to deletive on error."""
        mock_registry.get_operator_model.return_value = mock_mutator

        # Stage 1 fails
        mock_mutator.generate_async.side_effect = [
            Exception("Diagnosis failed"),
            "Fallback: Water is H2O = 18 g/mol"
        ]

        result = await mutation.mutate(sample_trajectory, mode="innovate")

        # Should fall back to deletive mutation
        assert isinstance(result, Trajectory)
        # Result may not be perfect but should not crash


class TestMutationResultExtraction:
    """Test result and answer extraction methods."""

    @pytest.fixture
    def mutation(self, mock_registry):
        return ReflectiveMutation(model_registry=mock_registry)

    @pytest.fixture
    def mock_registry(self):
        return MagicMock()

    def test_extract_result_with_markers(self, mutation):
        """Test extracting result with markers."""
        response = """Some prefix text.

[RESULT_START]
This is the actual result content.
It should be extracted.
[RESULT_END]

Some suffix text."""

        result = mutation._extract_result(response)

        assert "This is the actual result content" in result
        assert "prefix" not in result
        assert "suffix" not in result

    def test_extract_result_without_markers(self, mutation):
        """Test extracting result without markers."""
        response = "Plain response without markers"

        result = mutation._extract_result(response)

        assert result == "Plain response without markers"

    def test_extract_result_empty(self, mutation):
        """Test extracting from empty response."""
        result = mutation._extract_result("")
        assert result == ""

    def test_extract_diagnosis_with_bullets(self, mutation):
        """Test extracting diagnosis with bullet points."""
        response = """Analysis:

[RESULT_START]
* Error 1: Used approximate values
* Error 2: Incorrect formula
* Suggestion: Use precise atomic weights
[RESULT_END]"""

        diagnosis = mutation._extract_diagnosis(response)

        assert "Used approximate values" in diagnosis
        assert "Incorrect formula" in diagnosis
        assert "precise atomic weights" in diagnosis

    def test_extract_diagnosis_without_bullets(self, mutation):
        """Test extracting diagnosis without bullet format."""
        response = "[RESULT_START]General advice text[RESULT_END]"

        diagnosis = mutation._extract_diagnosis(response)

        assert diagnosis == "General advice text"

    def test_extract_answer_from_reasoning(self, mutation):
        """Test answer extraction from reasoning."""
        reasoning = """Step 1: Analyze
Step 2: Calculate
The answer is 42.5 grams"""

        answer = mutation._extract_answer(reasoning)

        assert "42.5" in answer

    def test_extract_answer_last_line(self, mutation):
        """Test extracting last line as answer."""
        reasoning = "Line 1\nLine 2\nFinal answer: 3.14"

        answer = mutation._extract_answer(reasoning)

        assert "3.14" in answer

    def test_extract_answer_empty(self, mutation):
        """Test extracting answer from empty reasoning."""
        answer = mutation._extract_answer("")

        assert answer == ""


class TestMutationModes:
    """Test specific mutation mode behaviors."""

    @pytest.fixture
    def mutation(self, mock_registry):
        return ReflectiveMutation(model_registry=mock_registry)

    @pytest.fixture
    def mock_registry(self):
        return MagicMock()

    @pytest.fixture
    def mock_mutator(self):
        model = MagicMock()
        model.model_name = "test-model"
        model.generate_async = AsyncMock()
        return model

    @pytest.mark.asyncio
    async def test_additive_mutation_enriches_detail(self, mutation, mock_mutator):
        """Test that additive mutation adds more detail."""
        trajectory = Trajectory(
            query="What is photosynthesis?",
            answer="Energy conversion",
            reasoning="Photosynthesis converts energy.",
            source_model="model"
        )

        mock_mutator.generate_async.return_value = """[RESULT_START]
Photosynthesis is the process used by plants to convert light energy into chemical energy.
It involves chlorophyll absorbing light, water splitting, and carbon dioxide fixation.
The process produces glucose and oxygen as byproducts.
[RESULT_END]"""

        result = await mutation._additive_mutation(trajectory, mock_mutator)

        # Should be more detailed
        assert len(result.reasoning) > len(trajectory.reasoning)
        assert "chlorophyll" in result.reasoning.lower() or "glucose" in result.reasoning.lower()

    @pytest.mark.asyncio
    async def test_deletive_mutation_removes_redundancy(self, mutation, mock_mutator):
        """Test that deletive mutation makes reasoning more concise."""
        trajectory = Trajectory(
            query="Calculate 2+2",
            answer="4",
            reasoning="""Let me think about this problem.
I need to calculate 2+2.
First, I consider what 2+2 means.
Then I perform the addition.
The result is 4.
So the answer is 4.""",
            source_model="model"
        )

        mock_mutator.generate_async.return_value = """[RESULT_START]
2+2 = 4.
[RESULT_END]"""

        result = await mutation._deletive_mutation(trajectory, mock_mutator)

        # Should be more concise
        assert len(result.reasoning) < len(trajectory.reasoning)

    @pytest.mark.asyncio
    async def test_innovative_mutation_two_stage(self, mutation, mock_mutator):
        """Test that innovative mutation performs two-stage mutation."""
        trajectory = Trajectory(
            query="What is pH of 0.1 M HCl?",
            answer="1.0",
            reasoning="pH = -log(0.1) = 1",
            source_model="model"
        )

        # Mock responses for diagnosis and then innovative generation
        mock_mutator.generate_async.side_effect = [
            "[RESULT_START]* Missing step: explain strong acid dissociation[RESULT_END]",
            "[RESULT_START]HCl is strong acid, fully dissociates. pH = -log[H+] = -log(0.1) = 1.0.[RESULT_END]"
        ]

        result = await mutation._innovative_mutation(trajectory, mock_mutator)

        assert isinstance(result, Trajectory)
        # Verify both stages were called
        assert mock_mutator.generate_async.call_count == 2


class TestMutationMetadata:
    """Test metadata tracking in mutations."""

    @pytest.fixture
    def mutation(self, mock_registry):
        return ReflectiveMutation(model_registry=mock_registry)

    @pytest.fixture
    def mock_registry(self):
        return MagicMock()

    @pytest.fixture
    def mock_mutator(self):
        model = MagicMock()
        model.model_name = "test-model"
        model.generate_async = AsyncMock()
        return model

    @pytest.mark.asyncio
    async def test_metadata_contains_parent(self, mutation, mock_registry, mock_mutator):
        """Test that mutation metadata includes parent ID."""
        trajectory = Trajectory(
            query="Test",
            answer="Answer",
            reasoning="Reasoning",
            source_model="model",
            generation_method="vanilla"
        )
        parent_id = trajectory.id

        mock_registry.get_operator_model.return_value = mock_mutator
        mock_mutator.generate_async.return_value = "[RESULT_START]New reasoning[RESULT_END]"

        result = await mutation.mutate(trajectory, mode="add")

        assert "parent" in result.metadata
        assert result.metadata["parent"] == parent_id

    @pytest.mark.asyncio
    async def test_innovative_mutation_metadata_diagnosis(self, mutation, mock_registry, mock_mutator):
        """Test that innovative mutation includes diagnosis in metadata."""
        trajectory = Trajectory(
            query="Test",
            answer="Answer",
            reasoning="Reasoning",
            source_model="model"
        )

        mock_registry.get_operator_model.return_value = mock_mutator
        mock_mutator.generate_async.side_effect = [
            "[RESULT_START]* Error identified in reasoning* Need to correct calculation[RESULT_END]",
            "[RESULT_START]Improved reasoning[RESULT_END]"
        ]

        result = await mutation.mutate(trajectory, mode="innovate")

        assert "diagnosis" in result.metadata
        assert len(result.metadata["diagnosis"]) <= 200  # Should be truncated
