"""
Unit tests for fitness evaluation classes.
"""

import pytest
from src.core.fitness import (
    FitnessEvaluator,
    LengthEvaluator,
    ExactMatchEvaluator,
    KnowledgeJudgeEvaluator,
)
from src.core.trajectory import Trajectory


class TestLengthEvaluator:
    """Test cases for LengthEvaluator class."""

    def test_score_too_short(self):
        """Test score for reasoning that's too short."""
        evaluator = LengthEvaluator(lower_percentile=100, upper_percentile=500)

        score = evaluator.score("This is short")

        assert score == 0.0

    def test_score_just_right(self):
        """Test score for reasoning with appropriate length."""
        evaluator = LengthEvaluator(lower_percentile=50, upper_percentile=200)

        # Create reasoning with >50 words and <200 words
        reasoning = "This is a reasonably long reasoning that falls within the acceptable range " + "word " * 60
        word_count = len(reasoning.split())
        assert 50 < word_count < 200, f"Reasoning has {word_count} words, expected between 50 and 200"

        score = evaluator.score(reasoning)

        assert score == 1.0

    def test_score_too_long(self):
        """Test score for reasoning that's too long."""
        evaluator = LengthEvaluator(lower_percentile=50, upper_percentile=100)

        long_reasoning = "word " * 150  # 150 words
        score = evaluator.score(long_reasoning)

        assert score == 0.5

    def test_score_boundary_lower(self):
        """Test score at lower boundary."""
        evaluator = LengthEvaluator(lower_percentile=10, upper_percentile=100)

        reasoning = "word " * 10  # Exactly at boundary
        score = evaluator.score(reasoning)

        assert score == 1.0  # At boundary is acceptable

    def test_score_boundary_upper(self):
        """Test score at upper boundary."""
        evaluator = LengthEvaluator(lower_percentile=10, upper_percentile=100)

        reasoning = "word " * 100  # Exactly at boundary
        score = evaluator.score(reasoning)

        assert score == 1.0  # At boundary is acceptable


class TestExactMatchEvaluator:
    """Test cases for ExactMatchEvaluator class."""

    @pytest.mark.asyncio
    async def test_exact_match_strict_true(self):
        """Test strict exact matching with identical answers."""
        evaluator = ExactMatchEvaluator(strict=True)

        result = await evaluator.match("42", "42")

        assert result is True

    @pytest.mark.asyncio
    async def test_exact_match_strict_false(self):
        """Test strict exact matching with different answers."""
        evaluator = ExactMatchEvaluator(strict=True)

        result = await evaluator.match("42", "43")

        assert result is False

    @pytest.mark.asyncio
    async def test_exact_match_with_whitespace(self):
        """Test exact matching with different whitespace."""
        evaluator = ExactMatchEvaluator(strict=True)

        result = await evaluator.match("  42  ", "42")

        assert result is True  # Normalized

    @pytest.mark.asyncio
    async def test_exact_match_lenient(self):
        """Test lenient matching."""
        evaluator = ExactMatchEvaluator(strict=False)

        result = await evaluator.match("4 2", "42")

        assert result is True  # Whitespace removed

    @pytest.mark.asyncio
    async def test_exact_match_case_insensitive(self):
        """Test case-insensitive matching."""
        evaluator = ExactMatchEvaluator(strict=True)

        result = await evaluator.match("Answer", "answer")

        assert result is True  # Normalized to lowercase

    @pytest.mark.asyncio
    async def test_exact_match_with_prefix(self):
        """Test matching with answer prefix."""
        evaluator = ExactMatchEvaluator(strict=True)

        result = await evaluator.match("Answer: 42", "42")

        assert result is True  # Prefix removed


class TestFitnessEvaluator:
    """Test cases for FitnessEvaluator class."""

    def test_initialization(self):
        """Test fitness evaluator initialization."""
        evaluator = FitnessEvaluator(
            lambda_length=0.3,
            lambda_knowledge=0.5
        )

        assert evaluator.lambda_length == 0.3
        assert evaluator.lambda_knowledge == 0.5

    def test_compute_fitness_from_scores_all_components(self):
        """Test computing fitness from all three components."""
        evaluator = FitnessEvaluator(
            lambda_length=0.3,
            lambda_knowledge=0.5,
            use_exact_match=True,
            use_length_score=True,
            use_knowledge_score=True,
        )

        fitness = evaluator.compute_fitness_from_scores(
            exact_match=True,
            length_score=1.0,
            knowledge_score=5
        )

        # R(t) = 1.0 + 0.3 * 1.0 + 0.5 * (5/5) = 1.0 + 0.3 + 0.5 = 1.8
        assert fitness == pytest.approx(1.8)

    def test_compute_fitness_from_scores_no_exact_match(self):
        """Test fitness computation when answer is incorrect."""
        evaluator = FitnessEvaluator(
            lambda_length=0.3,
            lambda_knowledge=0.5
        )

        fitness = evaluator.compute_fitness_from_scores(
            exact_match=False,
            length_score=1.0,
            knowledge_score=4
        )

        # R(t) = 0.0 + 0.3 * 1.0 + 0.5 * (4/5) = 0.0 + 0.3 + 0.4 = 0.7
        assert fitness == pytest.approx(0.7)

    def test_compute_fitness_from_scores_poor_length(self):
        """Test fitness computation with poor length score."""
        evaluator = FitnessEvaluator(
            lambda_length=0.3,
            lambda_knowledge=0.5
        )

        fitness = evaluator.compute_fitness_from_scores(
            exact_match=True,
            length_score=0.0,  # Too short
            knowledge_score=5
        )

        # R(t) = 1.0 + 0.3 * 0.0 + 0.5 * 1.0 = 1.0 + 0.0 + 0.5 = 1.5
        assert fitness == pytest.approx(1.5)

    def test_compute_fitness_from_scores_no_knowledge(self):
        """Test fitness computation without knowledge score."""
        evaluator = FitnessEvaluator(
            lambda_length=0.3,
            lambda_knowledge=0.5,
            use_knowledge_score=False
        )

        fitness = evaluator.compute_fitness_from_scores(
            exact_match=True,
            length_score=1.0,
            knowledge_score=None
        )

        # R(t) = 1.0 + 0.3 * 1.0 = 1.3 (knowledge not used)
        assert fitness == pytest.approx(1.3)

    def test_set_evaluators(self):
        """Test setting individual evaluators."""
        evaluator = FitnessEvaluator()

        length_eval = LengthEvaluator(100, 500)
        exact_eval = ExactMatchEvaluator()
        knowledge_eval = None  # Mock

        evaluator.set_length_evaluator(length_eval)
        evaluator.set_exact_match_evaluator(exact_eval)

        assert evaluator.length_evaluator is length_eval
        assert evaluator.exact_match_evaluator is exact_eval


class TestKnowledgeJudgeEvaluator:
    """Test cases for KnowledgeJudgeEvaluator class."""

    def test_initialization(self):
        """Test knowledge judge initialization."""
        mock_model = None  # Would be a real LLM in practice
        evaluator = KnowledgeJudgeEvaluator(mock_model)

        assert evaluator.judge_model is mock_model

    def test_construct_judge_prompt(self):
        """Test prompt construction for knowledge judgment."""
        mock_model = None
        evaluator = KnowledgeJudgeEvaluator(mock_model)

        prompt = evaluator._construct_judge_prompt(
            reasoning="The molecular weight is calculated by...",
            reference_knowledge="Molecular weight = sum of atomic weights"
        )

        assert "Molecular weight = sum of atomic weights" in prompt
        assert "The molecular weight is calculated by..." in prompt
        assert "1-5" in prompt

    def test_extract_score_valid(self):
        """Test extracting valid score from response."""
        mock_model = None
        evaluator = KnowledgeJudgeEvaluator(mock_model)

        # Test different formats
        assert evaluator._extract_score("4") == 4
        assert evaluator._extract_score("The score is 5") == 5
        assert evaluator._extract_score("I give this a 3/5") == 3

    def test_extract_score_invalid(self):
        """Test extracting score from invalid response."""
        mock_model = None
        evaluator = KnowledgeJudgeEvaluator(mock_model)

        # Should return default score of 3
        assert evaluator._extract_score("No score here") == 3

    def test_extract_score_out_of_range(self):
        """Test extracting score that's out of range."""
        mock_model = None
        evaluator = KnowledgeJudgeEvaluator(mock_model)

        # extract_score clamps values to [1, 5] range
        assert evaluator._extract_score("0") == 1  # Too low, clamped to 1
        assert evaluator._extract_score("10") == 5  # Too high, clamped to 5
