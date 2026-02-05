"""
Unit tests for Trajectory class.
"""

import pytest
import numpy as np
from src.core.trajectory import Trajectory


class TestTrajectory:
    """Test cases for Trajectory class."""

    def test_trajectory_creation(self):
        """Test creating a basic trajectory."""
        traj = Trajectory(
            query="What is 2+2?",
            answer="4",
            reasoning="To solve 2+2, I simply add the numbers together: 2 + 2 = 4.",
            source_model="test-model",
            generation_method="vanilla"
        )

        assert traj.query == "What is 2+2?"
        assert traj.answer == "4"
        assert traj.source_model == "test-model"
        assert traj.generation_method == "vanilla"
        assert traj.id is not None
        assert len(traj.id) == 16

    def test_trajectory_with_knowledge(self):
        """Test creating a trajectory with knowledge augmentation."""
        traj = Trajectory(
            query="Test question",
            answer="Test answer",
            reasoning="Test reasoning",
            knowledge="Additional domain knowledge",
            source_model="test-model",
            generation_method="knowledge_augmented"
        )

        assert traj.knowledge == "Additional domain knowledge"
        assert traj.generation_method == "knowledge_augmented"

    def test_fitness_score_management(self):
        """Test setting and getting fitness scores."""
        traj = Trajectory(
            query="Test",
            answer="Answer",
            reasoning="Reasoning",
            source_model="test"
        )

        # Initially no fitness score
        assert traj.fitness_score is None

        # Set fitness score
        traj.set_fitness_score(0.85)
        assert traj.fitness_score == 0.85
        assert traj._fitness_score == 0.85

    def test_exact_match_management(self):
        """Test setting and getting exact match results."""
        traj = Trajectory(
            query="Test",
            answer="Answer",
            reasoning="Reasoning",
            source_model="test"
        )

        # Initially no exact match
        assert traj.exact_match is None

        # Set correct
        traj.set_exact_match(True)
        assert traj.exact_match is True
        assert traj.is_correct() is True

        # Set incorrect
        traj.set_exact_match(False)
        assert traj.exact_match is False
        assert traj.is_correct() is False

    def test_length_score_management(self):
        """Test setting and getting length scores."""
        traj = Trajectory(
            query="Test",
            answer="Answer",
            reasoning="Reasoning",
            source_model="test"
        )

        # Test all possible scores
        for score in [0.0, 0.5, 1.0]:
            traj.set_length_score(score)
            assert traj.length_score == score

    def test_knowledge_score_validation(self):
        """Test knowledge score validation (must be 1-5)."""
        traj = Trajectory(
            query="Test",
            answer="Answer",
            reasoning="Reasoning",
            source_model="test"
        )

        # Valid scores
        for score in [1, 2, 3, 4, 5]:
            traj.set_knowledge_score(score)
            assert traj.knowledge_score == score

        # Invalid scores should raise ValueError
        with pytest.raises(ValueError):
            traj.set_knowledge_score(0)

        with pytest.raises(ValueError):
            traj.set_knowledge_score(6)

    def test_embedding_management(self):
        """Test setting and getting embeddings."""
        traj = Trajectory(
            query="Test",
            answer="Answer",
            reasoning="Reasoning",
            source_model="test"
        )

        # Initially no embedding
        assert traj.embedding is None
        assert not traj.has_embedding()

        # Set embedding
        embedding = np.random.rand(1024)
        traj.set_embedding(embedding)

        assert traj.embedding is not None
        assert traj.has_embedding()
        assert np.array_equal(traj.embedding, embedding)

    def test_novelty_and_local_competition(self):
        """Test setting NSLC-related scores."""
        traj = Trajectory(
            query="Test",
            answer="Answer",
            reasoning="Reasoning",
            source_model="test"
        )

        traj.set_novelty_score(0.75)
        traj.set_local_competition(0.30)

        assert traj.novelty_score == 0.75
        assert traj.local_competition == 0.30

    def test_reasoning_length(self):
        """Test reasoning length estimation."""
        traj = Trajectory(
            query="Test",
            answer="Answer",
            reasoning="This is a test reasoning with multiple words.",
            source_model="test"
        )

        # Count words
        expected_length = len("This is a test reasoning with multiple words.".split())
        assert traj.reasoning_length() == expected_length

    def test_evaluation_status(self):
        """Test is_evaluated method."""
        traj = Trajectory(
            query="Test",
            answer="Answer",
            reasoning="Reasoning",
            source_model="test"
        )

        # Not evaluated initially
        assert not traj.is_evaluated()

        # After setting fitness score, it's evaluated
        traj.set_fitness_score(0.8)
        assert traj.is_evaluated()

    def test_to_dict(self):
        """Test converting trajectory to dictionary."""
        traj = Trajectory(
            query="What is 2+2?",
            answer="4",
            reasoning="2 + 2 = 4",
            knowledge="Math knowledge",
            source_model="test-model",
            generation_method="knowledge_augmented"
        )

        traj.set_fitness_score(1.0)
        traj.set_exact_match(True)
        traj.set_length_score(1.0)
        traj.set_knowledge_score(5)

        data = traj.to_dict()

        assert data["query"] == "What is 2+2?"
        assert data["answer"] == "4"
        assert data["source_model"] == "test-model"
        assert data["knowledge"] == "Math knowledge"
        assert data["_fitness_score"] == 1.0
        assert data["_exact_match"] is True
        assert data["_length_score"] == 1.0
        assert data["_knowledge_score"] == 5

    def test_from_dict(self):
        """Test creating trajectory from dictionary."""
        data = {
            "query": "Test question",
            "answer": "Test answer",
            "reasoning": "Test reasoning",
            "knowledge": "Test knowledge",
            "source_model": "test-model",
            "generation_method": "crossover",
            "_fitness_score": 0.85,
            "_exact_match": False,
            "_length_score": 0.5,
            "_knowledge_score": 3,
            "_novelty_score": 0.7,
            "_local_competition": 0.4,
            "metadata": {"id": "test123"}
        }

        traj = Trajectory.from_dict(data)

        assert traj.query == "Test question"
        assert traj.answer == "Test answer"
        assert traj.source_model == "test-model"
        assert traj.generation_method == "crossover"
        assert traj.fitness_score == 0.85
        assert traj.exact_match is False
        assert traj.length_score == 0.5
        assert traj.knowledge_score == 3
        assert traj.novelty_score == 0.7
        assert traj.local_competition == 0.4

    def test_from_dict_with_embedding(self):
        """Test creating trajectory with embedding from dictionary."""
        embedding_list = [0.1, 0.2, 0.3, 0.4]
        data = {
            "query": "Test",
            "answer": "Answer",
            "reasoning": "Reasoning",
            "source_model": "test",
            "_embedding": embedding_list
        }

        traj = Trajectory.from_dict(data)

        assert traj.embedding is not None
        assert isinstance(traj.embedding, np.ndarray)
        assert np.array_equal(traj.embedding, np.array(embedding_list))

    def test_repr(self):
        """Test string representation."""
        traj = Trajectory(
            query="Test",
            answer="Answer",
            reasoning="Reasoning",
            source_model="test-model"
        )
        traj.set_fitness_score(0.9)
        traj.set_exact_match(True)

        repr_str = repr(traj)

        assert "Trajectory" in repr_str
        assert "test-model" in repr_str
        assert "0.9" in repr_str
        assert "True" in repr_str

    def test_metadata_id_generation(self):
        """Test that unique IDs are generated."""
        traj1 = Trajectory(
            query="Question 1",
            answer="Answer 1",
            reasoning="Reasoning 1",
            source_model="test"
        )

        traj2 = Trajectory(
            query="Question 2",
            answer="Answer 2",
            reasoning="Reasoning 2",
            source_model="test"
        )

        # Different content should generate different IDs
        assert traj1.id != traj2.id

    def test_same_content_same_id(self):
        """Test that same content generates same ID."""
        query = "Test question"
        answer = "Test answer"
        reasoning = "Test reasoning"

        traj1 = Trajectory(
            query=query,
            answer=answer,
            reasoning=reasoning,
            source_model="test"
        )

        traj2 = Trajectory(
            query=query,
            answer=answer,
            reasoning=reasoning,
            source_model="test"
        )

        # Same content should generate same ID
        assert traj1.id == traj2.id
