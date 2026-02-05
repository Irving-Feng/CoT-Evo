"""
Unit tests for NSLC selection algorithm module.
"""

import pytest
import numpy as np
from unittest.mock import AsyncMock, MagicMock, patch
from src.selection.nslc import NSLCSelector
from src.core.population import Population
from src.core.trajectory import Trajectory


class TestNSLCSelector:
    """Test suite for NSLCSelector class."""

    @pytest.fixture
    def mock_registry(self):
        """Create a mock model registry."""
        registry = MagicMock()
        registry.get_embedding_model = MagicMock()
        return registry

    @pytest.fixture
    def mock_embedding_model(self):
        """Create a mock embedding model."""
        model = MagicMock()
        model.embed_batch_async = AsyncMock()
        return model

    @pytest.fixture
    def selector(self, mock_registry, mock_embedding_model):
        """Create an NSLCSelector instance."""
        mock_registry.get_embedding_model.return_value = mock_embedding_model
        return NSLCSelector(
            model_registry=mock_registry,
            n_neighbors=3,
            epsilon=0.1
        )

    @pytest.fixture
    def sample_population(self):
        """Create a sample population for testing."""
        trajectories = [
            Trajectory(
                query="What is 2+2?",
                answer="4",
                reasoning="2+2=4",
                source_model="model-a",
                generation_method="vanilla"
            ),
            Trajectory(
                query="What is 2+2?",
                answer="5",
                reasoning="2+2=5",
                source_model="model-b",
                generation_method="vanilla"
            ),
            Trajectory(
                query="What is 2+2?",
                answer="4",
                reasoning="Let me calculate: 2+2=4",
                source_model="model-c",
                generation_method="knowledge_augmented"
            )
        ]
        return Population(trajectories)

    @pytest.mark.asyncio
    async def test_select_parents_empty_population(self, selector):
        """Test parent selection from empty population."""
        population = Population([])

        parents = await selector.select_parents(population, n_parents=2)

        assert parents == []

    @pytest.mark.asyncio
    async def test_select_parents_small_population(self, selector, sample_population):
        """Test parent selection when population size <= n_parents."""
        parents = await selector.select_parents(sample_population, n_parents=5)

        assert len(parents) == sample_population.size

    @pytest.mark.asyncio
    async def test_select_parents_compute_embeddings(self, selector, sample_population, mock_embedding_model):
        """Test that embeddings are computed during selection."""
        # Mock embeddings
        embeddings = [
            np.array([0.1, 0.2, 0.3]),
            np.array([0.4, 0.5, 0.6]),
            np.array([0.7, 0.8, 0.9])
        ]
        mock_embedding_model.embed_batch_async.return_value = embeddings

        await selector.select_parents(sample_population, n_parents=2)

        # Verify embeddings were computed and stored
        for traj in sample_population.trajectories:
            assert traj.has_embedding()

    @pytest.mark.asyncio
    async def test_select_parents_no_embedding_model(self, mock_registry, sample_population):
        """Test parent selection when no embedding model is available."""
        mock_registry.get_embedding_model.return_value = None
        selector = NSLCSelector(mock_registry)

        # Should not raise error, but log warning
        parents = await selector.select_parents(sample_population, n_parents=2)

        # Should return some parents (random sampling)
        assert len(parents) == 2

    @pytest.mark.asyncio
    async def test_compute_novelty_and_local_competition(self, selector, sample_population):
        """Test novelty and local competition score computation."""
        # Manually set embeddings
        embeddings = [
            np.array([0.0, 0.0, 0.0]),
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0])
        ]
        for traj, emb in zip(sample_population.trajectories, embeddings):
            traj.set_embedding(emb)

        # Set fitness scores
        sample_population.trajectories[0]._fitness_score = 1.0
        sample_population.trajectories[1]._fitness_score = 0.5
        sample_population.trajectories[2]._fitness_score = 0.8

        selector._compute_novelty_and_local_competition(sample_population)

        # Check that scores were computed
        for traj in sample_population.trajectories:
            assert traj.novelty_score is not None
            assert traj.local_competition is not None


class TestNSLCNoveltyComputation:
    """Test novelty score computation."""

    @pytest.fixture
    def selector(self, mock_registry, mock_embedding_model):
        mock_registry.get_embedding_model.return_value = mock_embedding_model
        return NSLCSelector(
            model_registry=mock_registry,
            n_neighbors=2
        )

    @pytest.fixture
    def mock_registry(self):
        return MagicMock()

    @pytest.fixture
    def mock_embedding_model(self):
        return MagicMock()

    def test_compute_novelty_score(self, selector):
        """Test novelty score calculation."""
        # Create target and neighbors
        target = Trajectory(
            query="Test",
            answer="4",
            reasoning="Test",
            source_model="model"
        )
        target.set_embedding(np.array([0.0, 0.0, 0.0]))

        neighbor1 = Trajectory(
            query="Test",
            answer="5",
            reasoning="Test",
            source_model="model"
        )
        neighbor1.set_embedding(np.array([1.0, 0.0, 0.0]))

        neighbor2 = Trajectory(
            query="Test",
            answer="4",
            reasoning="Test",
            source_model="model"
        )
        neighbor2.set_embedding(np.array([0.0, 1.0, 0.0]))

        neighbors = [neighbor1, neighbor2]

        novelty = selector._compute_novelty_score(target, neighbors)

        # Novelty should be positive
        assert novelty > 0

    def test_compute_novelty_score_no_neighbors(self, selector):
        """Test novelty score with no neighbors."""
        target = Trajectory(
            query="Test",
            answer="4",
            reasoning="Test",
            source_model="model"
        )
        target.set_embedding(np.array([0.0, 0.0, 0.0]))

        novelty = selector._compute_novelty_score(target, [])

        assert novelty == 0.0


class TestNSLCLocalComputation:
    """Test local competition score computation."""

    @pytest.fixture
    def selector(self, mock_registry):
        mock_registry.get_embedding_model.return_value = MagicMock()
        return NSLCSelector(model_registry=mock_registry)

    @pytest.fixture
    def mock_registry(self):
        return MagicMock()

    def test_compute_local_competition_score(self, selector):
        """Test local competition score calculation."""
        target = Trajectory(
            query="Test",
            answer="4",
            reasoning="Test",
            source_model="model"
        )
        target._fitness_score = 0.8

        neighbor1 = Trajectory(
            query="Test",
            answer="5",
            reasoning="Test",
            source_model="model"
        )
        neighbor1._fitness_score = 0.5

        neighbor2 = Trajectory(
            query="Test",
            answer="4",
            reasoning="Test",
            source_model="model"
        )
        neighbor2._fitness_score = 0.6

        neighbors = [neighbor1, neighbor2]

        local_comp = selector._compute_local_competition_score(target, neighbors)

        # Target should have positive local competition (better than both neighbors)
        assert local_comp > 0

    def test_compute_local_competition_no_target_fitness(self, selector):
        """Test local competition when target has no fitness."""
        target = Trajectory(
            query="Test",
            answer="4",
            reasoning="Test",
            source_model="model"
        )
        # No fitness score set

        neighbor = Trajectory(
            query="Test",
            answer="5",
            reasoning="Test",
            source_model="model"
        )
        neighbor._fitness_score = 0.5

        local_comp = selector._compute_local_competition_score(target, [neighbor])

        assert local_comp == 0.0

    def test_compute_local_competition_worse_than_neighbors(self, selector):
        """Test local competition when target is worse than neighbors."""
        target = Trajectory(
            query="Test",
            answer="4",
            reasoning="Test",
            source_model="model"
        )
        target._fitness_score = 0.3

        neighbor = Trajectory(
            query="Test",
            answer="5",
            reasoning="Test",
            source_model="model"
        )
        neighbor._fitness_score = 0.8

        local_comp = selector._compute_local_competition_score(target, [neighbor])

        # Should be 0 (no advantage over neighbors)
        assert local_comp == 0.0


class TestNSLCKNearestNeighbors:
    """Test k-nearest neighbors computation."""

    @pytest.fixture
    def selector(self, mock_registry):
        return NSLCSelector(model_registry=mock_registry, n_neighbors=2)

    @pytest.fixture
    def mock_registry(self):
        return MagicMock()

    def test_get_k_neighbors(self, selector):
        """Test finding k nearest neighbors."""
        # Create target
        target = Trajectory(
            query="Test",
            answer="4",
            reasoning="Target reasoning",
            source_model="model"
        )
        target.set_embedding(np.array([0.0, 0.0]))

        # Create population
        trajectories = [target]
        embeddings = [
            np.array([1.0, 0.0]),  # Distance = 1.0
            np.array([0.0, 1.0]),  # Distance = 1.0
            np.array([2.0, 0.0]),  # Distance = 2.0
            np.array([0.0, 2.0]),  # Distance = 2.0
        ]

        for i, emb in enumerate(embeddings):
            traj = Trajectory(
                query="Test",
                answer=str(i),
                reasoning=f"Reasoning {i}",  # Unique reasoning for unique ID
                source_model="model"
            )
            traj.set_embedding(emb)
            trajectories.append(traj)

        population = Population(trajectories)

        neighbors = selector._get_k_neighbors(population, target)

        # Should return 2 nearest neighbors
        assert len(neighbors) == 2
        # Target should not be in neighbors
        assert target not in neighbors

    def test_get_k_neighbors_target_no_embedding(self, selector):
        """Test k-nearest neighbors when target has no embedding."""
        target = Trajectory(
            query="Test",
            answer="4",
            reasoning="Test",
            source_model="model"
        )
        # No embedding set

        other = Trajectory(
            query="Test",
            answer="5",
            reasoning="Test",
            source_model="model"
        )
        other.set_embedding(np.array([1.0, 0.0]))

        population = Population([target, other])

        neighbors = selector._get_k_neighbors(population, target)

        assert neighbors == []


class TestNSLCSampling:
    """Test probability-based sampling from Pareto front."""

    @pytest.fixture
    def selector(self, mock_registry):
        return NSLCSelector(model_registry=mock_registry, epsilon=0.1)

    @pytest.fixture
    def mock_registry(self):
        return MagicMock()

    def test_sample_by_probability(self, selector):
        """Test probability-based sampling."""
        # Create Pareto front with varying local competition scores
        traj1 = Trajectory(
            query="Test",
            answer="1",
            reasoning="Test",
            source_model="model"
        )
        traj1._local_competition = 0.8

        traj2 = Trajectory(
            query="Test",
            answer="2",
            reasoning="Test",
            source_model="model"
        )
        traj2._local_competition = 0.2

        traj3 = Trajectory(
            query="Test",
            answer="3",
            reasoning="Test",
            source_model="model"
        )
        traj3._local_competition = 0.5

        pareto_front = [traj1, traj2, traj3]

        # Sample 5 times
        selected = selector._sample_by_probability(pareto_front, n_samples=5)

        assert len(selected) == 5
        # All should be from pareto front
        for traj in selected:
            assert traj in pareto_front

    def test_sample_by_probability_zero_total(self, selector):
        """Test sampling when all local competition scores are zero."""
        traj1 = Trajectory(
            query="Test",
            answer="1",
            reasoning="Test",
            source_model="model"
        )
        traj1._local_competition = 0.0

        traj2 = Trajectory(
            query="Test",
            answer="2",
            reasoning="Test",
            source_model="model"
        )
        traj2._local_competition = 0.0

        pareto_front = [traj1, traj2]

        # Should fallback to uniform sampling
        selected = selector._sample_by_probability(pareto_front, n_samples=2)

        assert len(selected) == 2

    def test_sample_by_probability_none_scores(self, selector):
        """Test sampling when local competition scores are None."""
        traj1 = Trajectory(
            query="Test",
            answer="1",
            reasoning="Test",
            source_model="model"
        )
        traj1._local_competition = None

        pareto_front = [traj1]

        # Should handle None gracefully
        selected = selector._sample_by_probability(pareto_front, n_samples=1)

        assert len(selected) == 1
