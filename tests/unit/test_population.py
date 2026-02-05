"""
Unit tests for Population class.
"""

import pytest
import numpy as np
from src.core.population import Population
from src.core.trajectory import Trajectory


class TestPopulation:
    """Test cases for Population class."""

    @pytest.fixture
    def sample_trajectories(self):
        """Create sample trajectories for testing."""
        trajs = []
        for i in range(10):
            traj = Trajectory(
                query=f"Question {i}",
                answer=f"Answer {i}",
                reasoning=f"Reasoning {i}",
                source_model=f"model-{i % 3}"
            )
            # Set fitness scores
            traj.set_fitness_score(0.5 + i * 0.05)
            if i < 7:
                traj.set_exact_match(True)
            else:
                traj.set_exact_match(False)
            trajs.append(traj)
        return trajs

    def test_population_creation(self):
        """Test creating an empty population."""
        pop = Population()
        assert pop.size == 0
        assert pop.is_empty()

    def test_population_with_trajectories(self, sample_trajectories):
        """Test creating population with initial trajectories."""
        pop = Population(sample_trajectories)
        assert pop.size == 10
        assert not pop.is_empty()

    def test_add_trajectory(self):
        """Test adding a single trajectory."""
        pop = Population()
        traj = Trajectory(
            query="Test",
            answer="Answer",
            reasoning="Reasoning",
            source_model="test"
        )

        pop.add(traj)

        assert pop.size == 1

    def test_extend_trajectories(self):
        """Test adding multiple trajectories."""
        pop = Population()
        trajs = [
            Trajectory(
                query=f"Q{i}",
                answer=f"A{i}",
                reasoning=f"R{i}",
                source_model="test"
            )
            for i in range(5)
        ]

        pop.extend(trajs)

        assert pop.size == 5

    def test_remove_trajectory(self, sample_trajectories):
        """Test removing a specific trajectory."""
        pop = Population(sample_trajectories)
        target = sample_trajectories[3]

        result = pop.remove(target)

        assert result is True
        assert pop.size == 9

    def test_remove_nonexistent_trajectory(self, sample_trajectories):
        """Test removing a trajectory that doesn't exist."""
        pop = Population(sample_trajectories)
        new_traj = Trajectory(
            query="New",
            answer="New",
            reasoning="New",
            source_model="test"
        )

        result = pop.remove(new_traj)

        assert result is False
        assert pop.size == 10

    def test_remove_lowest_fitness(self, sample_trajectories):
        """Test removing lowest fitness trajectories."""
        pop = Population(sample_trajectories)
        original_size = pop.size

        pop.remove_lowest(3)

        assert pop.size == original_size - 3

        # Remaining trajectories should have higher fitness
        for traj in pop.trajectories:
            assert traj.fitness_score >= 0.65  # First 3 have scores 0.5, 0.55, 0.6

    def test_remove_lowest_more_than_size(self, sample_trajectories):
        """Test removing more trajectories than exist."""
        pop = Population(sample_trajectories)

        pop.remove_lowest(20)

        assert pop.size == 0

    def test_remove_duplicates(self):
        """Test removing duplicate trajectories."""
        traj = Trajectory(
            query="Same",
            answer="Same",
            reasoning="Same",
            source_model="test"
        )

        pop = Population([traj, traj, traj])

        assert pop.size == 3

        pop.remove_duplicates()

        assert pop.size == 1

    def test_get_best(self, sample_trajectories):
        """Test getting best trajectories."""
        pop = Population(sample_trajectories)

        best = pop.get_best(3)

        assert len(best) == 3
        # Should be the 3 with highest fitness
        assert best[0].fitness_score >= best[1].fitness_score
        assert best[1].fitness_score >= best[2].fitness_score

    def test_get_best_from_empty(self):
        """Test getting best from empty population."""
        pop = Population()

        best = pop.get_best(5)

        assert len(best) == 0

    def test_get_trajectory_by_id(self, sample_trajectories):
        """Test getting trajectory by ID."""
        pop = Population(sample_trajectories)
        target_id = sample_trajectories[2].id

        found = pop.get_trajectory(target_id)

        assert found is not None
        assert found.id == target_id

    def test_get_trajectory_nonexistent_id(self, sample_trajectories):
        """Test getting trajectory with nonexistent ID."""
        pop = Population(sample_trajectories)

        found = pop.get_trajectory("nonexistent")

        assert found is None

    def test_filter_evaluated(self, sample_trajectories):
        """Test filtering evaluated trajectories."""
        pop = Population(sample_trajectories)

        # All should be evaluated (have fitness scores)
        evaluated_pop = pop.get_evaluated()

        assert evaluated_pop.size == 10

    def test_filter_correct(self, sample_trajectories):
        """Test filtering correct trajectories."""
        pop = Population(sample_trajectories)

        correct_pop = pop.get_correct()

        assert correct_pop.size == 7  # First 7 are correct

    def test_filter_incorrect(self, sample_trajectories):
        """Test filtering incorrect trajectories."""
        pop = Population(sample_trajectories)

        incorrect_pop = pop.get_incorrect()

        assert incorrect_pop.size == 3  # Last 3 are incorrect

    def test_get_pareto_front(self):
        """Test getting Pareto front."""
        pop = Population()

        # Create trajectories with different novelty/local competition scores
        for i in range(5):
            traj = Trajectory(
                query=f"Q{i}",
                answer=f"A{i}",
                reasoning=f"R{i}",
                source_model="test"
            )
            traj.set_novelty_score(0.5 + i * 0.1)
            traj.set_local_competition(0.8 - i * 0.15)
            pop.add(traj)

        pareto_front = pop.get_pareto_front()

        assert len(pareto_front) > 0
        assert len(pareto_front) <= 5

    def test_get_pareto_front_empty(self):
        """Test getting Pareto front from population without scores."""
        pop = Population([Trajectory("Q", "A", "R", "test") for _ in range(3)])

        pareto_front = pop.get_pareto_front()

        assert len(pareto_front) == 0

    def test_sample_uniform(self, sample_trajectories):
        """Test uniform sampling."""
        pop = Population(sample_trajectories)

        sampled = pop.sample(3)

        assert len(sampled) == 3
        assert all(isinstance(t, Trajectory) for t in sampled)

    def test_sample_with_probabilities(self, sample_trajectories):
        """Test weighted sampling."""
        pop = Population(sample_trajectories)

        # Higher probability for first few
        probs = np.array([0.5] * 5 + [0.1] * 5)
        probs = probs / probs.sum()

        sampled = pop.sample(3, probabilities=probs)

        assert len(sampled) == 3

    def test_shuffle(self, sample_trajectories):
        """Test shuffling population."""
        pop = Population(sample_trajectories)
        original_order = [t.id for t in pop.trajectories]

        pop.shuffle()

        shuffled_order = [t.id for t in pop.trajectories]

        # Same elements, different order (likely)
        assert set(original_order) == set(shuffled_order)

    def test_statistics(self, sample_trajectories):
        """Test population statistics."""
        pop = Population(sample_trajectories)

        stats = pop.statistics()

        assert stats["size"] == 10
        assert stats["evaluated"] == 10
        assert stats["correct"] == 7
        assert stats["accuracy"] == 0.7
        assert stats["avg_fitness"] is not None
        assert stats["best_fitness"] is not None
        assert stats["worst_fitness"] is not None

    def test_statistics_empty(self):
        """Test statistics for empty population."""
        pop = Population()

        stats = pop.statistics()

        assert stats["size"] == 0
        assert stats["evaluated"] == 0
        assert stats["correct"] == 0
        assert stats["avg_fitness"] is None

    def test_len_and_iteration(self, sample_trajectories):
        """Test __len__ and __iter__."""
        pop = Population(sample_trajectories)

        assert len(pop) == 10

        count = 0
        for traj in pop:
            assert isinstance(traj, Trajectory)
            count += 1

        assert count == 10

    def test_repr(self, sample_trajectories):
        """Test string representation."""
        pop = Population(sample_trajectories)

        repr_str = repr(pop)

        assert "Population" in repr_str
        assert "10" in repr_str
        assert "70.00%" in repr_str  # Accuracy with 2 decimal places
