"""
Fixtures for unit tests.
"""

import pytest
import numpy as np
from src.core.trajectory import Trajectory
from src.core.population import Population


@pytest.fixture
def sample_trajectory():
    """Create a sample trajectory for testing."""
    traj = Trajectory(
        query="What is the molecular weight of water (H2O)?",
        answer="The molecular weight of water is 18.015 g/mol.",
        reasoning="To calculate the molecular weight of H2O: "
                   "Hydrogen (H) has atomic weight ~1.008, Oxygen (O) has atomic weight ~15.999. "
                   "H2O has 2 hydrogen atoms and 1 oxygen atom: "
                   "2 * 1.008 + 15.999 = 2.016 + 15.999 = 18.015 g/mol.",
        source_model="test-model",
        generation_method="vanilla"
    )
    traj.set_fitness_score(1.0)
    traj.set_exact_match(True)
    traj.set_length_score(1.0)
    traj.set_knowledge_score(5)
    traj.set_novelty_score(0.75)
    traj.set_local_competition(0.40)
    return traj


@pytest.fixture
def sample_population():
    """Create a sample population for testing."""
    trajectories = []
    for i in range(10):
        traj = Trajectory(
            query=f"Question {i}",
            answer=f"Answer {i}",
            reasoning=f"Reasoning {i} " + "word " * 50,
            source_model=f"model-{i % 3}"
        )
        traj.set_fitness_score(0.5 + i * 0.05)
        traj.set_exact_match(i < 7)
        traj.set_length_score(1.0)
        traj.set_novelty_score(0.5 + i * 0.05)
        traj.set_local_competition(0.8 - i * 0.08)
        trajectories.append(traj)

    return Population(trajectories)


@pytest.fixture
def sample_embeddings():
    """Create sample embedding vectors."""
    np.random.seed(42)
    return np.random.rand(10, 1024)  # 10 trajectories, 1024-dim embeddings
