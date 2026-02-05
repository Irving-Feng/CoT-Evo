"""
Population management for CoT-Evo framework.

This module defines the Population class, which manages a collection of
trajectories and provides methods for selection, replacement, and analysis.
"""

from typing import List, Dict, Callable, Optional
import numpy as np
from pathlib import Path

from .trajectory import Trajectory


class Population:
    """
    Manages a population of trajectories for evolutionary optimization.

    The Population class maintains a collection of Trajectory objects and
    provides methods for adding, removing, selecting, and analyzing them.

    Attributes:
        trajectories: List of trajectories in the population
        _embedding_cache: Cache for computed embeddings
    """

    def __init__(self, trajectories: Optional[List[Trajectory]] = None):
        """
        Initialize a Population.

        Args:
            trajectories: Initial list of trajectories (empty if None)
        """
        self.trajectories = trajectories if trajectories is not None else []
        self._embedding_cache: Dict[str, np.ndarray] = {}

    @property
    def size(self) -> int:
        """Get the current population size."""
        return len(self.trajectories)

    def is_empty(self) -> bool:
        """Check if the population is empty."""
        return len(self.trajectories) == 0

    def add(self, trajectory: Trajectory) -> None:
        """
        Add a single trajectory to the population.

        Args:
            trajectory: Trajectory to add
        """
        self.trajectories.append(trajectory)

    def extend(self, trajectories: List[Trajectory]) -> None:
        """
        Add multiple trajectories to the population.

        Args:
            trajectories: List of trajectories to add
        """
        self.trajectories.extend(trajectories)

    def remove(self, trajectory: Trajectory) -> bool:
        """
        Remove a specific trajectory from the population.

        Args:
            trajectory: Trajectory to remove

        Returns:
            True if trajectory was found and removed, False otherwise
        """
        try:
            self.trajectories.remove(trajectory)
            return True
        except ValueError:
            return False

    def remove_lowest(self, n: int, fitness_func: Optional[Callable] = None) -> None:
        """
        Remove the n lowest-fitness trajectories from the population.

        Args:
            n: Number of trajectories to remove
            fitness_func: Optional function to compute fitness (defaults to trajectory._fitness_score)
        """
        if n <= 0:
            return

        if n >= len(self.trajectories):
            self.trajectories = []
            return

        # Sort by fitness score (ascending)
        if fitness_func is None:
            # Use pre-computed fitness scores
            sorted_trajs = sorted(
                self.trajectories,
                key=lambda t: t._fitness_score if t._fitness_score is not None else float('-inf')
            )
        else:
            sorted_trajs = sorted(self.trajectories, key=fitness_func)

        # Remove the n lowest
        self.trajectories = sorted_trajs[n:]

    def remove_duplicates(self) -> None:
        """Remove duplicate trajectories based on their IDs."""
        seen_ids = set()
        unique_trajs = []

        for traj in self.trajectories:
            if traj.id not in seen_ids:
                seen_ids.add(traj.id)
                unique_trajs.append(traj)

        self.trajectories = unique_trajs

    def get_best(self, n: int = 1) -> List[Trajectory]:
        """
        Get the n best trajectories from the population.

        Args:
            n: Number of best trajectories to return

        Returns:
            List of the n best trajectories (sorted by fitness descending)
        """
        # Filter out trajectories without fitness scores
        evaluated = [t for t in self.trajectories if t._fitness_score is not None]

        if not evaluated:
            return []

        sorted_trajs = sorted(evaluated, key=lambda t: t._fitness_score, reverse=True)
        return sorted_trajs[:n]

    def get_trajectory(self, trajectory_id: str) -> Optional[Trajectory]:
        """
        Get a trajectory by its ID.

        Args:
            trajectory_id: ID of the trajectory to find

        Returns:
            Trajectory if found, None otherwise
        """
        for traj in self.trajectories:
            if traj.id == trajectory_id:
                return traj
        return None

    def filter(self, condition: Callable[[Trajectory], bool]) -> "Population":
        """
        Filter trajectories based on a condition.

        Args:
            condition: Function that takes a Trajectory and returns True to keep it

        Returns:
            New Population containing filtered trajectories
        """
        filtered = [t for t in self.trajectories if condition(t)]
        return Population(filtered)

    def get_evaluated(self) -> "Population":
        """Get a new population containing only evaluated trajectories."""
        return self.filter(lambda t: t.is_evaluated())

    def get_correct(self) -> "Population":
        """Get a new population containing only correct trajectories."""
        return self.filter(lambda t: t._exact_match is True)

    def get_incorrect(self) -> "Population":
        """Get a new population containing only incorrect trajectories."""
        return self.filter(lambda t: t._exact_match is False)

    def get_pareto_front(self) -> List[Trajectory]:
        """
        Get the Pareto front based on novelty and local competition scores.

        Implements formula 10 from the paper: Pareto front F_t

        Returns:
            List of non-dominated trajectories
        """
        # Filter trajectories with both scores computed
        valid_trajs = [
            t for t in self.trajectories
            if t._novelty_score is not None and t._local_competition is not None
        ]

        if not valid_trajs:
            return []

        pareto_front = []

        for traj in valid_trajs:
            is_dominated = False

            for other in valid_trajs:
                if traj.id == other.id:
                    continue

                # Check if 'other' dominates 'traj'
                # g(other) >= g(traj) with at least one strict inequality
                n_better = other._novelty_score >= traj._novelty_score
                l_better = other._local_competition >= traj._local_competition

                if (n_better and l_better) and (n_better or l_better):
                    is_dominated = True
                    break

            if not is_dominated:
                pareto_front.append(traj)

        return pareto_front

    def sample(self, n: int, probabilities: Optional[np.ndarray] = None) -> List[Trajectory]:
        """
        Sample n trajectories from the population.

        Args:
            n: Number of trajectories to sample
            probabilities: Optional probability weights for each trajectory

        Returns:
            List of sampled trajectories
        """
        if not self.trajectories:
            return []

        if n <= 0:
            return []

        if n >= len(self.trajectories):
            return self.trajectories.copy()

        if probabilities is None:
            # Uniform sampling
            indices = np.random.choice(len(self.trajectories), size=n, replace=False)
        else:
            # Weighted sampling
            indices = np.random.choice(len(self.trajectories), size=n, replace=False, p=probabilities)

        return [self.trajectories[i] for i in indices]

    def shuffle(self) -> None:
        """Randomly shuffle the trajectories in the population."""
        np.random.shuffle(self.trajectories)

    def statistics(self) -> Dict[str, any]:
        """
        Compute population statistics.

        Returns:
            Dictionary containing population statistics
        """
        if not self.trajectories:
            return {
                "size": 0,
                "evaluated": 0,
                "correct": 0,
                "avg_fitness": None,
                "best_fitness": None,
                "worst_fitness": None,
            }

        evaluated = [t for t in self.trajectories if t.is_evaluated()]
        correct = [t for t in self.trajectories if t.is_correct()]

        fitness_scores = [t._fitness_score for t in evaluated if t._fitness_score is not None]

        stats = {
            "size": len(self.trajectories),
            "evaluated": len(evaluated),
            "correct": len(correct),
            "accuracy": len(correct) / len(evaluated) if evaluated else 0.0,
            "avg_fitness": np.mean(fitness_scores) if fitness_scores else None,
            "best_fitness": np.max(fitness_scores) if fitness_scores else None,
            "worst_fitness": np.min(fitness_scores) if fitness_scores else None,
        }

        return stats

    def save(self, filepath: Path) -> None:
        """
        Save population to a file.

        Args:
            filepath: Path to save the population
        """
        import pickle

        data = {
            "trajectories": [t.to_dict() for t in self.trajectories],
            "size": len(self.trajectories),
        }

        with open(filepath, "wb") as f:
            pickle.dump(data, f)

    @classmethod
    def load(cls, filepath: Path) -> "Population":
        """
        Load population from a file.

        Args:
            filepath: Path to load the population from

        Returns:
            Loaded Population instance
        """
        import pickle

        with open(filepath, "rb") as f:
            data = pickle.load(f)

        trajectories = [Trajectory.from_dict(t) for t in data["trajectories"]]
        return cls(trajectories)

    def __repr__(self) -> str:
        """String representation of the population."""
        stats = self.statistics()
        return (
            f"Population(size={stats['size']}, "
            f"evaluated={stats['evaluated']}, "
            f"correct={stats['correct']}, "
            f"accuracy={stats['accuracy']:.2%})"
        )

    def __len__(self) -> int:
        """Get the population size."""
        return len(self.trajectories)

    def __iter__(self):
        """Iterate over trajectories."""
        return iter(self.trajectories)
