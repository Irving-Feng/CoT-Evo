"""
NSLC (Novelty Search with Local Competition) selection algorithm for CoT-Evo.

This module implements Formula 7-11 from the paper:
- Formula 7: Behavioral embedding z_t = b(t)
- Formula 8: Novelty score N(t)
- Formula 9: Local competition score L(t)
- Formula 10: Pareto front F_t
- Formula 11: Probability-based sampling
"""

import asyncio
import logging
from typing import List, Optional, Dict
import numpy as np

from ..core.population import Population
from ..core.trajectory import Trajectory
from ..models.registry import ModelRegistry
from ..models.base import EmbeddingProvider


logger = logging.getLogger(__name__)


class NSLCSelector:
    """
    Novelty Search with Local Competition (NSLC) selector.

    Implements the bi-objective selection strategy that balances:
    1. Novelty: Encouraging diverse reasoning patterns
    2. Local competition: Rewarding local quality improvements

    This is the core innovation of the CoT-Evo paper.
    """

    def __init__(
        self,
        model_registry: ModelRegistry,
        n_neighbors: int = 5,
        epsilon: float = 0.1,
        embedding_cache=None
    ):
        """
        Initialize the NSLC selector.

        Args:
            model_registry: Model registry (for accessing embedding model)
            n_neighbors: Number of nearest neighbors for k-NN (k in formulas)
            epsilon: Small constant for numerical stability (ε in formula 11)
            embedding_cache: Optional embedding cache for performance
        """
        self.registry = model_registry
        self.n_neighbors = n_neighbors
        self.epsilon = epsilon

        # Get embedding model from registry
        self.embedding_model: Optional[EmbeddingProvider] = model_registry.get_embedding_model()

        # Optional embedding cache
        self.embedding_cache = embedding_cache
        if embedding_cache:
            logger.info("NSLC selector initialized with embedding cache")

    async def select_parents(
        self,
        population: Population,
        n_parents: int
    ) -> List[Trajectory]:
        """
        Select parent trajectories using NSLC algorithm (Formula 7-11).

        Args:
            population: Current population of trajectories
            n_parents: Number of parents to select

        Returns:
            List of selected parent trajectories
        """
        if population.size == 0:
            logger.warning("Cannot select from empty population")
            return []

        if population.size <= n_parents:
            logger.debug(f"Population size ({population.size}) <= n_parents ({n_parents}), returning all")
            return population.trajectories.copy()

        # Step 1: Compute behavioral embeddings (Formula 7)
        await self._compute_embeddings(population)

        # Step 2 & 3: Compute novelty and local competition scores (Formula 8-9)
        self._compute_novelty_and_local_competition(population)

        # Step 4: Get Pareto front (Formula 10)
        pareto_front = population.get_pareto_front()

        if not pareto_front:
            logger.warning("Empty Pareto front, falling back to random sampling")
            return population.sample(n_parents)

        logger.debug(f"Pareto front size: {len(pareto_front)}")

        # Step 5: Sample by probability (Formula 11)
        selected = self._sample_by_probability(pareto_front, n_parents)

        logger.info(
            f"Selected {len(selected)} parents from Pareto front of {len(pareto_front)} "
            f"(novelty + local competition)"
        )

        return selected

    async def _compute_embeddings(self, population: Population) -> None:
        """
        Compute behavioral embeddings for all trajectories (Formula 7).

        z_t = b(t)

        Uses embedding cache if available to avoid redundant computation.

        Args:
            population: Population to compute embeddings for
        """
        if self.embedding_model is None:
            logger.warning("No embedding model available, skipping embedding computation")
            return

        # Collect trajectories that need embeddings
        trajectories_to_embed = [
            t for t in population.trajectories
            if not t.has_embedding()
        ]

        if not trajectories_to_embed:
            logger.debug("All trajectories already have embeddings")
            return

        logger.debug(f"Computing embeddings for {len(trajectories_to_embed)} trajectories")

        # Check cache for trajectories that already have cached embeddings
        cache_hits = []
        cache_misses = []

        if self.embedding_cache:
            model_name = self.embedding_model.model_name

            for traj in trajectories_to_embed:
                cached = self.embedding_cache.get(traj.reasoning, model_name)
                if cached is not None:
                    traj.set_embedding(cached)
                    cache_hits.append(traj)
                else:
                    cache_misses.append(traj)

            logger.info(f"Embedding cache: {len(cache_hits)} hits, {len(cache_misses)} misses")
        else:
            cache_misses = trajectories_to_embed

        # Compute embeddings for cache misses
        if cache_misses:
            try:
                embeddings = await self.embedding_model.embed_batch_async([
                    t.reasoning for t in cache_misses
                ])

                # Store embeddings and cache them
                model_name = self.embedding_model.model_name
                for traj, embedding in zip(cache_misses, embeddings):
                    traj.set_embedding(np.array(embedding))

                    # Store in cache if available
                    if self.embedding_cache:
                        self.embedding_cache.put(traj.reasoning, model_name, np.array(embedding))

            except Exception as e:
                logger.error(f"Failed to compute embeddings: {e}")

    def _compute_novelty_and_local_competition(self, population: Population) -> None:
        """
        Compute novelty and local competition scores (Formula 8-9).

        Formula 8: N(t) = (1/k) * Σ_{t' in N_k(t)} ||z_t - z_t'||_2
        Formula 9: L(t) = (1/k) * Σ_{t' in N_k(t)} (R(t) - R(t'))_+

        Args:
            population: Population to compute scores for
        """
        for traj in population.trajectories:
            if not traj.has_embedding():
                logger.warning(f"Trajectory {traj.id} has no embedding, skipping NSLC scores")
                continue

            # Find k nearest neighbors
            neighbors = self._get_k_neighbors(population, traj)

            if not neighbors:
                logger.warning(f"No neighbors found for trajectory {traj.id}")
                continue

            # Formula 8: Novelty score (average distance to k-NN)
            novelty_score = self._compute_novelty_score(traj, neighbors)
            traj.set_novelty_score(novelty_score)

            # Formula 9: Local competition score (average local advantage)
            local_competition = self._compute_local_competition_score(traj, neighbors)
            traj.set_local_competition(local_competition)

    def _get_k_neighbors(
        self,
        population: Population,
        target: Trajectory
    ) -> List[Trajectory]:
        """
        Find k nearest neighbors in behavioral space.

        Args:
            population: Population to search in
            target: Target trajectory

        Returns:
            List of k nearest neighbors (excluding target itself)
        """
        if not target.has_embedding():
            return []

        # Compute distances to all other trajectories
        distances = []
        for traj in population.trajectories:
            if traj.id == target.id or not traj.has_embedding():
                continue

            dist = np.linalg.norm(target.embedding - traj.embedding)
            distances.append((dist, traj))

        # Sort by distance and take top k
        distances.sort(key=lambda x: x[0])
        k_neighbors = [traj for _, traj in distances[:self.n_neighbors]]

        return k_neighbors

    def _compute_novelty_score(
        self,
        target: Trajectory,
        neighbors: List[Trajectory]
    ) -> float:
        """
        Compute novelty score (Formula 8).

        N(t) = (1/k) * Σ_{t' in N_k(t)} ||z_t - z_t'||_2

        Args:
            target: Target trajectory
            neighbors: k nearest neighbors

        Returns:
            Novelty score (higher = more novel)
        """
        if not neighbors:
            return 0.0

        total_distance = 0.0
        for neighbor in neighbors:
            dist = np.linalg.norm(target.embedding - neighbor.embedding)
            total_distance += dist

        return total_distance / len(neighbors)

    def _compute_local_competition_score(
        self,
        target: Trajectory,
        neighbors: List[Trajectory]
    ) -> float:
        """
        Compute local competition score (Formula 9).

        L(t) = (1/k) * Σ_{t' in N_k(t)} (R(t) - R(t'))_+

        Args:
            target: Target trajectory
            neighbors: k nearest neighbors

        Returns:
            Local competition score (higher = locally better)
        """
        if not neighbors:
            return 0.0

        if target.fitness_score is None:
            logger.warning(f"Trajectory {target.id} has no fitness score")
            return 0.0

        advantages = []
        for neighbor in neighbors:
            if neighbor.fitness_score is None:
                continue

            advantage = target.fitness_score - neighbor.fitness_score
            advantages.append(max(0, advantage))  # (a)_+ = max(a, 0)

        if not advantages:
            return 0.0

        return sum(advantages) / len(advantages)

    def _sample_by_probability(
        self,
        pareto_front: List[Trajectory],
        n_samples: int
    ) -> List[Trajectory]:
        """
        Sample from Pareto front by probability (Formula 11).

        p(t) = (L(t) + ε) / Σ(L(t') + ε)

        Args:
            pareto_front: Pareto-optimal trajectories
            n_samples: Number of samples to draw

        Returns:
            Sampled trajectories
        """
        # Compute sampling probabilities
        probs = []
        for traj in pareto_front:
            local_comp = traj.local_competition if traj.local_competition is not None else 0.0
            prob = local_comp + self.epsilon
            probs.append(prob)

        # Normalize
        total = sum(probs)
        if total == 0:
            # Uniform sampling as fallback
            probs = [1.0 / len(pareto_front)] * len(pareto_front)
        else:
            probs = [p / total for p in probs]

        # Sample with replacement
        indices = np.random.choice(len(pareto_front), size=n_samples, p=probs, replace=True)
        selected = [pareto_front[i] for i in indices]

        return selected
