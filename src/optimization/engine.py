"""
Evolution engine for CoT-Evo framework.

This module implements Algorithm 1 from the paper, integrating all core algorithm modules
(initialization, selection, variation) into a cohesive evolutionary loop.
"""

import asyncio
import random
import logging
from typing import List, Optional, Dict
from pathlib import Path
import numpy as np

from ..models.registry import ModelRegistry
from ..initialization.generators import MultiThinkerGenerator
from ..selection.nslc import NSLCSelector
from ..variation.crossover import ReflectiveCrossover
from ..variation.mutation import ReflectiveMutation
from ..core.fitness import FitnessEvaluator
from ..core.population import Population
from ..core.trajectory import Trajectory
from .config import EvolutionEngineConfig, GenerationHistory, EvolutionCheckpoint

logger = logging.getLogger(__name__)


class EvolutionEngine:
    """
    Main evolution engine implementing Algorithm 1 from the CoT-Evo paper.

    This class orchestrates the entire evolutionary optimization process for a single query,
    including initialization, evaluation, selection, variation, and replacement.

    Example usage:
        ```python
        engine = EvolutionEngine(
            model_registry=registry,
            generator=generator,
            selector=selector,
            crossover=crossover,
            mutation=mutation,
            fitness_evaluator=fitness_evaluator,
            config=config
        )

        best_trajectory = await engine.evolve(
            query="What is the molecular weight of water?",
            ground_truth="18.015 g/mol"
        )
        ```
    """

    def __init__(
        self,
        model_registry: ModelRegistry,
        generator: MultiThinkerGenerator,
        selector: NSLCSelector,
        crossover: ReflectiveCrossover,
        mutation: ReflectiveMutation,
        fitness_evaluator: FitnessEvaluator,
        config: EvolutionEngineConfig,
        output_dir: Optional[Path] = None,
        sample_id: Optional[str] = None
    ):
        """
        Initialize the evolution engine.

        Args:
            model_registry: Model registry for accessing all LLM models
            generator: Multi-thinker generator for initial population (Formula 1-2)
            selector: NSLC selector for novelty-driven parent selection (Formula 7-11)
            crossover: Reflective crossover operator (Section 2.4)
            mutation: Reflective mutation operator (Formula 12-14)
            fitness_evaluator: Fitness evaluator (Formula 6)
            config: Engine configuration hyperparameters
            output_dir: Output directory for detailed logging (optional)
            sample_id: Sample ID for logging (optional)
        """
        self.model_registry = model_registry
        self.generator = generator
        self.selector = selector
        self.crossover = crossover
        self.mutation = mutation
        self.fitness_evaluator = fitness_evaluator
        self.config = config

        # State tracking
        self.generation: int = 0
        self.history: List[GenerationHistory] = []
        self.best_trajectory: Optional[Trajectory] = None
        self.query: Optional[str] = None
        self.ground_truth: Optional[str] = None
        self.reference_knowledge: Optional[str] = None

        # Evolution logger (optional, for detailed logging)
        self.evo_logger = None
        if output_dir and sample_id:
            from ..utils.evolution_logger import EvolutionLogger
            self.evo_logger = EvolutionLogger(output_dir, sample_id)

        # Configure logging
        logging.getLogger().setLevel(getattr(logging, config.log_level))

        logger.info(f"Initialized EvolutionEngine with {config.n_generations} generations, "
                   f"population size {config.population_size}")

    async def evolve(
        self,
        query: str,
        ground_truth: str,
        reference_knowledge: Optional[str] = None,
        checkpoint_path: Optional[Path] = None
    ) -> Trajectory:
        """
        Run the full evolution loop for a single query (Algorithm 1).

        Args:
            query: The input problem/question
            ground_truth: The correct answer for evaluation
            reference_knowledge: Optional reference knowledge for fitness evaluation
            checkpoint_path: Optional path to save/load checkpoints

        Returns:
            Best trajectory found: t* = argmax R(t)
        """
        logger.info(f"Starting evolution for query: {query[:100]}...")

        # Store for later use
        self.query = query
        self.ground_truth = ground_truth
        self.reference_knowledge = reference_knowledge

        # Try to resume from checkpoint if provided
        if checkpoint_path and checkpoint_path.exists():
            logger.info(f"Attempting to resume from checkpoint: {checkpoint_path}")
            population = self._resume_from_checkpoint(checkpoint_path)
        else:
            # Step 1: Initialize population (Formula 1-2)
            logger.info("Step 1: Initializing population...")
            population = await self._initialize_population(query, ground_truth)
            self.generation = 0

            # Log initial population
            if self.evo_logger:
                self.evo_logger.log_initial_population(population.trajectories, 0)

        # Step 2: Main evolution loop (Algorithm 1, lines 4-12)
        logger.info("Step 2: Starting evolution loop...")
        for generation in range(self.generation + 1, self.config.n_generations + 1):
            self.generation = generation
            logger.info(f"=== Generation {generation}/{self.config.n_generations} ===")

            # a. Evaluate all trajectories
            await self._evaluate_population(population, ground_truth, reference_knowledge)

            # b. Check convergence
            if self._check_convergence(population):
                logger.info(f"Converged at generation {generation}")
                break

            # c. Select parents using NSLC (Formula 7-11)
            n_parents = max(2, int(self.config.population_size * (1 - self.config.elitism_ratio)))
            parents = await self.selector.select_parents(population, n_parents=n_parents)
            logger.info(f"Selected {len(parents)} parents from Pareto front")

            # Log selection
            if self.evo_logger:
                self.evo_logger.log_selection(parents, generation)

            # d. Generate offspring via variation
            offspring = await self._generate_offspring(parents, population)
            logger.info(f"Generated {len(offspring)} offspring")

            # e. Merge and replacement
            population = self._replace_population(population, offspring)
            logger.info(f"New population size: {population.size}")

            # Log new generation
            if self.evo_logger:
                self.evo_logger.log_new_generation(population.trajectories, generation)

            # f. Log progress and save checkpoint
            self._log_generation_stats(generation, population)

            if self.config.save_checkpoints and generation % self.config.checkpoint_interval == 0:
                if checkpoint_path:
                    self._save_checkpoint(population, checkpoint_path)

        # Step 3: Return best trajectory (Algorithm 1, line 13)
        best = population.get_best(1)[0]
        logger.info(f"Evolution complete. Best fitness: {best.fitness_score:.4f}")

        # Log final best
        if self.evo_logger:
            self.evo_logger.log_final_best(best, self.generation)

        return best

    async def _initialize_population(
        self,
        query: str,
        ground_truth: str
    ) -> Population:
        """
        Initialize population using Formula 1-2.

        Formula 1: t_i = l_i(x) - vanilla generation
        Formula 2: t_j = l_j(x, K_x) - knowledge-augmented generation

        Args:
            query: The input problem
            ground_truth: The correct answer (for knowledge generation)

        Returns:
            Initial population P⁰
        """
        logger.debug(f"Generating {self.config.n_vanilla} vanilla + "
                    f"{self.config.n_knowledge_augmented} knowledge-augmented trajectories")

        population = await self.generator.generate_initial_pool(
            query=query,
            ground_truth=ground_truth,
            n_vanilla=self.config.n_vanilla,
            n_knowledge_augmented=self.config.n_knowledge_augmented
        )

        logger.debug(f"Generated initial population with {population.size} trajectories")
        return population

    async def _evaluate_population(
        self,
        population: Population,
        ground_truth: str,
        reference_knowledge: Optional[str] = None
    ) -> None:
        """
        Evaluate fitness for all unevaluated trajectories in the population.

        Implements Formula 6: R(t) = s_EM + λ₁s_LEN + λ₂s_KNOW

        Args:
            population: Population to evaluate
            ground_truth: Correct answer for exact match evaluation
            reference_knowledge: Reference knowledge for knowledge evaluation
        """
        # Find unevaluated trajectories (lazy evaluation)
        unevaluated = [t for t in population.trajectories if not t.is_evaluated()]

        if not unevaluated:
            logger.debug("All trajectories already evaluated")
            return

        logger.debug(f"Evaluating {len(unevaluated)} unevaluated trajectories...")

        # Evaluate each trajectory
        for trajectory in unevaluated:
            try:
                fitness = await self.fitness_evaluator.evaluate(
                    trajectory,
                    ground_truth,
                    reference_knowledge=reference_knowledge
                )
                logger.debug(f"Trajectory {trajectory.id[:8]} fitness: {fitness:.4f}")
            except Exception as e:
                logger.error(f"Failed to evaluate trajectory {trajectory.id[:8]}: {e}")
                # Set a low fitness score so it doesn't get selected
                trajectory._fitness_score = 0.0

    def _check_convergence(self, population: Population) -> bool:
        """
        Check if the population has converged to a satisfactory solution.

        Args:
            population: Current population

        Returns:
            True if any trajectory has fitness >= convergence_threshold
        """
        best = population.get_best(1)
        if not best:
            return False

        max_fitness = best[0].fitness_score
        converged = max_fitness >= self.config.convergence_threshold

        if converged:
            logger.info(f"Convergence check passed: {max_fitness:.4f} >= {self.config.convergence_threshold}")

        return converged

    async def _generate_offspring(
        self,
        parents: List[Trajectory],
        population: Population
    ) -> List[Trajectory]:
        """
        Generate offspring population through crossover and mutation.

        Strategy:
        1. Determine number of offspring needed
        2. For each offspring:
           - Sample a parent pair from parents
           - Apply crossover with probability (only if target is incorrect)
           - Apply mutation otherwise
           - Retry on failure

        Args:
            parents: Selected parent trajectories from NSLC
            population: Current population (for sampling providers)

        Returns:
            List of offspring trajectories
        """
        n_offspring = self.config.population_size - len(parents)
        offspring = []

        # Track operations for logging
        crossover_ops = []
        mutation_ops = []

        logger.debug(f"Need to generate {n_offspring} offspring")

        for i in range(n_offspring):
            # Sample parent pair
            target, provider = self._sample_parent_pair(parents, population)

            # Decide operation type
            # Crossover only if target answer is incorrect (s_EM = 0, from Section 2.4)
            use_crossover = (
                random.random() < self.config.crossover_prob and
                target.exact_match is False
            )

            try:
                if use_crossover:
                    # Crossover (only for incorrect answers)
                    logger.debug(f"Offspring {i+1}: crossover between {target.id[:8]} and {provider.id[:8]}")
                    child = await self.crossover.crossover(target, provider)

                    # Track crossover operation
                    crossover_ops.append({
                        "target": target.id[:8],
                        "provider": provider.id[:8],
                        "offspring": child.id[:8] if child else None,
                        "success": child is not None
                    })
                else:
                    # Mutation
                    mode = self._sample_mutation_mode()
                    logger.debug(f"Offspring {i+1}: {mode}-mutation on {target.id[:8]}")
                    child = await self.mutation.mutate(
                        target,
                        mode=mode,
                        query=self.query,
                        ground_truth=self.ground_truth
                    )

                    # Track mutation operation
                    mutation_ops.append({
                        "target": target.id[:8],
                        "mode": mode,
                        "offspring": child.id[:8] if child else None,
                        "success": child is not None
                    })

                if child is not None:
                    offspring.append(child)
                else:
                    # Fallback: use parent as is
                    logger.warning(f"Offspring {i+1} generation failed, using parent as fallback")
                    offspring.append(target)

            except Exception as e:
                logger.error(f"Offspring {i+1} generation failed: {e}")
                # Retry with different parent
                if i < n_offspring - 1:
                    continue
                # Last fallback: use parent
                offspring.append(target)

        # Log operations if logger is available
        if self.evo_logger:
            if crossover_ops:
                self.evo_logger.log_crossover(crossover_ops, self.generation)
            if mutation_ops:
                self.evo_logger.log_mutation(mutation_ops, self.generation)

        return offspring

    def _sample_parent_pair(
        self,
        parents: List[Trajectory],
        population: Population
    ) -> tuple[Trajectory, Trajectory]:
        """
        Sample a target and provider parent pair.

        Args:
            parents: Selected parents from NSLC
            population: Full population (for sampling providers)

        Returns:
            Tuple of (target, provider) trajectories
        """
        # Sample target from parents (prefer high local competition)
        if parents[0].local_competition is not None:
            # Weight by local competition score
            weights = [p.local_competition + self.config.epsilon for p in parents]
            weights = np.array(weights) / np.sum(weights)
            target_idx = np.random.choice(len(parents), p=weights)
            target = parents[target_idx]
        else:
            target = random.choice(parents)

        # Sample provider from full population (can be any trajectory)
        provider = random.choice(population.trajectories)

        return target, provider

    def _sample_mutation_mode(self) -> str:
        """
        Sample a mutation mode based on configured distribution.

        Returns:
            One of "add", "delete", or "innovate"
        """
        modes = list(self.config.mutation_mode_distribution.keys())
        probs = list(self.config.mutation_mode_distribution.values())

        return np.random.choice(modes, p=probs)

    def _replace_population(
        self,
        current: Population,
        offspring: List[Trajectory]
    ) -> Population:
        """
        Replace population using elitism strategy.

        Steps:
        1. Keep top N% as elites
        2. Add all offspring
        3. Remove duplicates based on trajectory IDs
        4. Remove lowest-fitness to maintain population size

        Args:
            current: Current population
            offspring: Newly generated offspring

        Returns:
            New population after replacement
        """
        # Elitism: keep best performers
        n_elites = int(self.config.population_size * self.config.elitism_ratio)
        elites = current.get_best(n_elites)

        # Merge elites and offspring
        new_population = Population(elites + offspring)

        # Remove duplicates (based on trajectory IDs)
        new_population.remove_duplicates()

        # Trim to target size
        if new_population.size > self.config.population_size:
            n_remove = new_population.size - self.config.population_size
            new_population.remove_lowest(n_remove)

        logger.debug(f"Population replacement: {len(elites)} elites + {len(offspring)} offspring "
                    f"→ {new_population.size} (after dedup and trimming)")

        return new_population

    def _log_generation_stats(self, generation: int, population: Population) -> None:
        """
        Compute and log statistics for the current generation.

        Args:
            generation: Current generation number
            population: Current population
        """
        if not self.config.log_generation_stats:
            return

        stats = population.statistics()

        # Compute NSLC-specific stats
        pareto_front = population.get_pareto_front()

        avg_novelty = None
        avg_local_competition = None

        novelty_scores = [t.novelty_score for t in population.trajectories
                         if t.novelty_score is not None]
        if novelty_scores:
            avg_novelty = np.mean(novelty_scores)

        local_comp_scores = [t.local_competition for t in population.trajectories
                            if t.local_competition is not None]
        if local_comp_scores:
            avg_local_competition = np.mean(local_comp_scores)

        history = GenerationHistory(
            generation=generation,
            population_size=stats['size'],
            avg_fitness=stats['avg_fitness'],
            best_fitness=stats['best_fitness'],
            worst_fitness=stats['worst_fitness'],
            n_correct=stats['correct'],
            accuracy=stats['accuracy'],
            avg_novelty=avg_novelty,
            avg_local_competition=avg_local_competition,
            pareto_front_size=len(pareto_front)
        )

        self.history.append(history)

        logger.info(
            f"Gen {generation}: "
            f"fitness={stats['avg_fitness']:.3f}/{stats['best_fitness']:.3f}, "
            f"accuracy={stats['accuracy']:.2%}, "
            f"pareto_size={len(pareto_front)}, "
            f"pop_size={population.size}"
        )

    def _save_checkpoint(
        self,
        population: Population,
        checkpoint_path: Path
    ) -> None:
        """
        Save current evolution state to disk.

        Args:
            population: Current population
            checkpoint_path: Path where to save checkpoint
        """
        # Serialize trajectories
        serialized_trajectories = [t.to_dict() for t in population.trajectories]
        serialized_history = [h.to_dict() for h in self.history]
        serialized_best = self.best_trajectory.to_dict() if self.best_trajectory else None

        checkpoint = EvolutionCheckpoint(
            query=population.trajectories[0].query if population.trajectories else "",
            ground_truth=self.ground_truth,
            reference_knowledge=self.reference_knowledge,
            current_population=serialized_trajectories,
            generation=self.generation,
            history=serialized_history,
            best_trajectory=serialized_best,
            config={
                "n_generations": self.config.n_generations,
                "population_size": self.config.population_size,
                "convergence_threshold": self.config.convergence_threshold,
                "k_neighbors": self.config.k_neighbors,
                "n_vanilla": self.config.n_vanilla,
                "n_knowledge_augmented": self.config.n_knowledge_augmented,
                "crossover_prob": self.config.crossover_prob,
                "mutation_prob": self.config.mutation_prob,
                "mutation_mode_distribution": self.config.mutation_mode_distribution,
                "elitism_ratio": self.config.elitism_ratio,
                "epsilon": self.config.epsilon,
            }
        )

        checkpoint.save(checkpoint_path)

    def _resume_from_checkpoint(self, checkpoint_path: Path) -> Population:
        """
        Resume evolution from a checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file

        Returns:
            Restored population
        """
        checkpoint = EvolutionCheckpoint.load(checkpoint_path)

        # Restore state
        self.generation = checkpoint.generation
        self.ground_truth = checkpoint.ground_truth
        self.reference_knowledge = checkpoint.reference_knowledge
        self.history = [GenerationHistory.from_dict(h) for h in checkpoint.history]

        if checkpoint.best_trajectory:
            self.best_trajectory = Trajectory.from_dict(checkpoint.best_trajectory)

        # Restore population
        trajectories = [Trajectory.from_dict(t) for t in checkpoint.current_population]
        population = Population(trajectories)

        logger.info(f"Resumed from generation {self.generation}, population size {population.size}")

        return population

    async def evolve_batch(
        self,
        samples: List[Dict[str, str]],
        max_concurrent: int = 4,
        checkpoint_dir: Optional[Path] = None
    ) -> List[Trajectory]:
        """
        Evolve CoTs for multiple samples in parallel.

        Args:
            samples: List of {"query": str, "ground_truth": str} dicts
            max_concurrent: Maximum number of concurrent evolution tasks
            checkpoint_dir: Directory to save per-sample checkpoints

        Returns:
            List of best trajectories, one per sample
        """
        if checkpoint_dir:
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

        semaphore = asyncio.Semaphore(max_concurrent)

        async def evolve_with_limit(sample, idx):
            async with semaphore:
                sample_checkpoint_path = checkpoint_dir / f"sample_{idx}.pkl" if checkpoint_dir else None

                # Create a new engine instance for this sample
                # (Note: In production, you might want to reuse the engine configuration)
                result = await self.evolve(
                    query=sample["query"],
                    ground_truth=sample["ground_truth"],
                    checkpoint_path=sample_checkpoint_path
                )
                return result

        logger.info(f"Processing {len(samples)} samples with max {max_concurrent} concurrent")

        tasks = [evolve_with_limit(sample, idx) for idx, sample in enumerate(samples)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out exceptions
        valid_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Sample {i} failed: {result}")
            else:
                valid_results.append(result)

        logger.info(f"Completed {len(valid_results)}/{len(samples)} samples successfully")

        return valid_results
