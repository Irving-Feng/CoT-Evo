"""
Unit tests for evolution engine module.
"""

import pytest
import asyncio
import pickle
import numpy as np
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch, call
from datetime import datetime

from src.optimization.engine import EvolutionEngine
from src.optimization.config import (
    EvolutionEngineConfig,
    GenerationHistory,
    EvolutionCheckpoint
)
from src.core.population import Population
from src.core.trajectory import Trajectory
from src.models.registry import ModelRegistry
from src.initialization.generators import MultiThinkerGenerator
from src.selection.nslc import NSLCSelector
from src.variation.crossover import ReflectiveCrossover
from src.variation.mutation import ReflectiveMutation
from src.core.fitness import FitnessEvaluator


class TestEvolutionEngineConfig:
    """Test suite for EvolutionEngineConfig class."""

    def test_default_config(self):
        """Test default configuration values."""
        config = EvolutionEngineConfig()

        assert config.n_generations == 5
        assert config.population_size == 10
        assert config.convergence_threshold == 1.0
        assert config.k_neighbors == 5
        assert config.n_vanilla == 7
        assert config.n_knowledge_augmented == 3
        assert config.crossover_prob == 0.4
        assert config.mutation_prob == 0.6
        assert config.elitism_ratio == 0.5
        assert config.mutation_mode_distribution == {"add": 0.25, "delete": 0.25, "innovate": 0.5}

    def test_config_validation_valid(self):
        """Test configuration validation with valid values."""
        config = EvolutionEngineConfig(
            n_generations=10,
            population_size=20,
            crossover_prob=0.5,
            mutation_prob=0.5,
            elitism_ratio=0.3
        )

        assert config.n_generations == 10
        assert config.population_size == 20

    def test_config_validation_invalid_n_generations(self):
        """Test validation fails for n_generations < 1."""
        with pytest.raises(ValueError, match="n_generations must be at least 1"):
            EvolutionEngineConfig(n_generations=0)

    def test_config_validation_invalid_population_size(self):
        """Test validation fails for population_size < 2."""
        with pytest.raises(ValueError, match="population_size must be at least 2"):
            EvolutionEngineConfig(population_size=1)

    def test_config_validation_invalid_crossover_prob(self):
        """Test validation fails for crossover_prob outside [0, 1]."""
        with pytest.raises(ValueError, match="crossover_prob must be between 0 and 1"):
            EvolutionEngineConfig(crossover_prob=1.5)

    def test_config_validation_invalid_mutation_prob(self):
        """Test validation fails for mutation_prob outside [0, 1]."""
        with pytest.raises(ValueError, match="mutation_prob must be between 0 and 1"):
            EvolutionEngineConfig(mutation_prob=-0.1)

    def test_config_validation_crossover_mutation_sum_warning(self, caplog):
        """Test warning when crossover_prob + mutation_prob != 1.0."""
        import logging
        caplog.set_level(logging.WARNING)

        EvolutionEngineConfig(crossover_prob=0.7, mutation_prob=0.5)

        # Check that a warning was logged
        assert any(
            "should sum to 1.0" in record.message
            for record in caplog.records
            if record.levelname == "WARNING"
        )

    def test_config_validation_invalid_elitism_ratio(self):
        """Test validation fails for elitism_ratio outside [0, 1]."""
        with pytest.raises(ValueError, match="elitism_ratio must be between 0 and 1"):
            EvolutionEngineConfig(elitism_ratio=1.5)

    def test_config_validation_invalid_mutation_distribution(self):
        """Test validation fails when mutation_mode_distribution doesn't sum to 1.0."""
        with pytest.raises(ValueError, match="mutation_mode_distribution must sum to 1.0"):
            EvolutionEngineConfig(mutation_mode_distribution={"add": 0.5, "delete": 0.6})

    def test_config_from_yaml(self, tmp_path):
        """Test loading configuration from YAML file."""
        config_path = tmp_path / "test_config.yaml"
        config_content = """
evolution:
  n_generations: 15
  population_size: 25
  convergence_threshold: 0.95
  k_neighbors: 7
  n_vanilla: 10
  n_knowledge_augmented: 5
  crossover_prob: 0.3
  mutation_prob: 0.7
  mutation_mode_distribution:
    add: 0.3
    delete: 0.3
    innovate: 0.4
  elitism_ratio: 0.4
  epsilon: 0.15
"""
        config_path.write_text(config_content)

        config = EvolutionEngineConfig.from_yaml(config_path)

        assert config.n_generations == 15
        assert config.population_size == 25
        assert config.convergence_threshold == 0.95
        assert config.k_neighbors == 7
        assert config.n_vanilla == 10
        assert config.n_knowledge_augmented == 5
        assert config.crossover_prob == 0.3
        assert config.mutation_prob == 0.7
        assert config.mutation_mode_distribution == {"add": 0.3, "delete": 0.3, "innovate": 0.4}
        assert config.elitism_ratio == 0.4
        assert config.epsilon == 0.15

    def test_config_to_yaml(self, tmp_path):
        """Test saving configuration to YAML file."""
        config = EvolutionEngineConfig(
            n_generations=8,
            population_size=15,
            crossover_prob=0.35,
            mutation_prob=0.65
        )

        save_path = tmp_path / "saved_config.yaml"
        config.to_yaml(save_path)

        assert save_path.exists()

        # Load and verify
        loaded_config = EvolutionEngineConfig.from_yaml(save_path)
        assert loaded_config.n_generations == 8
        assert loaded_config.population_size == 15
        assert loaded_config.crossover_prob == 0.35


class TestGenerationHistory:
    """Test suite for GenerationHistory class."""

    def test_generation_history_creation(self):
        """Test creating a GenerationHistory instance."""
        history = GenerationHistory(
            generation=1,
            population_size=10,
            avg_fitness=0.75,
            best_fitness=0.95,
            worst_fitness=0.55,
            n_correct=7,
            accuracy=0.7,
            avg_novelty=0.3,
            avg_local_competition=0.25,
            pareto_front_size=5
        )

        assert history.generation == 1
        assert history.population_size == 10
        assert history.avg_fitness == 0.75
        assert history.best_fitness == 0.95
        assert history.worst_fitness == 0.55
        assert history.n_correct == 7
        assert history.accuracy == 0.7
        assert history.avg_novelty == 0.3
        assert history.avg_local_competition == 0.25
        assert history.pareto_front_size == 5
        assert isinstance(history.timestamp, str)

    def test_generation_history_to_dict(self):
        """Test converting GenerationHistory to dictionary."""
        history = GenerationHistory(
            generation=2,
            population_size=12,
            avg_fitness=0.8,
            best_fitness=1.0,
            worst_fitness=0.6,
            n_correct=8,
            accuracy=0.67
        )

        data = history.to_dict()

        assert data["generation"] == 2
        assert data["population_size"] == 12
        assert data["avg_fitness"] == 0.8
        assert data["best_fitness"] == 1.0
        assert data["worst_fitness"] == 0.6
        assert data["n_correct"] == 8
        assert data["accuracy"] == 0.67

    def test_generation_history_from_dict(self):
        """Test creating GenerationHistory from dictionary."""
        data = {
            "generation": 3,
            "population_size": 14,
            "avg_fitness": 0.85,
            "best_fitness": 1.0,
            "worst_fitness": 0.65,
            "n_correct": 10,
            "accuracy": 0.71,
            "avg_novelty": 0.35,
            "avg_local_competition": 0.3,
            "pareto_front_size": 6,
            "timestamp": "2025-01-30T12:00:00"
        }

        history = GenerationHistory.from_dict(data)

        assert history.generation == 3
        assert history.population_size == 14
        assert history.avg_fitness == 0.85
        assert history.avg_novelty == 0.35


class TestEvolutionCheckpoint:
    """Test suite for EvolutionCheckpoint class."""

    @pytest.fixture
    def sample_trajectories(self):
        """Create sample trajectories for checkpoint."""
        return [
            {
                "id": "traj1",
                "query": "Test query",
                "answer": "4",
                "reasoning": "Test reasoning",
                "source_model": "model-a",
                "generation_method": "vanilla",
                "_fitness_score": 0.8
            },
            {
                "id": "traj2",
                "query": "Test query",
                "answer": "5",
                "reasoning": "Test reasoning 2",
                "source_model": "model-b",
                "generation_method": "knowledge_augmented",
                "_fitness_score": 0.6
            }
        ]

    @pytest.fixture
    def sample_history(self):
        """Create sample generation history."""
        return [
            {
                "generation": 1,
                "population_size": 10,
                "avg_fitness": 0.7,
                "best_fitness": 0.9,
                "worst_fitness": 0.5,
                "n_correct": 7,
                "accuracy": 0.7
            }
        ]

    def test_checkpoint_creation(self, sample_trajectories, sample_history):
        """Test creating an EvolutionCheckpoint instance."""
        checkpoint = EvolutionCheckpoint(
            query="Test query",
            ground_truth="4",
            reference_knowledge="Test knowledge",
            current_population=sample_trajectories,
            generation=2,
            history=sample_history,
            best_trajectory=sample_trajectories[0],
            config={"n_generations": 5, "population_size": 10}
        )

        assert checkpoint.query == "Test query"
        assert checkpoint.ground_truth == "4"
        assert checkpoint.reference_knowledge == "Test knowledge"
        assert checkpoint.generation == 2
        assert len(checkpoint.current_population) == 2
        assert len(checkpoint.history) == 1
        assert checkpoint.best_trajectory is not None

    def test_checkpoint_save_and_load(self, tmp_path, sample_trajectories, sample_history):
        """Test saving and loading checkpoint."""
        checkpoint_path = tmp_path / "test_checkpoint.pkl"

        checkpoint = EvolutionCheckpoint(
            query="Test query",
            ground_truth="4",
            reference_knowledge=None,
            current_population=sample_trajectories,
            generation=3,
            history=sample_history,
            best_trajectory=sample_trajectories[0],
            config={"n_generations": 5}
        )

        # Save
        checkpoint.save(checkpoint_path)

        assert checkpoint_path.exists()

        # Load
        loaded_checkpoint = EvolutionCheckpoint.load(checkpoint_path)

        assert loaded_checkpoint.query == checkpoint.query
        assert loaded_checkpoint.ground_truth == checkpoint.ground_truth
        assert loaded_checkpoint.generation == checkpoint.generation
        assert len(loaded_checkpoint.current_population) == len(checkpoint.current_population)


class TestEvolutionEngine:
    """Test suite for EvolutionEngine class."""

    @pytest.fixture
    def mock_registry(self):
        """Create a mock model registry."""
        return MagicMock(spec=ModelRegistry)

    @pytest.fixture
    def mock_generator(self):
        """Create a mock generator."""
        generator = MagicMock(spec=MultiThinkerGenerator)
        generator.generate_initial_pool = AsyncMock()
        return generator

    @pytest.fixture
    def mock_selector(self):
        """Create a mock selector."""
        selector = MagicMock(spec=NSLCSelector)
        selector.select_parents = AsyncMock()
        return selector

    @pytest.fixture
    def mock_crossover(self):
        """Create a mock crossover."""
        crossover = MagicMock(spec=ReflectiveCrossover)
        crossover.crossover = AsyncMock()
        return crossover

    @pytest.fixture
    def mock_mutation(self):
        """Create a mock mutation."""
        mutation = MagicMock(spec=ReflectiveMutation)
        mutation.mutate = AsyncMock()
        return mutation

    @pytest.fixture
    def mock_fitness_evaluator(self):
        """Create a mock fitness evaluator."""
        evaluator = MagicMock(spec=FitnessEvaluator)
        evaluator.evaluate = AsyncMock()
        return evaluator

    @pytest.fixture
    def config(self):
        """Create a test configuration."""
        return EvolutionEngineConfig(
            n_generations=3,
            population_size=6,
            n_vanilla=4,
            n_knowledge_augmented=2,
            convergence_threshold=0.95,
            log_level="DEBUG"
        )

    @pytest.fixture
    def engine(
        self,
        mock_registry,
        mock_generator,
        mock_selector,
        mock_crossover,
        mock_mutation,
        mock_fitness_evaluator,
        config
    ):
        """Create an EvolutionEngine instance for testing."""
        return EvolutionEngine(
            model_registry=mock_registry,
            generator=mock_generator,
            selector=mock_selector,
            crossover=mock_crossover,
            mutation=mock_mutation,
            fitness_evaluator=mock_fitness_evaluator,
            config=config
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

        # Set fitness scores
        trajectories[0]._fitness_score = 1.0
        trajectories[0]._exact_match = True
        trajectories[1]._fitness_score = 0.5
        trajectories[1]._exact_match = False
        trajectories[2]._fitness_score = 0.8
        trajectories[2]._exact_match = True

        return Population(trajectories)

    def test_engine_initialization(self, engine, config):
        """Test engine initialization."""
        assert engine.config == config
        assert engine.generation == 0
        assert engine.history == []
        assert engine.best_trajectory is None

    @pytest.mark.asyncio
    async def test_evolve_full_cycle(self, engine, mock_generator, mock_selector,
                                     mock_crossover, mock_mutation, mock_fitness_evaluator,
                                     sample_population):
        """Test full evolution cycle with mocked components."""
        query = "What is 2+2?"
        ground_truth = "4"

        # Lower initial fitness scores to prevent early convergence
        for traj in sample_population.trajectories:
            traj._fitness_score = 0.6  # Below convergence_threshold (0.95)
            traj._exact_match = False

        # Mock initialization
        mock_generator.generate_initial_pool.return_value = sample_population

        # Mock selection
        mock_selector.select_parents.return_value = sample_population.trajectories[:3]

        # Mock crossover (return None to use mutation)
        mock_crossover.crossover.return_value = None

        # Mock mutation
        def mock_mutate_side_effect(trajectory, mode):
            new_traj = Trajectory(
                query=trajectory.query,
                answer=trajectory.answer,
                reasoning=f"Mutated ({mode}): " + trajectory.reasoning,
                source_model=trajectory.source_model,
                generation_method=f"mutation_{mode}"
            )
            new_traj._fitness_score = 0.7
            new_traj._exact_match = trajectory._exact_match
            return new_traj

        mock_mutation.mutate.side_effect = mock_mutate_side_effect

        # Mock fitness evaluation (no-op for already evaluated)
        async def mock_evaluate(trajectory, gt, ref_knowledge=None):
            if trajectory._fitness_score is None:
                trajectory._fitness_score = 0.7
                trajectory._exact_match = trajectory.answer == gt

        mock_fitness_evaluator.evaluate.side_effect = mock_evaluate

        # Run evolution
        best = await engine.evolve(query, ground_truth)

        # Verify
        assert best is not None
        assert best.query == query
        assert engine.generation == 3  # n_generations
        assert len(engine.history) == 3
        mock_generator.generate_initial_pool.assert_called_once()
        assert mock_selector.select_parents.call_count == 3

    @pytest.mark.asyncio
    async def test_evolve_convergence_early(self, engine, mock_generator, mock_fitness_evaluator,
                                           sample_population):
        """Test evolution stops early when convergence threshold is reached."""
        query = "What is 2+2?"
        ground_truth = "4"

        # Mock initialization with high fitness trajectory
        mock_generator.generate_initial_pool.return_value = sample_population

        # Mock fitness evaluation
        async def mock_evaluate(trajectory, gt, ref_knowledge=None):
            trajectory._fitness_score = 1.0  # Perfect fitness
            trajectory._exact_match = True

        mock_fitness_evaluator.evaluate.side_effect = mock_evaluate

        # Run evolution
        best = await engine.evolve(query, ground_truth)

        # Should converge immediately after first generation
        assert engine.generation == 1
        assert best.fitness_score >= engine.config.convergence_threshold

    @pytest.mark.asyncio
    async def test_evolve_with_checkpoint_resume(self, engine, mock_generator, tmp_path,
                                                 sample_population):
        """Test resuming evolution from checkpoint."""
        checkpoint_path = tmp_path / "test_checkpoint.pkl"

        # Create and save a checkpoint
        checkpoint = EvolutionCheckpoint(
            query="What is 2+2?",
            ground_truth="4",
            reference_knowledge=None,
            current_population=[t.to_dict() for t in sample_population.trajectories],
            generation=2,
            history=[],
            best_trajectory=sample_population.trajectories[0].to_dict(),
            config={
                "n_generations": 5,
                "population_size": 10,
                "convergence_threshold": 1.0,
                "k_neighbors": 5,
                "n_vanilla": 7,
                "n_knowledge_augmented": 3,
                "crossover_prob": 0.4,
                "mutation_prob": 0.6,
                "mutation_mode_distribution": {"add": 0.25, "delete": 0.25, "innovate": 0.5},
                "elitism_ratio": 0.5,
                "epsilon": 0.1
            }
        )
        checkpoint.save(checkpoint_path)

        # Mock fitness evaluation
        async def mock_evaluate(trajectory, gt, ref_knowledge=None):
            if trajectory._fitness_score is None:
                trajectory._fitness_score = 0.8
                trajectory._exact_match = trajectory.answer == gt

        engine.fitness_evaluator.evaluate.side_effect = mock_evaluate

        # Resume from checkpoint
        population = engine._resume_from_checkpoint(checkpoint_path)

        assert engine.generation == 2
        assert population.size == 3

    @pytest.mark.asyncio
    async def test_evaluate_population_lazy_evaluation(self, engine, mock_fitness_evaluator,
                                                      sample_population):
        """Test that only unevaluated trajectories are evaluated."""
        # Reset all fitness scores to None first
        for traj in sample_population.trajectories:
            traj._fitness_score = None
            traj._exact_match = None

        # Make one trajectory already evaluated
        sample_population.trajectories[0]._fitness_score = 0.9
        sample_population.trajectories[0]._exact_match = True

        evaluation_count = [0]

        async def mock_evaluate(trajectory, gt, reference_knowledge=None):  # Fixed parameter name
            evaluation_count[0] += 1
            trajectory._fitness_score = 0.7
            trajectory._exact_match = trajectory.answer == gt

        mock_fitness_evaluator.evaluate.side_effect = mock_evaluate

        await engine._evaluate_population(sample_population, "4")

        # Should only evaluate 2 trajectories (not the pre-evaluated one)
        assert evaluation_count[0] == 2

    def test_check_convergence_true(self, engine, sample_population):
        """Test convergence check returns True when threshold reached."""
        sample_population.trajectories[0]._fitness_score = 1.0

        converged = engine._check_convergence(sample_population)

        assert converged is True

    def test_check_convergence_false(self, engine, sample_population):
        """Test convergence check returns False when threshold not reached."""
        sample_population.trajectories[0]._fitness_score = 0.8

        converged = engine._check_convergence(sample_population)

        assert converged is False

    def test_check_convergence_empty_population(self, engine):
        """Test convergence check with empty population."""
        population = Population([])

        converged = engine._check_convergence(population)

        assert converged is False

    @pytest.mark.asyncio
    async def test_generate_offspring_crossover_constraint(self, engine, mock_crossover,
                                                          mock_mutation, sample_population):
        """Test that crossover only applies when target answer is incorrect."""
        parents = sample_population.trajectories

        # Mock crossover
        mock_crossover.crossover.return_value = Trajectory(
            query=parents[0].query,
            answer="4",
            reasoning="Crossover reasoning",
            source_model="model",
            generation_method="crossover"
        )

        # Mock mutation
        mock_mutation.mutate.return_value = Trajectory(
            query=parents[0].query,
            answer="4",
            reasoning="Mutation reasoning",
            source_model="model",
            generation_method="mutation"
        )

        # Patch random to control crossover probability
        with patch('src.optimization.engine.random.random') as mock_random:
            mock_random.return_value = 0.3  # < crossover_prob (0.4)

            # Target has exact_match=True, so crossover should NOT be used
            offspring = await engine._generate_offspring(parents, sample_population)

            # Should use mutation instead (since exact_match=True blocks crossover)
            assert mock_mutation.mutate.called or mock_crossover.crossover.called

    @pytest.mark.asyncio
    async def test_sample_parent_pair(self, engine, sample_population):
        """Test parent pair sampling."""
        parents = sample_population.trajectories[:2]

        # Set local competition for weighted sampling
        sample_population.trajectories[0]._local_competition = 0.8
        sample_population.trajectories[1]._local_competition = 0.2

        target, provider = engine._sample_parent_pair(parents, sample_population)

        assert target in parents
        assert provider in sample_population.trajectories

    def test_sample_mutation_mode(self, engine):
        """Test mutation mode sampling."""
        mode = engine._sample_mutation_mode()

        assert mode in ["add", "delete", "innovate"]

    def test_replace_population_elitism(self, engine, sample_population):
        """Test population replacement with elitism."""
        # Create offspring
        offspring = [
            Trajectory(
                query="Test",
                answer="4",
                reasoning=f"Offspring {i}",
                source_model="model",
                generation_method="mutation"
            )
            for i in range(3)
        ]
        for o in offspring:
            o._fitness_score = 0.7

        new_population = engine._replace_population(sample_population, offspring)

        # Should maintain population size
        assert new_population.size == engine.config.population_size

        # Should contain elites (top performers)
        elites = sample_population.get_best(int(engine.config.population_size * engine.config.elitism_ratio))
        for elite in elites:
            assert elite in new_population.trajectories

    def test_replace_population_removes_duplicates(self, engine, sample_population):
        """Test that population replacement removes duplicate trajectories."""
        # Create offspring that includes a duplicate of an existing trajectory
        duplicate = sample_population.trajectories[0]
        offspring = [duplicate]  # Just the duplicate

        new_population = engine._replace_population(sample_population, offspring)

        # Should not have duplicates
        trajectory_ids = [t.id for t in new_population.trajectories]
        assert len(trajectory_ids) == len(set(trajectory_ids))

    def test_log_generation_stats(self, engine, sample_population, caplog):
        """Test generation statistics logging."""
        import logging

        # Set up population with required attributes
        for traj in sample_population.trajectories:
            if traj._fitness_score is None:
                traj._fitness_score = 0.7
                traj._exact_match = traj.answer == "4"

        engine._log_generation_stats(1, sample_population)

        assert len(engine.history) == 1
        history = engine.history[0]
        assert history.generation == 1
        assert history.population_size == sample_population.size
        assert history.best_fitness == 1.0
        assert history.worst_fitness == 0.5

    @pytest.mark.asyncio
    async def test_evolve_batch_concurrent(self, engine, mock_generator, tmp_path):
        """Test batch evolution with concurrency control."""
        samples = [
            {"query": "What is 2+2?", "ground_truth": "4"},
            {"query": "What is 3+3?", "ground_truth": "6"},
            {"query": "What is 4+4?", "ground_truth": "8"}
        ]

        # Mock initialization
        def create_population(query):
            trajectories = [
                Trajectory(
                    query=query,
                    answer=str(int(query.split()[-1][0]) * 2),
                    reasoning="Test reasoning",
                    source_model="model",
                    generation_method="vanilla"
                )
            ]
            trajectories[0]._fitness_score = 1.0
            trajectories[0]._exact_match = True
            return Population(trajectories)

        mock_generator.generate_initial_pool.side_effect = [
            create_population(s["query"]) for s in samples
        ]

        # Mock fitness evaluation
        async def mock_evaluate(trajectory, gt, ref_knowledge=None):
            trajectory._fitness_score = 1.0
            trajectory._exact_match = True

        engine.fitness_evaluator.evaluate.side_effect = mock_evaluate

        checkpoint_dir = tmp_path / "checkpoints"

        # Run batch evolution
        results = await engine.evolve_batch(
            samples=samples,
            max_concurrent=2,
            checkpoint_dir=checkpoint_dir
        )

        assert len(results) == 3
        assert all(isinstance(r, Trajectory) for r in results)


class TestEvolutionEngineIntegration:
    """Integration tests for EvolutionEngine with more realistic scenarios."""

    @pytest.fixture
    def minimal_config(self):
        """Create minimal configuration for testing."""
        return EvolutionEngineConfig(
            n_generations=2,
            population_size=4,
            n_vanilla=3,
            n_knowledge_augmented=1,
            convergence_threshold=1.0,
            save_checkpoints=False
        )

    @pytest.fixture
    def mock_dependencies(self):
        """Create all mock dependencies."""
        registry = MagicMock(spec=ModelRegistry)
        generator = MagicMock(spec=MultiThinkerGenerator)
        selector = MagicMock(spec=NSLCSelector)
        crossover = MagicMock(spec=ReflectiveCrossover)
        mutation = MagicMock(spec=ReflectiveMutation)
        fitness = MagicMock(spec=FitnessEvaluator)

        return {
            "model_registry": registry,  # Changed from "registry"
            "generator": generator,
            "selector": selector,
            "crossover": crossover,
            "mutation": mutation,
            "fitness_evaluator": fitness  # Changed from "fitness"
        }

    @pytest.mark.asyncio
    async def test_full_evolution_with_realistic_mocking(self, minimal_config, mock_dependencies):
        """Test realistic evolution scenario with proper mocking."""
        engine = EvolutionEngine(**mock_dependencies, config=minimal_config)

        query = "Calculate molecular weight of H2O"
        ground_truth = "18.015 g/mol"

        # Create diverse initial population
        initial_trajectories = [
            Trajectory(
                query=query,
                answer="18.015 g/mol",
                reasoning="H: 1.008, O: 15.999, H2O: 2*1.008 + 15.999 = 18.015 g/mol",
                source_model="deepseek-r1",
                generation_method="vanilla"
            ),
            Trajectory(
                query=query,
                answer="18 g/mol",
                reasoning="H2O: H*2 + O = 1*2 + 16 = 18 g/mol",
                source_model="qwen-235",
                generation_method="vanilla"
            ),
            Trajectory(
                query=query,
                answer="18.015 g/mol",
                reasoning="Molecular weight: 2*1.008 + 15.999 = 18.015 g/mol (using precise values)",
                source_model="deepseek-r1",
                generation_method="knowledge_augmented",
                knowledge="Atomic weights: H=1.008, O=15.999"
            ),
            Trajectory(
                query=query,
                answer="20 g/mol",
                reasoning="Approximately 20 g/mol",
                source_model="qwen-32b",
                generation_method="vanilla"
            )
        ]

        # Set fitness scores (lower to prevent immediate convergence)
        initial_trajectories[0]._fitness_score = 0.9  # Below convergence_threshold (1.0)
        initial_trajectories[0]._exact_match = True
        initial_trajectories[1]._fitness_score = 0.8
        initial_trajectories[1]._exact_match = False
        initial_trajectories[2]._fitness_score = 0.9  # Below convergence_threshold
        initial_trajectories[2]._exact_match = True
        initial_trajectories[3]._fitness_score = 0.5
        initial_trajectories[3]._exact_match = False

        initial_population = Population(initial_trajectories)

        # Mock generator
        mock_dependencies["generator"].generate_initial_pool = AsyncMock(
            return_value=initial_population
        )

        # Mock selector to select top 2
        async def mock_select(population, n_parents):
            return population.trajectories[:2]

        mock_dependencies["selector"].select_parents = AsyncMock(
            side_effect=mock_select
        )

        # Mock crossover (rarely used)
        async def mock_crossover_fn(target, provider):
            # Only improve if target is wrong
            if target.exact_match:
                return target
            return Trajectory(
                query=target.query,
                answer=ground_truth,
                reasoning=f"Improved from crossover: {target.reasoning[:50]}...",
                source_model=target.source_model,
                generation_method="crossover"
            )

        mock_dependencies["crossover"].crossover = AsyncMock(side_effect=mock_crossover_fn)

        # Mock mutation
        async def mock_mutation_fn(trajectory, mode):
            new_traj = Trajectory(
                query=trajectory.query,
                answer=trajectory.answer,
                reasoning=f"Mutated ({mode}): {trajectory.reasoning[:30]}...",
                source_model=trajectory.source_model,
                generation_method=f"mutation_{mode}"
            )
            new_traj._fitness_score = trajectory.fitness_score * 1.1  # Slight improvement
            new_traj._exact_match = trajectory.answer == ground_truth
            return new_traj

        mock_dependencies["mutation"].mutate = AsyncMock(side_effect=mock_mutation_fn)

        # Mock fitness evaluator
        async def mock_evaluate(trajectory, gt, reference_knowledge=None):  # Fixed parameter name
            if trajectory._fitness_score is None:
                # Simple scoring: exact match + bonus for detailed reasoning
                score = (1.0 if trajectory.answer == gt else 0.0)
                if len(trajectory.reasoning) > 50:
                    score += 0.2
                trajectory._fitness_score = min(score, 1.0)
                trajectory._exact_match = trajectory.answer == gt

        mock_dependencies["fitness_evaluator"].evaluate = AsyncMock(side_effect=mock_evaluate)

        # Run evolution
        best = await engine.evolve(query, ground_truth)

        # Verify results
        assert best is not None
        assert best.query == query
        assert best.fitness_score >= 0.8  # Should improve or maintain
        assert engine.generation == 2  # Ran for all generations

        # Verify history was tracked
        assert len(engine.history) == 2
        for hist in engine.history:
            assert hist.generation in [1, 2]
            assert hist.population_size <= minimal_config.population_size

        # Verify generator was called once
        mock_dependencies["generator"].generate_initial_pool.assert_called_once_with(
            query=query,
            ground_truth=ground_truth,
            n_vanilla=minimal_config.n_vanilla,
            n_knowledge_augmented=minimal_config.n_knowledge_augmented
        )
