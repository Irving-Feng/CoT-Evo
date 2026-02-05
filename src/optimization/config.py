"""
Configuration and data classes for CoT-Evo evolution engine.
"""

import logging
import os
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Union
from pathlib import Path
import yaml
import pickle
from datetime import datetime

logger = logging.getLogger(__name__)


def parse_ratio(value: Union[str, int, float]) -> float:
    """
    Parse a ratio value that can be a fraction string, int, or float.

    Args:
        value: Ratio value as:
               - Fraction string like "2/3", "1/3"
               - Integer like 2 (will be converted to float)
               - Float like 0.6667

    Returns:
        Float representation of the ratio

    Examples:
        >>> parse_ratio("2/3")
        0.6666666666666666
        >>> parse_ratio("1/3")
        0.3333333333333333
        >>> parse_ratio(0.5)
        0.5
        >>> parse_ratio(1)
        1.0
    """
    if isinstance(value, float):
        return value

    if isinstance(value, int):
        return float(value)

    if isinstance(value, str):
        value = value.strip()

        # Try to parse as fraction "a/b"
        if '/' in value:
            try:
                numerator, denominator = value.split('/')
                return float(numerator) / float(denominator)
            except (ValueError, ZeroDivisionError):
                logger.warning(f"Invalid fraction format: {value}, falling back to 0.0")
                return 0.0

        # Try to parse as float
        try:
            return float(value)
        except ValueError:
            logger.warning(f"Invalid ratio format: {value}, falling back to 0.0")
            return 0.0

    logger.warning(f"Unknown ratio type: {type(value)}, falling back to 0.0")
    return 0.0


@dataclass
class EvolutionEngineConfig:
    """
    Configuration for the evolution engine.

    Contains all hyperparameters for controlling the evolutionary optimization process.
    """

    # Evolution hyperparameters
    n_generations: int = 5
    population_size: int = 10
    convergence_threshold: float = 1.0
    k_neighbors: int = 5

    # Initialization parameters
    n_vanilla: int = 7
    n_knowledge_augmented: int = 3

    # Variation parameters
    crossover_prob: float = 0.4
    mutation_prob: float = 0.6
    mutation_mode_distribution: Dict[str, float] = field(default_factory=lambda: {
        "add": 0.25, "delete": 0.25, "innovate": 0.5
    })

    # Selection parameters
    elitism_ratio: float = 0.5
    epsilon: float = 0.1

    # Retry and error handling
    max_retries: int = 3
    retry_backoff: float = 2.0

    # Checkpoint
    save_checkpoints: bool = True
    checkpoint_interval: int = 1

    # Logging
    log_level: str = "INFO"
    log_generation_stats: bool = True

    # Ratio parameters (for initialization)
    initial_vanilla_ratio: Union[str, float] = 0.7
    initial_knowledge_augmented_ratio: Union[str, float] = 0.3

    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.n_generations < 1:
            raise ValueError("n_generations must be at least 1")
        if self.population_size < 2:
            raise ValueError("population_size must be at least 2")
        if not 0.0 <= self.crossover_prob <= 1.0:
            raise ValueError("crossover_prob must be between 0 and 1")
        if not 0.0 <= self.mutation_prob <= 1.0:
            raise ValueError("mutation_prob must be between 0 and 1")
        if abs(self.crossover_prob + self.mutation_prob - 1.0) > 0.01:
            logger.warning(
                f"crossover_prob + mutation_prob = {self.crossover_prob + self.mutation_prob}, "
                "should sum to 1.0"
            )
        if not 0.0 <= self.elitism_ratio <= 1.0:
            raise ValueError("elitism_ratio must be between 0 and 1")

        # Validate mutation mode distribution
        total_prob = sum(self.mutation_mode_distribution.values())
        if abs(total_prob - 1.0) > 0.01:
            raise ValueError(f"mutation_mode_distribution must sum to 1.0, got {total_prob}")

        # Parse and validate ratio parameters
        vanilla_ratio = parse_ratio(self.initial_vanilla_ratio)
        knowledge_ratio = parse_ratio(self.initial_knowledge_augmented_ratio)

        # Check if ratios sum to approximately 1.0
        if abs(vanilla_ratio + knowledge_ratio - 1.0) > 0.01:
            logger.warning(
                f"initial_vanilla_ratio + initial_knowledge_augmented_ratio = "
                f"{vanilla_ratio + knowledge_ratio}, should sum to 1.0"
            )

        # Update n_vanilla and n_knowledge_augmented based on ratios and population_size
        if not hasattr(self, '_ratios_parsed'):
            self.n_vanilla = max(1, int(self.population_size * vanilla_ratio))
            self.n_knowledge_augmented = max(1, self.population_size - self.n_vanilla)
            self._ratios_parsed = True

    @classmethod
    def from_yaml(cls, config_path: Path) -> "EvolutionEngineConfig":
        """
        Load configuration from YAML file.

        Args:
            config_path: Path to YAML configuration file

        Returns:
            EvolutionEngineConfig instance
        """
        with open(config_path) as f:
            config = yaml.safe_load(f)

        # Extract nested parameters if present
        evolution_config = config.get("evolution", config)

        # Handle ratio parameters
        if "initial_vanilla_ratio" in evolution_config:
            evolution_config["initial_vanilla_ratio"] = parse_ratio(evolution_config["initial_vanilla_ratio"])
        if "initial_knowledge_augmented_ratio" in evolution_config:
            evolution_config["initial_knowledge_augmented_ratio"] = parse_ratio(evolution_config["initial_knowledge_augmented_ratio"])

        return cls(**evolution_config)

    @classmethod
    def from_env(cls) -> "EvolutionEngineConfig":
        """
        Load configuration from environment variables.

        Reads the following environment variables:
        - MAX_GENERATIONS: n_generations
        - POPULATION_SIZE: population_size
        - N_NEIGHBORS: k_neighbors
        - CROSSOVER_RATE: crossover_prob
        - MUTATION_RATE: mutation_prob
        - INITIAL_VANILLA_RATIO: initial_vanilla_ratio (supports fractions like "2/3")
        - INITIAL_KNOWLEDGE_AUGMENTED_RATIO: initial_knowledge_augmented_ratio (supports fractions like "1/3")
        - EPSILON: epsilon
        - MAX_GENERATIONS: n_generations
        - CONVERGENCE_PATIENCE: (for future use)
        - LOG_LEVEL: log_level

        Returns:
            EvolutionEngineConfig instance
        """
        return cls(
            n_generations=int(os.getenv("MAX_GENERATIONS", "5")),
            population_size=int(os.getenv("POPULATION_SIZE", "10")),
            k_neighbors=int(os.getenv("N_NEIGHBORS", "5")),
            crossover_prob=float(os.getenv("CROSSOVER_RATE", "0.4")),
            mutation_prob=float(os.getenv("MUTATION_RATE", "0.6")),
            initial_vanilla_ratio=os.getenv("INITIAL_VANILLA_RATIO", "0.7"),
            initial_knowledge_augmented_ratio=os.getenv("INITIAL_KNOWLEDGE_AUGMENTED_RATIO", "0.3"),
            epsilon=float(os.getenv("EPSILON", "0.1")),
            log_level=os.getenv("LOG_LEVEL", "INFO"),
        )

    def to_yaml(self, save_path: Path) -> None:
        """
        Save configuration to YAML file.

        Args:
            save_path: Path where to save the configuration
        """
        config_dict = {
            "evolution": {
                "n_generations": self.n_generations,
                "population_size": self.population_size,
                "convergence_threshold": self.convergence_threshold,
                "k_neighbors": self.k_neighbors,
                "n_vanilla": self.n_vanilla,
                "n_knowledge_augmented": self.n_knowledge_augmented,
                "crossover_prob": self.crossover_prob,
                "mutation_prob": self.mutation_prob,
                "mutation_mode_distribution": self.mutation_mode_distribution,
                "elitism_ratio": self.elitism_ratio,
                "epsilon": self.epsilon,
                "max_retries": self.max_retries,
                "retry_backoff": self.retry_backoff,
                "save_checkpoints": self.save_checkpoints,
                "checkpoint_interval": self.checkpoint_interval,
                "log_level": self.log_level,
                "log_generation_stats": self.log_generation_stats,
            }
        }

        with open(save_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)

        logger.info(f"Saved configuration to {save_path}")


@dataclass
class GenerationHistory:
    """
    Statistics for a single generation of evolution.

    Tracks key metrics to monitor evolution progress.
    """

    generation: int
    population_size: int
    avg_fitness: float
    best_fitness: float
    worst_fitness: float
    n_correct: int
    accuracy: float
    avg_novelty: Optional[float] = None
    avg_local_competition: Optional[float] = None
    pareto_front_size: int = 0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "generation": self.generation,
            "population_size": self.population_size,
            "avg_fitness": self.avg_fitness,
            "best_fitness": self.best_fitness,
            "worst_fitness": self.worst_fitness,
            "n_correct": self.n_correct,
            "accuracy": self.accuracy,
            "avg_novelty": self.avg_novelty,
            "avg_local_competition": self.avg_local_competition,
            "pareto_front_size": self.pareto_front_size,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GenerationHistory":
        """Create instance from dictionary."""
        return cls(**data)


@dataclass
class EvolutionCheckpoint:
    """
    Checkpoint data for resuming evolution.

    Contains all necessary state to resume evolution from a previous point.
    """

    query: str
    ground_truth: str
    reference_knowledge: Optional[str]
    current_population: List[Dict[str, Any]]  # Serialized trajectories
    generation: int
    history: List[Dict[str, Any]]
    best_trajectory: Optional[Dict[str, Any]]
    config: Dict[str, Any]  # EvolutionEngineConfig
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def save(self, checkpoint_path: Path) -> None:
        """
        Save checkpoint to disk.

        Args:
            checkpoint_path: Path where to save the checkpoint
        """
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        with open(checkpoint_path, 'wb') as f:
            pickle.dump(self, f)

        logger.info(f"Saved checkpoint at generation {self.generation} to {checkpoint_path}")

    @classmethod
    def load(cls, checkpoint_path: Path) -> "EvolutionCheckpoint":
        """
        Load checkpoint from disk.

        Args:
            checkpoint_path: Path to checkpoint file

        Returns:
            EvolutionCheckpoint instance
        """
        with open(checkpoint_path, 'rb') as f:
            checkpoint = pickle.load(f)

        logger.info(f"Loaded checkpoint from generation {checkpoint.generation} at {checkpoint_path}")
        return checkpoint
