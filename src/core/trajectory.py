"""
Core data structures for CoT-Evo framework.

This module defines the Trajectory class, which represents a single reasoning chain
with its associated metadata, evaluation scores, and behavioral characteristics.
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any
import numpy as np
import hashlib
import json


@dataclass
class Trajectory:
    """
    Represents a single reasoning trajectory (Chain-of-Thought).

    This class encapsulates all information about a reasoning chain, including
    the input query, generated reasoning process, final answer, and various
    evaluation scores and metadata.

    Attributes:
        query: The input question or problem
        answer: The final answer extracted from the reasoning
        reasoning: The full chain-of-thought reasoning process
        knowledge: Optional knowledge snippet used for knowledge-augmented generation
        source_model: Name of the model that generated this trajectory
        generation_method: How this trajectory was generated
            (vanilla/knowledge_augmented/crossover/mutation)
        metadata: Additional metadata as a dictionary

    Evaluation Attributes (computed lazily):
        _fitness_score: Combined fitness score (formula 6)
        _exact_match: Whether the answer exactly matches ground truth (formula 3)
        _length_score: Length appropriateness score (formula 4)
        _knowledge_score: Knowledge usage correctness (formula 5, scale 1-5)

    NSLC Attributes (for novelty-driven selection):
        _embedding: Behavioral embedding vector (formula 7)
        _novelty_score: Novelty score based on k-NN distance (formula 8)
        _local_competition: Local competition score (formula 9)
    """

    # Core content
    query: str
    answer: str
    reasoning: str

    # Generation metadata
    knowledge: Optional[str] = None
    source_model: str = ""
    generation_method: str = "vanilla"  # vanilla, knowledge_augmented, crossover, mutation
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Evaluation scores (computed lazily)
    _fitness_score: Optional[float] = None
    _exact_match: Optional[bool] = None
    _length_score: Optional[float] = None
    _knowledge_score: Optional[int] = None

    # NSLC attributes
    _embedding: Optional[np.ndarray] = None
    _novelty_score: Optional[float] = None
    _local_competition: Optional[float] = None

    def __post_init__(self):
        """Compute a unique ID for this trajectory after initialization."""
        if "id" not in self.metadata:
            self.metadata["id"] = self._compute_id()

    def _compute_id(self) -> str:
        """Compute a unique hash-based ID for this trajectory."""
        content = f"{self.query}|{self.reasoning}|{self.source_model}"
        return hashlib.md5(content.encode()).hexdigest()[:16]

    @property
    def id(self) -> str:
        """Get the unique ID of this trajectory."""
        return self.metadata["id"]

    @property
    def fitness_score(self) -> Optional[float]:
        """Get the fitness score if computed."""
        return self._fitness_score

    @property
    def exact_match(self) -> Optional[bool]:
        """Get whether the answer exactly matches ground truth."""
        return self._exact_match

    @property
    def length_score(self) -> Optional[float]:
        """Get the length appropriateness score."""
        return self._length_score

    @property
    def knowledge_score(self) -> Optional[int]:
        """Get the knowledge usage correctness score (1-5)."""
        return self._knowledge_score

    @property
    def embedding(self) -> Optional[np.ndarray]:
        """Get the behavioral embedding vector."""
        return self._embedding

    @property
    def novelty_score(self) -> Optional[float]:
        """Get the novelty score."""
        return self._novelty_score

    @property
    def local_competition(self) -> Optional[float]:
        """Get the local competition score."""
        return self._local_competition

    def set_fitness_score(self, score: float) -> None:
        """Set the fitness score."""
        self._fitness_score = score

    def set_exact_match(self, is_match: bool) -> None:
        """Set the exact match result."""
        self._exact_match = is_match

    def set_length_score(self, score: float) -> None:
        """Set the length appropriateness score."""
        self._length_score = score

    def set_knowledge_score(self, score: int) -> None:
        """Set the knowledge usage correctness score (1-5)."""
        if not 1 <= score <= 5:
            raise ValueError(f"Knowledge score must be between 1 and 5, got {score}")
        self._knowledge_score = score

    def set_embedding(self, embedding: np.ndarray) -> None:
        """Set the behavioral embedding vector."""
        self._embedding = embedding

    def set_novelty_score(self, score: float) -> None:
        """Set the novelty score."""
        self._novelty_score = score

    def set_local_competition(self, score: float) -> None:
        """Set the local competition score."""
        self._local_competition = score

    def has_embedding(self) -> bool:
        """Check if this trajectory has a computed embedding."""
        return self._embedding is not None

    def is_evaluated(self) -> bool:
        """Check if this trajectory has been evaluated (has fitness score)."""
        return self._fitness_score is not None

    def is_correct(self) -> Optional[bool]:
        """Check if the answer is correct (requires exact match to be evaluated)."""
        return self._exact_match

    def reasoning_length(self) -> int:
        """Estimate the length of reasoning in tokens (rough word count)."""
        return len(self.reasoning.split())

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert trajectory to a dictionary for serialization.

        Excludes numpy arrays and other non-serializable objects.
        """
        data = asdict(self)

        # Remove private attributes that are not serializable
        for key in list(data.keys()):
            if key.startswith("_") and key not in ["_exact_match", "_knowledge_score", "_fitness_score", "_length_score"]:
                del data[key]

        # Convert numpy arrays to lists if present
        if "_embedding" in data and data["_embedding"] is not None:
            data["_embedding"] = data["_embedding"].tolist()

        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Trajectory":
        """
        Create a Trajectory from a dictionary.

        Args:
            data: Dictionary containing trajectory data

        Returns:
            Trajectory instance
        """
        # Convert embedding back to numpy array
        if "_embedding" in data and data["_embedding"] is not None:
            data["_embedding"] = np.array(data["_embedding"])

        return cls(**data)

    def __repr__(self) -> str:
        """String representation of the trajectory."""
        fitness_str = f"{self._fitness_score:.3f}" if self._fitness_score is not None else "None"
        return (
            f"Trajectory(id={self.id}, "
            f"model={self.source_model}, "
            f"method={self.generation_method}, "
            f"fitness={fitness_str}, "
            f"correct={self._exact_match})"
        )
