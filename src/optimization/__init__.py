"""
Optimization module for CoT-Evo framework.

This module contains the evolution engine that implements Algorithm 1 from the paper,
integrating all core algorithm modules (initialization, selection, variation) into a
cohesive evolutionary loop.
"""

from .engine import EvolutionEngine
from .config import EvolutionEngineConfig, GenerationHistory, EvolutionCheckpoint
from .batch_processor import BatchProcessor, BatchConfig, BatchResult, create_batch_processor

__all__ = [
    "EvolutionEngine",
    "EvolutionEngineConfig",
    "GenerationHistory",
    "EvolutionCheckpoint",
    "BatchProcessor",
    "BatchConfig",
    "BatchResult",
    "create_batch_processor",
]
