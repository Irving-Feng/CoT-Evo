"""
Evaluation framework for CoT-Evo.

This module provides evaluation functions for different datasets and task types,
including ChemCoT (chemistry reasoning) and BioProBench (biology protocol reasoning).
"""

from .chemcot import (
    ChemCoTEvaluator,
    ChemCoTMetrics,
    create_chemcot_evaluator
)

from .bioprobench import (
    BioProBenchEvaluator,
    BioProBenchMetrics,
    create_bioprobench_evaluator,
    compute_classification_metrics,
    calculate_kendall_tau
)

__all__ = [
    # ChemCoT
    "ChemCoTEvaluator",
    "ChemCoTMetrics",
    "create_chemcot_evaluator",

    # BioProBench
    "BioProBenchEvaluator",
    "BioProBenchMetrics",
    "create_bioprobench_evaluator",
    "compute_classification_metrics",
    "calculate_kendall_tau",
]
