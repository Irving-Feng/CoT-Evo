"""
Evolution logger for tracking detailed CoT evolution process.

This module provides detailed logging of every stage of the evolutionary algorithm:
- Initial population generation
- Evaluation results
- Crossover operations
- Mutation operations
- Final best trajectory

Logs are organized by sample and generation for easy analysis.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

from ..core.trajectory import Trajectory


logger = logging.getLogger(__name__)


class EvolutionLogger:
    """
    Detailed logger for CoT evolution process.

    Records every stage of evolution with full CoT content for analysis.
    """

    def __init__(self, output_dir: Path, sample_id: str):
        """
        Initialize the evolution logger.

        Args:
            output_dir: Base output directory
            sample_id: Sample identifier (will be used for subdirectory name)
        """
        self.output_dir = output_dir / f"sample_{sample_id}"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.sample_id = sample_id

        logger.info(f"EvolutionLogger initialized for sample {sample_id} at {self.output_dir}")

    def log_initial_population(self, trajectories: List[Trajectory], gen: int):
        """
        Log initial population generation.

        Args:
            trajectories: List of generated trajectories
            gen: Generation number (0 for initial)
        """
        data = {
            "sample_id": self.sample_id,
            "generation": gen,
            "phase": "initial_population",
            "timestamp": datetime.now().isoformat(),
            "trajectories": [self._traj_to_dict(t, i) for i, t in enumerate(trajectories)],
            "statistics": self._compute_statistics(trajectories)
        }

        self._save_json(f"generation_{gen}_initial.json", data)
        logger.info(f"Logged initial population for generation {gen}: {len(trajectories)} trajectories")

    def log_evaluation(self, trajectories: List[Trajectory], gen: int):
        """
        Log evaluation results.

        Args:
            trajectories: List of evaluated trajectories
            gen: Generation number
        """
        data = {
            "sample_id": self.sample_id,
            "generation": gen,
            "phase": "evaluation",
            "timestamp": datetime.now().isoformat(),
            "population": [self._eval_to_dict(t, i) for i, t in enumerate(trajectories)],
            "statistics": self._compute_statistics(trajectories)
        }

        self._save_json(f"generation_{gen}_eval.json", data)
        logger.info(f"Logged evaluation for generation {gen}")

    def log_selection(self, selected: List[Trajectory], gen: int):
        """
        Log selection results.

        Args:
            selected: List of selected parent trajectories
            gen: Generation number
        """
        data = {
            "sample_id": self.sample_id,
            "generation": gen,
            "phase": "selection",
            "timestamp": datetime.now().isoformat(),
            "selected_parents": [self._traj_to_dict(t, i) for i, t in enumerate(selected)],
            "num_selected": len(selected)
        }

        self._save_json(f"generation_{gen}_selection.json", data)
        logger.info(f"Logged selection for generation {gen}: {len(selected)} parents")

    def log_crossover(self, operations: List[Dict], gen: int):
        """
        Log crossover operations.

        Args:
            operations: List of crossover operation details
            gen: Generation number
        """
        data = {
            "sample_id": self.sample_id,
            "generation": gen,
            "phase": "crossover",
            "timestamp": datetime.now().isoformat(),
            "operations": operations,
            "total_operations": len(operations),
            "successful_operations": sum(1 for op in operations if op.get("success", False))
        }

        self._save_json(f"generation_{gen}_crossover.json", data)
        logger.info(f"Logged crossover for generation {gen}: {len(operations)} operations")

    def log_mutation(self, operations: List[Dict], gen: int):
        """
        Log mutation operations.

        Args:
            operations: List of mutation operation details
            gen: Generation number
        """
        data = {
            "sample_id": self.sample_id,
            "generation": gen,
            "phase": "mutation",
            "timestamp": datetime.now().isoformat(),
            "operations": operations,
            "total_operations": len(operations),
            "successful_operations": sum(1 for op in operations if op.get("success", False))
        }

        self._save_json(f"generation_{gen}_mutation.json", data)
        logger.info(f"Logged mutation for generation {gen}: {len(operations)} operations")

    def log_new_generation(self, trajectories: List[Trajectory], gen: int):
        """
        Log the start of a new generation.

        Args:
            trajectories: List of trajectories in the new generation
            gen: Generation number
        """
        data = {
            "sample_id": self.sample_id,
            "generation": gen,
            "phase": "new_generation",
            "timestamp": datetime.now().isoformat(),
            "trajectories": [self._traj_to_dict(t, i) for i, t in enumerate(trajectories)],
            "population_size": len(trajectories),
            "statistics": self._compute_statistics(trajectories)
        }

        self._save_json(f"generation_{gen}_population.json", data)
        logger.info(f"Logged new generation {gen}: {len(trajectories)} trajectories")

    def log_final_best(self, trajectory: Trajectory, gen: int):
        """
        Log the final best trajectory.

        Args:
            trajectory: Best trajectory found
            gen: Final generation number
        """
        data = {
            "sample_id": self.sample_id,
            "final_generation": gen,
            "phase": "final_best",
            "timestamp": datetime.now().isoformat(),
            "best_trajectory": self._traj_to_dict(trajectory, 0, full=True),
            "summary": {
                "fitness": trajectory.fitness_score,
                "exact_match": trajectory.exact_match if trajectory.exact_match is not None else False,
                "reasoning_length": len(trajectory.reasoning) if trajectory.reasoning else 0,
                "source_model": trajectory.source_model,
                "generation_method": trajectory.generation_method
            }
        }

        self._save_json("final_best.json", data)
        logger.info(f"Logged final best trajectory with fitness {trajectory.fitness_score:.3f}")

    def log_early_stop(self, trajectory: Trajectory, generation: int, fitness: float):
        """
        Log early stopping event.

        Args:
            trajectory: Best trajectory that triggered early stop
            generation: Generation number when early stop occurred
            fitness: Fitness score that triggered early stop
        """
        data = {
            "sample_id": self.sample_id,
            "generation": generation,
            "phase": "early_stop",
            "timestamp": datetime.now().isoformat(),
            "best_trajectory": self._traj_to_dict(trajectory, 0, full=True),
            "early_stop_reason": f"Perfect fitness {fitness:.3f} achieved",
            "summary": {
                "fitness": fitness,
                "stopped_at_generation": generation,
                "max_generations": generation  # Early stop means we didn't complete all planned generations
            }
        }

        self._save_json(f"generation_{generation}_early_stop.json", data)
        logger.info(f"Logged early stop at generation {generation} with fitness {fitness:.3f}")

    def log_error(self, error_type: str, error_message: str, context: Optional[Dict] = None):
        """
        Log an error that occurred during evolution.

        Args:
            error_type: Type of error (e.g., "crossover_failed", "mutation_failed")
            error_message: Error message
            context: Additional context information
        """
        data = {
            "sample_id": self.sample_id,
            "phase": "error",
            "timestamp": datetime.now().isoformat(),
            "error_type": error_type,
            "error_message": error_message,
            "context": context or {}
        }

        self._save_json(f"error_{error_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", data)
        logger.error(f"Logged error: {error_type} - {error_message}")

    def _traj_to_dict(self, traj: Trajectory, idx: int, full: bool = False) -> Dict:
        """
        Convert a Trajectory to a dictionary for logging.

        Args:
            traj: Trajectory to convert
            idx: Index in the population
            full: Whether to include full reasoning text

        Returns:
            Dictionary representation of the trajectory
        """
        return {
            "id": f"traj_{idx}",
            "trajectory_id": traj.id,
            "model": traj.source_model,
            "method": traj.generation_method,
            "reasoning": traj.reasoning,  # Always save full reasoning, no truncation
            "reasoning_length": len(self._extract_pure_reasoning(traj.reasoning)) if traj.reasoning else 0,
            "answer": traj.answer,
            "fitness": traj.fitness_score,
            "exact_match": traj.exact_match if traj.exact_match is not None else False,
            "has_knowledge": traj.knowledge is not None and len(traj.knowledge) > 0,
            "knowledge_length": len(traj.knowledge) if traj.knowledge else 0,
            "metadata": traj.metadata if full else {}
        }

    def _eval_to_dict(self, traj: Trajectory, idx: int) -> Dict:
        """
        Convert evaluation results to a dictionary.

        Args:
            traj: Trajectory to convert
            idx: Index in the population

        Returns:
            Dictionary with evaluation metrics
        """
        return {
            "id": f"traj_{idx}",
            "trajectory_id": traj.id,
            "fitness": traj.fitness_score,
            "exact_match": traj.exact_match if traj.exact_match is not None else False,
            "length_score": traj.length_score if traj.length_score is not None else 0.0,
            "knowledge_score": traj.knowledge_score if traj.knowledge_score is not None else 0,
            "reasoning_length": len(traj.reasoning) if traj.reasoning else 0
        }

    def _compute_statistics(self, trajectories: List[Trajectory]) -> Dict:
        """
        Compute statistics for a population.

        Args:
            trajectories: List of trajectories

        Returns:
            Dictionary with population statistics
        """
        valid_trajectories = [t for t in trajectories if t.fitness_score is not None]

        if not valid_trajectories:
            return {
                "population_size": len(trajectories),
                "avg_fitness": 0.0,
                "best_fitness": 0.0,
                "worst_fitness": 0.0,
                "num_exact_matches": 0,
                "exact_match_rate": 0.0
            }

        fitnesses = [t.fitness_score for t in valid_trajectories]
        exact_matches = [t for t in valid_trajectories if t.exact_match is True]

        return {
            "population_size": len(trajectories),
            "avg_fitness": sum(fitnesses) / len(fitnesses),
            "best_fitness": max(fitnesses),
            "worst_fitness": min(fitnesses),
            "num_exact_matches": len(exact_matches),
            "exact_match_rate": len(exact_matches) / len(valid_trajectories),
            "avg_reasoning_length": sum(len(t.reasoning) if t.reasoning else 0 for t in valid_trajectories) / len(valid_trajectories)
        }

    def _extract_pure_reasoning(self, reasoning: str) -> str:
        """
        Extract pure reasoning content, excluding <|think|> markers.

        This ensures consistent reasoning_length calculation across all log files:
        - Excludes <|think|> and <|answer|> markers themselves
        - Only counts pure reasoning content between markers
        - Does not include knowledge augmentation content

        Args:
            reasoning: Full reasoning string potentially containing markers

        Returns:
            Pure reasoning content without markers
        """
        if not reasoning:
            return ""

        # Extract content between <|think|> and <|answer|>
        if "<|think|>" in reasoning:
            start = reasoning.find("<|think|>") + 9
            end = reasoning.find("<|answer|>")
            if end != -1:
                reasoning = reasoning[start:end].strip()

        return reasoning

    def _truncate_text(self, text: str, max_length: int = 500) -> str:
        """
        Truncate text to a maximum length.

        NOTE: Currently not used to preserve full reasoning in logs.
        Kept for potential future use if needed.

        Args:
            text: Text to truncate
            max_length: Maximum length

        Returns:
            Truncated text with indicator
        """
        if not text:
            return ""

        if len(text) <= max_length:
            return text

        return text[:max_length] + "... [truncated]"

    def _save_json(self, filename: str, data: Dict):
        """
        Save data to a JSON file.

        Args:
            filename: Name of the file
            data: Data to save
        """
        filepath = self.output_dir / filename

        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to save {filename}: {e}")
