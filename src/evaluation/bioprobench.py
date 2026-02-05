"""
BioProBench evaluation framework.

This module provides evaluation functions for biological protocol reasoning tasks,
including error recognition, protocol Q&A, step ordering, and protocol generation.
"""

import logging
import re
import ast
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Try to import NLTK for BLEU score calculation
try:
    import nltk
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    logger.warning("NLTK not available. BLEU score calculation will be limited.")


@dataclass
class BioProBenchMetrics:
    """Metrics for BioProBench evaluation."""
    task_type: str  # 'ERR', 'PQA', 'ORD', 'GEN'
    accuracy: float = 0.0
    exact_match: bool = False
    confidence: Optional[float] = None
    bleu_score: Optional[float] = None
    ordering_correct: bool = False
    error_message: Optional[str] = None
    additional_metrics: Dict[str, Any] = None

    def __post_init__(self):
        if self.additional_metrics is None:
            self.additional_metrics = {}


class BioProBenchEvaluator:
    """
    Evaluator for BioProBench (Biology Protocol Benchmark) tasks.

    Supports four task types:
    - ERR: Error Recognition - Identify if a protocol has errors
    - PQA: Protocol Question Answering - Answer questions about protocols
    - ORD: Ordering - Order protocol steps correctly
    - GEN: Generation - Generate protocol text

    Attributes:
        strict_mode: If True, requires exact format compliance
    """

    def __init__(self, strict_mode: bool = False):
        """
        Initialize the BioProBench evaluator.

        Args:
            strict_mode: Whether to require strict format compliance
        """
        self.strict_mode = strict_mode

    def extract_answer(self, text: str, dataset_name: str = "BioProBench") -> str:
        """
        Extract answer from LLM output.

        Handles common answer formats:
        - [ANSWER_START]answer[ANSWER_END]
        - Answer: answer
        - Final answer: answer
        - Last line

        Args:
            text: LLM output text
            dataset_name: Name of dataset (for format detection)

        Returns:
            Extracted answer string
        """
        # Try special marker format
        marker_pattern = r'\[ANSWER_START\](.*?)\[ANSWER_END\]'
        match = re.search(marker_pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip()

        # Try "Answer:" or "Final answer:" format
        answer_patterns = [
            r'(?:Final )?[Aa]swer:\s*(.*?)(?:\n|$)',
            r'[Aa]nswer\s+(?:is\s+)?:\s*(.*?)(?:\n|$)',
        ]

        for pattern in answer_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()

        # Fall back to last non-empty line
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        if lines:
            return lines[-1]

        return text.strip()

    def evaluate_err(
        self,
        prediction: str,
        ground_truth: bool
    ) -> BioProBenchMetrics:
        """
        Evaluate Error Recognition (ERR) task.

        Args:
            prediction: Predicted error status (True/False or text)
            ground_truth: Ground truth error status

        Returns:
            BioProBenchMetrics with evaluation results
        """
        metrics = BioProBenchMetrics(task_type="ERR")

        # Extract answer from prediction
        answer = self.extract_answer(prediction)

        # Convert to boolean
        if isinstance(answer, str):
            answer_lower = answer.lower()
            # Check for explicit true/false
            if 'true' in answer_lower:
                predicted_bool = True
            elif 'false' in answer_lower:
                predicted_bool = False
            else:
                # Try to infer from context
                predicted_bool = len(answer_lower) > 0 and answer_lower not in ['no', 'none', '0']
        else:
            predicted_bool = bool(answer)

        # Calculate accuracy
        metrics.exact_match = predicted_bool == ground_truth
        metrics.accuracy = 1.0 if metrics.exact_match else 0.0

        return metrics

    def evaluate_pqa(
        self,
        prediction: str,
        ground_truth: str,
        extract_confidence: bool = False
    ) -> BioProBenchMetrics:
        """
        Evaluate Protocol Question Answering (PQA) task.

        Args:
            prediction: Predicted answer (may include confidence)
            ground_truth: Ground truth answer
            extract_confidence: Whether to extract confidence score

        Returns:
            BioProBenchMetrics with evaluation results
        """
        metrics = BioProBenchMetrics(task_type="PQA")

        # Extract answer
        answer = self.extract_answer(prediction)

        # Check for confidence score (format: "answer & confidence" or "answer confidence")
        confidence = None
        if extract_confidence:
            # Try to extract confidence
            conf_patterns = [
                r'(.*?)\s*&\s*(\d+)',  # "answer & 80"
                r'(.*?)\s+(\d+)$',  # "answer 80"
            ]

            for pattern in conf_patterns:
                match = re.match(pattern, answer.strip())
                if match:
                    answer = match.group(1).strip()
                    try:
                        confidence = int(match.group(2)) / 100.0
                    except (ValueError, IndexError):
                        pass
                    break

        metrics.confidence = confidence

        # Compare with ground truth
        gt_lower = ground_truth.lower().strip()
        pred_lower = answer.lower().strip()

        # Exact match
        if pred_lower == gt_lower:
            metrics.exact_match = True
            metrics.accuracy = 1.0
        # Contains match (ground truth is contained in prediction)
        elif gt_lower in pred_lower:
            metrics.exact_match = True
            metrics.accuracy = 0.9
        # Partial match (words overlap)
        else:
            gt_words = set(gt_lower.split())
            pred_words = set(pred_lower.split())
            if gt_words and pred_words:
                overlap = len(gt_words & pred_words) / len(gt_words | pred_words)
                metrics.accuracy = overlap
                metrics.exact_match = overlap > 0.8

        return metrics

    def evaluate_ord(
        self,
        prediction: str,
        correct_steps: List[Any],
        wrong_steps: Optional[List[Any]] = None
    ) -> BioProBenchMetrics:
        """
        Evaluate Ordering (ORD) task.

        Args:
            prediction: Predicted ordering (indices or step numbers)
            correct_steps: Ground truth correct ordering
            wrong_steps: Ground truth wrong steps (for remapping)

        Returns:
            BioProBenchMetrics with evaluation results
        """
        metrics = BioProBenchMetrics(task_type="ORD")

        if wrong_steps is None:
            wrong_steps = []

        # Extract answer
        answer = self.extract_answer(prediction)

        # Parse prediction
        try:
            # Try to evaluate as Python literal (list, tuple, etc.)
            predicted_indices = ast.literal_eval(answer)

            # Ensure it's a list
            if isinstance(predicted_indices, tuple):
                predicted_indices = list(predicted_indices)
            elif not isinstance(predicted_indices, list):
                predicted_indices = [predicted_indices]

            # Remap indices if wrong_steps provided
            if wrong_steps and predicted_indices:
                try:
                    # Check if indices need remapping
                    if set(predicted_indices) == set(range(len(correct_steps))):
                        # Remap using wrong_steps
                        predicted_indices = [wrong_steps[i] for i in predicted_indices]
                except (IndexError, TypeError):
                    metrics.error_message = "Failed to remap indices"
                    return metrics

            # Compare with ground truth
            if predicted_indices == correct_steps:
                metrics.exact_match = True
                metrics.accuracy = 1.0
                metrics.ordering_correct = True
            else:
                # Calculate partial accuracy
                if len(correct_steps) > 0:
                    correct_positions = sum(
                        1 for i, step in enumerate(predicted_indices)
                        if i < len(correct_steps) and step == correct_steps[i]
                    )
                    metrics.accuracy = correct_positions / len(correct_steps)
                else:
                    metrics.accuracy = 0.0

        except (ValueError, SyntaxError) as e:
            metrics.error_message = f"Failed to parse ordering: {str(e)}"
            metrics.accuracy = 0.0

        return metrics

    def evaluate_gen(
        self,
        prediction: str,
        ground_truth: str
    ) -> BioProBenchMetrics:
        """
        Evaluate Generation (GEN) task using BLEU score.

        Args:
            prediction: Generated protocol text
            ground_truth: Ground truth protocol text

        Returns:
            BioProBenchMetrics with evaluation results
        """
        metrics = BioProBenchMetrics(task_type="GEN")

        if not NLTK_AVAILABLE:
            # Fall back to simple text similarity
            pred_lower = prediction.lower().strip()
            gt_lower = ground_truth.lower().strip()

            if pred_lower == gt_lower:
                metrics.bleu_score = 1.0
                metrics.accuracy = 1.0
            else:
                # Simple word overlap
                pred_words = set(pred_lower.split())
                gt_words = set(gt_lower.split())
                if pred_words and gt_words:
                    overlap = len(pred_words & gt_words) / len(pred_words | gt_words)
                    metrics.bleu_score = overlap
                    metrics.accuracy = overlap
                else:
                    metrics.bleu_score = 0.0
                    metrics.accuracy = 0.0

            return metrics

        try:
            # Tokenize
            gt_tokens = nltk.word_tokenize(gt_lower := ground_truth.lower())
            pred_tokens = nltk.word_tokenize(prediction.lower())

            # Calculate BLEU score
            smoothing = SmoothingFunction().method1
            bleu = sentence_bleu(
                [gt_tokens],
                pred_tokens,
                weights=(0.5, 0.5),
                smoothing_function=smoothing
            )

            metrics.bleu_score = bleu
            metrics.accuracy = bleu

            # Also check exact match
            if prediction.strip() == ground_truth.strip():
                metrics.exact_match = True

        except Exception as e:
            metrics.error_message = f"Failed to calculate BLEU: {str(e)}"
            metrics.bleu_score = 0.0
            metrics.accuracy = 0.0

        return metrics

    def evaluate(
        self,
        prediction: str,
        task_type: str,
        ground_truth: Any,
        **kwargs
    ) -> BioProBenchMetrics:
        """
        Evaluate a BioProBench task (auto-detects task type).

        Args:
            prediction: Model prediction
            task_type: Type of task ('ERR', 'PQA', 'ORD', 'GEN')
            ground_truth: Ground truth (format depends on task_type)
            **kwargs: Additional task-specific arguments

        Returns:
            BioProBenchMetrics with evaluation results
        """
        task_type = task_type.upper()

        if task_type == "ERR":
            return self.evaluate_err(prediction, ground_truth)
        elif task_type == "PQA":
            return self.evaluate_pqa(prediction, ground_truth, **kwargs)
        elif task_type == "ORD":
            return self.evaluate_ord(prediction, ground_truth.get('correct_steps', []), ground_truth.get('wrong_steps'))
        elif task_type == "GEN":
            return self.evaluate_gen(prediction, ground_truth)
        else:
            metrics = BioProBenchMetrics(task_type=task_type)
            metrics.error_message = f"Unknown task type: {task_type}"
            return metrics

    def evaluate_batch(
        self,
        predictions: List[str],
        task_types: List[str],
        ground_truths: List[Any]
    ) -> List[BioProBenchMetrics]:
        """
        Evaluate multiple BioProBench tasks.

        Args:
            predictions: List of predictions
            task_types: List of task types
            ground_truths: List of ground truths

        Returns:
            List of BioProBenchMetrics
        """
        if not (len(predictions) == len(task_types) == len(ground_truths)):
            raise ValueError("predictions, task_types, and ground_truths must have the same length")

        results = []
        for pred, task_type, gt in zip(predictions, task_types, ground_truths):
            metrics = self.evaluate(pred, task_type, gt)
            results.append(metrics)

        return results


def create_bioprobench_evaluator(strict_mode: bool = False) -> BioProBenchEvaluator:
    """
    Convenience function to create a BioProBench evaluator.

    Args:
        strict_mode: Whether to require strict format compliance

    Returns:
        Configured BioProBenchEvaluator instance
    """
    return BioProBenchEvaluator(strict_mode=strict_mode)


def compute_classification_metrics(
    predictions: List[Any],
    ground_truths: List[Any]
) -> Dict[str, float]:
    """
    Compute classification metrics (for ERR tasks).

    Args:
        predictions: List of predicted labels
        ground_truths: List of ground truth labels

    Returns:
        Dictionary with accuracy, precision, recall, F1
    """
    if not predictions or not ground_truths:
        return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}

    # Calculate accuracy
    correct = sum(1 for p, g in zip(predictions, ground_truths) if p == g)
    accuracy = correct / len(predictions)

    # For binary classification
    unique_labels = set(ground_truths)
    if len(unique_labels) == 2:
        # Calculate precision, recall, F1
        tp = sum(1 for p, g in zip(predictions, ground_truths) if p == True and g == True)
        fp = sum(1 for p, g in zip(predictions, ground_truths) if p == True and g == False)
        fn = sum(1 for p, g in zip(predictions, ground_truths) if p == False and g == True)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }
    else:
        return {"accuracy": accuracy}


def calculate_kendall_tau(
    ground_truths: List[List[Any]],
    predictions: List[List[Any]]
) -> float:
    """
    Calculate Kendall's Tau for ranking (ORD tasks).

    Args:
        ground_truths: List of ground truth orderings
        predictions: List of predicted orderings

    Returns:
        Kendall's Tau score
    """
    try:
        from scipy.stats import kendalltau
    except ImportError:
        logger.warning("SciPy not available for Kendall's Tau calculation")
        return 0.0

    taus = []
    for gt, pred in zip(ground_truths, predictions):
        if not gt or not pred:
            continue

        try:
            # Calculate rank correlation
            tau, _ = kendalltau(gt, pred)
            taus.append(tau)
        except Exception as e:
            logger.error(f"Error calculating Kendall's Tau: {e}")
            continue

    return sum(taus) / len(taus) if taus else 0.0
