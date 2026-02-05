"""
Fitness function module for CoT-Evo framework.

This module defines the fitness evaluator that combines multiple evaluation
metrics into a single fitness score according to formula 6 in the paper.
"""

from typing import Optional
from .trajectory import Trajectory
from ..utils.answer_extractor import clean_answer


class FitnessEvaluator:
    """
    Evaluates the fitness of trajectories using multiple metrics.

    Implements formula 6 from the paper:
    R(t) = s_EM + λ1 * s_LEN + λ2 * s_KNOW

    Attributes:
        lambda_length: Weight for length appropriateness score (λ1)
        lambda_knowledge: Weight for knowledge usage correctness score (λ2)
        use_exact_match: Whether to use exact match evaluation
        use_length_score: Whether to use length appropriateness evaluation
        use_knowledge_score: Whether to use knowledge usage evaluation
    """

    def __init__(
        self,
        lambda_length: float = 0.3,
        lambda_knowledge: float = 0.5,
        use_exact_match: bool = True,
        use_length_score: bool = True,
        use_knowledge_score: bool = True,
    ):
        """
        Initialize the fitness evaluator.

        Args:
            lambda_length: Weight for length score (default: 0.3)
            lambda_knowledge: Weight for knowledge score (default: 0.5)
            use_exact_match: Whether to include exact match in fitness
            use_length_score: Whether to include length score in fitness
            use_knowledge_score: Whether to include knowledge score in fitness
        """
        self.lambda_length = lambda_length
        self.lambda_knowledge = lambda_knowledge
        self.use_exact_match = use_exact_match
        self.use_length_score = use_length_score
        self.use_knowledge_score = use_knowledge_score

        # Individual evaluators (to be injected)
        self.exact_match_evaluator = None
        self.length_evaluator = None
        self.knowledge_evaluator = None

    def set_exact_match_evaluator(self, evaluator) -> None:
        """Set the exact match evaluator."""
        self.exact_match_evaluator = evaluator

    def set_length_evaluator(self, evaluator) -> None:
        """Set the length evaluator."""
        self.length_evaluator = evaluator

    def set_knowledge_evaluator(self, evaluator) -> None:
        """Set the knowledge evaluator."""
        self.knowledge_evaluator = evaluator

    async def evaluate(
        self,
        trajectory: Trajectory,
        ground_truth: str,
        reference_knowledge: Optional[str] = None,
    ) -> float:
        """
        Evaluate the fitness of a trajectory.

        Implements formula 6: R(t) = s_EM + λ1 * s_LEN + λ2 * s_KNOW

        Args:
            trajectory: Trajectory to evaluate
            ground_truth: Ground truth answer
            reference_knowledge: Reference knowledge for knowledge evaluation

        Returns:
            Combined fitness score
        """
        fitness = 0.0

        # 1. Exact match (formula 3)
        if self.use_exact_match and self.exact_match_evaluator:
            is_match = await self.exact_match_evaluator.match(trajectory.answer, ground_truth)
            s_em = 1.0 if is_match else 0.0
            trajectory.set_exact_match(is_match)
            fitness += s_em

        # 2. Length appropriateness (formula 4)
        if self.use_length_score and self.length_evaluator:
            s_len = self.length_evaluator.score(trajectory.reasoning)
            trajectory.set_length_score(s_len)
            fitness += self.lambda_length * s_len

        # 3. Knowledge usage correctness (formula 5)
        if self.use_knowledge_score and self.knowledge_evaluator and reference_knowledge:
            s_know_raw = await self.knowledge_evaluator.judge(
                trajectory.reasoning,
                reference_knowledge,
            )
            # Normalize from [1, 5] to [0, 1]
            s_know = s_know_raw / 5.0
            trajectory.set_knowledge_score(s_know_raw)
            fitness += self.lambda_knowledge * s_know

        # Cache the final fitness score
        trajectory.set_fitness_score(fitness)

        return fitness

    def compute_fitness_from_scores(
        self,
        exact_match: bool,
        length_score: float,
        knowledge_score: Optional[int] = None,
    ) -> float:
        """
        Compute fitness from pre-computed scores.

        Useful when scores are already computed and cached.

        Args:
            exact_match: Whether answer exactly matches ground truth
            length_score: Length appropriateness score (0.0, 0.5, or 1.0)
            knowledge_score: Knowledge usage correctness (1-5), optional

        Returns:
            Combined fitness score
        """
        fitness = 0.0

        if self.use_exact_match:
            fitness += 1.0 if exact_match else 0.0

        if self.use_length_score:
            fitness += self.lambda_length * length_score

        if self.use_knowledge_score and knowledge_score is not None:
            fitness += self.lambda_knowledge * (knowledge_score / 5.0)

        return fitness


class LengthEvaluator:
    """
    Evaluates the length appropriateness of reasoning.

    Implements formula 4 from the paper:
    - 0.0 if len < 15th percentile
    - 0.5 if len > 85th percentile
    - 1.0 otherwise
    """

    def __init__(self, lower_percentile: int, upper_percentile: int):
        """
        Initialize the length evaluator.

        Args:
            lower_percentile: Lower bound (15th percentile in tokens)
            upper_percentile: Upper bound (85th percentile in tokens)
        """
        self.lower = lower_percentile
        self.upper = upper_percentile

    def score(self, reasoning: str) -> float:
        """
        Compute length appropriateness score.

        Args:
            reasoning: Reasoning text to evaluate

        Returns:
            Score of 0.0, 0.5, or 1.0
        """
        # Estimate token count by word count (rough approximation)
        length = len(reasoning.split())

        if length < self.lower:
            return 0.0
        elif length > self.upper:
            return 0.5
        else:
            return 1.0


class ExactMatchEvaluator:
    """
    Evaluates exact match between predicted and ground truth answers.

    Implements formula 3 from the paper.
    """

    def __init__(self, strict: bool = True):
        """
        Initialize the exact match evaluator.

        Args:
            strict: If True, requires exact string match (after normalization).
                   If False, uses more lenient matching.
        """
        self.strict = strict

    async def match(self, predicted: str, ground_truth: str) -> bool:
        """
        Check if predicted answer matches ground truth.

        Args:
            predicted: Predicted answer
            ground_truth: Ground truth answer

        Returns:
            True if match, False otherwise
        """
        if self.strict:
            return self._exact_match_strict(predicted, ground_truth)
        else:
            return self._exact_match_lenient(predicted, ground_truth)

    def _exact_match_strict(self, predicted: str, ground_truth: str) -> bool:
        """Strict exact match after normalization (matching test_complete_evolution.py)."""
        # Use clean_answer from answer_extractor (matches test version)
        pred_normalized = clean_answer(predicted).strip().lower()
        gt_normalized = clean_answer(ground_truth).strip().lower()
        return pred_normalized == gt_normalized

    def _exact_match_lenient(self, predicted: str, ground_truth: str) -> bool:
        """Lenient matching that allows for minor variations."""
        # Use clean_answer from answer_extractor
        pred_normalized = clean_answer(predicted).strip().lower()
        gt_normalized = clean_answer(ground_truth).strip().lower()

        # Remove all whitespace and compare
        pred_no_space = pred_normalized.replace(" ", "").replace("\t", "").replace("\n", "")
        gt_no_space = gt_normalized.replace(" ", "").replace("\t", "").replace("\n", "")

        return pred_no_space == gt_no_space

    def _normalize_answer(self, answer: str) -> str:
        """Normalize answer for comparison."""
        # Strip whitespace and convert to lowercase
        normalized = answer.strip().lower()

        # Remove common prefixes/suffixes
        prefixes_to_remove = ["answer:", "the answer is", "result:"]
        for prefix in prefixes_to_remove:
            if normalized.startswith(prefix):
                normalized = normalized[len(prefix):].strip()

        return normalized


class KnowledgeJudgeEvaluator:
    """
    Evaluates knowledge usage correctness using LLM-as-a-Judge.

    Implements formula 5 from the paper: s_KNOW = Judge(K_x, t), s_KNOW ∈ {1,2,3,4,5}

    This evaluator uses an LLM to assess how accurately the trajectory uses
    the reference knowledge in its reasoning.
    """

    def __init__(self, judge_model):
        """
        Initialize the knowledge judge evaluator.

        Args:
            judge_model: LLM model to use for judging
        """
        self.judge_model = judge_model

    async def judge(self, reasoning: str, reference_knowledge: str) -> int:
        """
        Judge the knowledge usage correctness.

        Args:
            reasoning: Trajectory's reasoning process
            reference_knowledge: Reference knowledge snippet

        Returns:
            Score from 1 to 5
        """
        prompt = self._construct_judge_prompt(reasoning, reference_knowledge)

        response = await self.judge_model.generate_async(prompt)

        score = self._extract_score(response)

        return score

    def _construct_judge_prompt(self, reasoning: str, reference_knowledge: str) -> str:
        """Construct the prompt for knowledge judgment."""
        prompt = f"""You are evaluating how well a reasoning trajectory uses given knowledge.

Reference Knowledge:
{reference_knowledge}

Reasoning to Evaluate:
{reasoning}

Your task: Assess how accurately and appropriately the reasoning uses the reference knowledge.
Consider:
1. Does the reasoning correctly interpret the knowledge?
2. Does the reasoning apply the knowledge appropriately?
3. Is the knowledge used to reach the correct conclusion?

Provide a score from 1 to 5:
- 1: Knowledge is misinterpreted or misused
- 2: Knowledge is used but with significant errors
- 3: Knowledge is used adequately but with minor issues
- 4: Knowledge is used well and correctly
- 5: Knowledge is used excellently with deep understanding

Output only the score as a single digit (1-5)."""

        return prompt

    def _extract_score(self, response: str) -> int:
        """Extract the score from the judge's response."""
        import re

        # Look for the first number in the response
        match = re.search(r'\d+', response.strip())

        if match:
            score = int(match.group())
            # Clamp to valid range [1, 5]
            return max(1, min(5, score))
        else:
            # Default to middle score if parsing fails
            return 3
