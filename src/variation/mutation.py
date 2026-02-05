"""
Reflective mutation operations for CoT-Evo framework.

This module implements the three mutation modes from Formula 12-14 of the paper:
- Formula 12: Additive mutation (enrich with details)
- Formula 13: Deletive mutation (remove redundancy)
- Formula 14: Innovative mutation (diagnose errors, then delete)

IMPORTANT: This module properly separates reasoning (CoT) from final answers.
During mutation, only the reasoning part is modified, while the final answer
is preserved and re-attached at the end.
"""

import logging
import re
from typing import Optional

from ..models.registry import ModelRegistry
from ..core.trajectory import Trajectory
from ..utils.answer_extractor import (
    extract_cot_and_answer,
    extract_cot_from_markers,
    combine_cot_and_answer,
    clean_answer
)
from .prompts import (
    MUTATION_ADD_PROMPT,
    MUTATION_DELETE_PROMPT,
    MUTATION_INNOVATE_DIAGNOSE_PROMPT,
)


logger = logging.getLogger(__name__)


class ReflectiveMutation:
    """
    Reflective mutation operator for CoT evolution.

    Implements three mutation modes from Formula 12-14:
    1. Additive mutation: Enrich logical detail and explanations
    2. Deletive mutation: Prune redundancy and unproductive exploration
    3. Innovative mutation: Diagnose errors and generate new approach
    """

    # Mutation mode constants
    MODE_ADD = "add"
    MODE_DELETE = "delete"
    MODE_INNOVATE = "innovate"

    def __init__(self, model_registry: ModelRegistry):
        """
        Initialize the reflective mutation operator.

        Args:
            model_registry: Model registry for getting operator models
        """
        self.registry = model_registry

    async def mutate(
        self,
        trajectory: Trajectory,
        mode: str = "add",
        query: Optional[str] = None,
        ground_truth: Optional[str] = None
    ) -> Optional[Trajectory]:
        """
        Perform mutation on a trajectory.

        Args:
            trajectory: Trajectory to mutate
            mode: Mutation mode ("add", "delete", or "innovate")
            query: The original query (required for innovate mode)
            ground_truth: The ground truth answer (required for innovate mode)

        Returns:
            New mutated trajectory, or None if mutation fails
        """
        logger.debug(f"Performing {mode}-mutation on trajectory {trajectory.id}")

        # Validate innovate mode requirements
        if mode == self.MODE_INNOVATE:
            if not query or not ground_truth:
                logger.warning("Innovate mutation requires both query and ground_truth")
                return None

        # Get operator model (auto or global mode)
        mutator = self.registry.get_operator_model(trajectory)

        try:
            if mode == self.MODE_ADD:
                return await self._additive_mutation(trajectory, mutator)
            elif mode == self.MODE_DELETE:
                return await self._deletive_mutation(trajectory, mutator)
            elif mode == self.MODE_INNOVATE:
                return await self._innovative_mutation(trajectory, mutator, query, ground_truth)
            else:
                logger.error(f"Unknown mutation mode: {mode}")
                return None

        except Exception as e:
            logger.error(f"Mutation failed (mode={mode}): {e}")
            return None

    async def _additive_mutation(
        self,
        trajectory: Trajectory,
        model
    ) -> Optional[Trajectory]:
        """
        Additive mutation (Formula 12): t'_(a) = M_u(t_o, Add)

        Enrich the reasoning with more details, explanations, and domain knowledge.

        Args:
            trajectory: Trajectory to mutate
            model: Model to use for mutation

        Returns:
            Mutated trajectory with added details
        """
        # Use the add mutation prompt
        prompt = MUTATION_ADD_PROMPT.format(
            query=trajectory.query,
            thought_current=trajectory.reasoning
        )

        # Generate mutated reasoning
        new_reasoning = await model.generate_async(prompt)

        # Extract the result
        new_reasoning = self._extract_result(new_reasoning)

        # Extract answer
        answer = self._extract_answer(new_reasoning)

        new_trajectory = Trajectory(
            query=trajectory.query,
            answer=answer,
            reasoning=new_reasoning,
            source_model=model.model_name,
            generation_method="mutation_add",
            metadata={"parent": trajectory.id}
        )

        logger.info(f"Successfully created additive mutation {new_trajectory.id}")
        return new_trajectory

    async def _deletive_mutation(
        self,
        trajectory: Trajectory,
        model
    ) -> Optional[Trajectory]:
        """
        Deletive mutation (Formula 13): t'_(d) = M_u(t_o, Delete)

        Prune redundancy, unproductive exploration, and extraneous knowledge.

        Args:
            trajectory: Trajectory to mutate
            model: Model to use for mutation

        Returns:
            Mutated trajectory with removed redundancy
        """
        # Use the delete mutation prompt
        prompt = MUTATION_DELETE_PROMPT.format(
            query=trajectory.query,
            thought_current=trajectory.reasoning
        )

        # Generate mutated reasoning
        new_reasoning = await model.generate_async(prompt)

        # Extract the result
        new_reasoning = self._extract_result(new_reasoning)

        # Extract answer
        answer = self._extract_answer(new_reasoning)

        new_trajectory = Trajectory(
            query=trajectory.query,
            answer=answer,
            reasoning=new_reasoning,
            source_model=model.model_name,
            generation_method="mutation_delete",
            metadata={"parent": trajectory.id}
        )

        logger.info(f"Successfully created deletive mutation {new_trajectory.id}")
        return new_trajectory

    async def _innovative_mutation(
        self,
        trajectory: Trajectory,
        model,
        query: str,
        ground_truth: str
    ) -> Optional[Trajectory]:
        """
        Innovative mutation (Formula 14): t'_(c) = M_u(M_u(t_o, Innovate), Delete)

        Two-stage mutation:
        1. Diagnose erroneous logic using correct answer (ground truth)
        2. Generate new trajectory that avoids the mistakes
        3. Then prune via deletive mutation for compactness

        Args:
            trajectory: Trajectory to mutate
            model: Model to use for mutation
            query: The original query/problem
            ground_truth: The ground truth answer

        Returns:
            Mutated trajectory with corrected logic
        """
        # Stage 1: Diagnose errors using the ground truth
        diagnose_prompt = MUTATION_INNOVATE_DIAGNOSE_PROMPT.format(
            query=query,
            thought_current=trajectory.reasoning,
            answer=trajectory.answer,  # Changed from current_answer to answer
            ground_truth=ground_truth
        )

        try:
            # Get diagnosis/advice
            diagnosis = await model.generate_async(diagnose_prompt)

            # Extract advice list
            advice = self._extract_diagnosis(diagnosis)

            # Stage 2: Generate new reasoning with the advice in mind
            # Build prompt in <|think|>/<|answer|> format
            prompt = f"""Given the problem and the following advice for improving the reasoning, generate a new solution approach.

[Problem]
{query}

[Ground Truth Answer]
{ground_truth}

[Current Incorrect Reasoning]
{trajectory.reasoning}

[Advice for Improvement]
{advice}

Generate your corrected reasoning in this format:
<|think|>
Your step-by-step reasoning that incorporates the advice and avoids previous errors
<|answer|>
{{
    "Major Product": "your answer in SMILES format"
}}
"""

            # Generate innovative reasoning
            innovative_output = await model.generate_async(prompt)

            # Extract reasoning and answer using new format
            try:
                innovative_reasoning, innovative_answer = extract_cot_from_markers(innovative_output)
            except ValueError:
                # Fallback to old extraction method
                innovative_reasoning, innovative_answer = extract_cot_and_answer(innovative_output)

            # Stage 3: Apply deletive mutation to prune redundancy
            delete_prompt = MUTATION_DELETE_PROMPT.format(
                query=query,
                thought_current=innovative_reasoning
            )

            final_output = await model.generate_async(delete_prompt)

            # Extract final reasoning and answer
            try:
                final_reasoning, final_answer = extract_cot_from_markers(final_output)
            except ValueError:
                # Fallback
                final_reasoning, final_answer = extract_cot_and_answer(final_output)

            # Ensure answer is preserved (use innovative_answer if it's better)
            if innovative_answer and len(innovative_answer) > len(final_answer):
                final_answer = innovative_answer

            new_trajectory = Trajectory(
                query=trajectory.query,
                answer=final_answer,
                reasoning=final_reasoning,
                source_model=model.model_name,
                generation_method="mutation_innovate",
                metadata={
                    "parent": trajectory.id,
                    "diagnosis": advice[:200] if advice else ""  # Truncate for metadata
                }
            )

            logger.info(f"Successfully created innovative mutation {new_trajectory.id}")
            return new_trajectory

        except Exception as e:
            logger.error(f"Innovative mutation failed: {e}")
            # Fallback to simple deletive mutation
            return await self._deletive_mutation(trajectory, model)

    def _extract_result(self, response: str) -> str:
        """
        Extract the result from mutation response.

        Removes [RESULT_START] and [RESULT_END] markers.

        Args:
            response: Full model response

        Returns:
            Extracted result content
        """
        # Look for result markers
        start_marker = "[RESULT_START]"
        end_marker = "[RESULT_END]"

        start_idx = response.find(start_marker)
        end_idx = response.find(end_marker)

        if start_idx != -1 and end_idx != -1:
            # Extract content between markers
            result = response[start_idx + len(start_marker):end_idx].strip()
            return result
        else:
            # Fallback: return entire response
            return response.strip()

    def _extract_diagnosis(self, response: str) -> str:
        """
        Extract diagnosis/advice from innovate mutation response.

        Args:
            response: Full model response

        Returns:
            Extracted advice as a string
        """
        # Look for result markers
        start_marker = "[RESULT_START]"
        end_marker = "[RESULT_END]"

        start_idx = response.find(start_marker)
        end_idx = response.find(end_marker)

        if start_idx != -1 and end_idx != -1:
            # Extract content between markers
            diagnosis = response[start_idx + len(start_marker):end_idx].strip()

            # Convert bullet list to coherent text
            advice_items = re.findall(r'\*\s*(.+)', diagnosis)
            if advice_items:
                return "\n".join(advice_items)

            return diagnosis

        else:
            # Fallback: return entire response
            return response.strip()

    def _extract_answer(self, text: str) -> str:
        """
        Extract the final answer from reasoning text.

        Tries <|think|>...<|answer|>... format first, then falls back to other methods.

        Args:
            text: Full reasoning text with possible markers

        Returns:
            Extracted answer
        """
        # Priority 1: Try <|think|>...<|answer|>... format
        try:
            reasoning, answer = extract_cot_from_markers(text)
            return answer
        except ValueError:
            pass  # Markers not found, try other methods

        # Priority 2: Try to extract JSON
        # Look for {"Major Product": "..."} or similar
        json_patterns = [
            r'\{\s*"Major Product"\s*:\s*"[^"]+)"\s*\}',
            r'\{\s*"result"\s*:\s*"[^"]+)"\s*\}',
            r'\{\s*"answer"\s*:\s*"[^"]+)"\s*\}',
        ]

        for pattern in json_patterns:
            match = re.search(pattern, text)
            if match:
                json_str = match.group(0)
                try:
                    import json
                    data = json.loads(json_str)
                    # Return the value
                    for key in ["Major Product", "result", "answer"]:
                        if key in data:
                            return json.dumps({key: data[key]})
                except json.JSONDecodeError:
                    continue

        # Priority 3: Try to find the last line as the answer
        lines = text.strip().split('\n')
        lines = [l.strip() for l in lines if l.strip()]

        if not lines:
            return ""

        # Last non-empty line is typically the answer
        return lines[-1]
