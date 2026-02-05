"""
Reflective crossover operation for CoT-Evo framework.

This module implements the reflective crossover strategy from Section 2.4 of the paper.
Crossover is only triggered when s_EM(t_o) = 0 (i.e., the answer is incorrect).
"""

import logging
import re
from typing import Optional

from ..models.registry import ModelRegistry
from ..core.trajectory import Trajectory
from ..utils.answer_extractor import (
    extract_cot_from_markers,
    extract_cot_and_answer,
    clean_answer
)
from .prompts import (
    CROSSOVER_TEMPLATE,
    PREFIX_IDENTIFICATION_PROMPT,
    BREAKPOINT_EXTRACTION_PROMPT,
    REASONING_MODEL_GUIDANCE,
)


logger = logging.getLogger(__name__)


class ReflectiveCrossover:
    """
    Reflective crossover operator for CoT evolution.

    Implements the two-step crossover process from Section 2.4:
    1. Identify binding point B (last reasonable thought in provider)
    2. Cross-chain recombination: t' = t_o[:B] + C_r(t_o[:B], I)

    The crossover is only applied to trajectories with incorrect answers (s_EM = 0).
    """

    def __init__(self, model_registry: ModelRegistry):
        """
        Initialize the reflective crossover operator.

        Args:
            model_registry: Model registry for getting operator models
        """
        self.registry = model_registry

    async def crossover(
        self,
        target: Trajectory,
        provider: Trajectory
    ) -> Optional[Trajectory]:
        """
        Perform reflective crossover between target and provider trajectories.

        Args:
            target: Target trajectory (t_o) to improve
            provider: Provider trajectory (t_p) with useful strategies

        Returns:
            New trajectory after crossover, or None if crossover should not be applied
        """
        # Crossover is only triggered if answer is incorrect
        if target.exact_match is True:
            logger.debug(f"Target {target.id} already has correct answer, skipping crossover")
            return target

        logger.debug(f"Performing crossover: target={target.id}, provider={provider.id}")

        # Get operator model (auto or global mode)
        recombiner = self.registry.get_operator_model(target)

        try:
            # Step 1: Identify binding point B in provider (for extracting useful info)
            provider_binding_point = await self._identify_binding_point(provider, recombiner)

            if provider_binding_point is None or provider_binding_point <= 0:
                logger.warning(f"Invalid binding point ({provider_binding_point}) for provider {provider.id}")
                return None

            # Step 2: Identify binding point in TARGET (for where to insert new info)
            target_binding_point = await self._identify_binding_point(target, recombiner)

            if target_binding_point is None or target_binding_point <= 0:
                logger.warning(f"Invalid binding point ({target_binding_point}) for target {target.id}")
                return None

            # Step 3: Extract unique information from provider
            unique_info = await self._extract_unique_info(
                target, provider, recombiner
            )

            if not unique_info:
                logger.warning(f"No unique info extracted from provider {provider.id}")
                return None

            # Step 4: Generate recombined CoT using TARGET's binding point
            prefix = target.reasoning[:target_binding_point]
            new_reasoning = await self._generate_recombined(
                prefix, unique_info, recombiner, provider
            )

            # Extract answer from new reasoning
            answer = self._extract_answer(new_reasoning)

            # Create new trajectory
            new_trajectory = Trajectory(
                query=target.query,
                answer=answer,
                reasoning=new_reasoning,
                source_model=recombiner.model_name,
                generation_method="crossover",
                metadata={
                    "parents": [target.id, provider.id],
                    "target_binding_point": target_binding_point,
                    "provider_binding_point": provider_binding_point
                }
            )

            logger.info(f"Successfully created crossover trajectory {new_trajectory.id}")
            return new_trajectory

        except Exception as e:
            logger.error(f"Crossover failed: {e}")
            return None

    async def _identify_binding_point(
        self,
        provider: Trajectory,
        model
    ) -> Optional[int]:
        """
        Identify the binding point B in provider's reasoning.

        The binding point is the end of the last reasonable thought before errors occur.

        Args:
            provider: Provider trajectory
            model: Model to use for analysis

        Returns:
            Character position of binding point, or None if identification fails
        """
        # Use the prefix identification prompt
        prompt = PREFIX_IDENTIFICATION_PROMPT.format(
            query=provider.query,
            thought_current=provider.reasoning,
            answer=provider.answer
        )

        try:
            response = await model.generate_async(prompt)

            # Extract the sentence to be deleted
            match = re.search(r'\[RESULT_START\](.*?)\[RESULT_END\]', response, re.DOTALL)
            if match:
                sentence_to_delete = match.group(1).strip()

                # Find this sentence in the original reasoning
                # Return the position before this sentence
                idx = provider.reasoning.find(sentence_to_delete)
                if idx != -1:
                    return idx

            # Fallback: return 80% of the reasoning length
            return int(len(provider.reasoning) * 0.8)

        except Exception as e:
            logger.warning(f"Failed to identify binding point: {e}")
            # Fallback
            return int(len(provider.reasoning) * 0.8)

    async def _extract_unique_info(
        self,
        target: Trajectory,
        provider: Trajectory,
        model
    ) -> Optional[str]:
        """
        Extract unique/correct information from provider that's missing in target.

        Args:
            target: Target trajectory (current progress)
            provider: Provider trajectory (external exploration)
            model: Model to use for analysis

        Returns:
            Extracted unique information
        """
        # Use the breakpoint extraction prompt
        prompt = BREAKPOINT_EXTRACTION_PROMPT.format(
            query=target.query,
            thought_current=target.reasoning,
            thought_external=provider.reasoning,
            answer=target.answer
        )

        try:
            response = await model.generate_async(prompt)

            # Extract the result list
            match = re.search(r'\[RESULT_START\](.*?)\[RESULT_END\]', response, re.DOTALL)
            if match:
                result_text = match.group(1).strip()

                # Convert bullet points to a coherent text
                info_items = re.findall(r'\*\s*(.+)', result_text)
                if info_items:
                    unique_info = "\n".join(info_items)
                    return unique_info

            return None

        except Exception as e:
            logger.warning(f"Failed to extract unique info: {e}")
            return None

    async def _generate_recombined(
        self,
        prefix: str,
        unique_info: str,
        model,
        provider: Trajectory
    ) -> str:
        """
        Generate recombined reasoning chain.

        t' = t_o[:B] + C_r(t_o[:B], I)

        Args:
            prefix: Target reasoning prefix (t_o[:B])
            unique_info: Unique information from provider (I)
            model: Model to use for generation
            provider: Provider trajectory (for reference)

        Returns:
            Recombined reasoning chain
        """
        # Use the crossover template
        prompt = CROSSOVER_TEMPLATE.format(
            prefix=prefix,
            breakpoint=unique_info
        )

        # Check if model is a reasoning model that needs special guidance
        if self._is_reasoning_model(model):
            # Add special guidance for reasoning models
            prompt += "\n\n" + REASONING_MODEL_GUIDANCE

        try:
            new_reasoning = await model.generate_async(prompt)
            return new_reasoning

        except Exception as e:
            logger.error(f"Failed to generate recombined reasoning: {e}")
            raise

    def _is_reasoning_model(self, model) -> bool:
        """
        Check if the model is a reasoning model that needs special guidance.

        Args:
            model: Model to check

        Returns:
            True if the model is a reasoning model
        """
        # Check model name for known reasoning models
        reasoning_indicators = ["deepseek", "r1", "thinking", "qwen-235"]
        model_name_lower = model.model_name.lower()

        return any(indicator in model_name_lower for indicator in reasoning_indicators)

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
            r'\{\s*"Major Product"\s*:\s*"[^"]+"\s*\}',
            r'\{\s*"result"\s*:\s*"[^"]+"\s*\}',
            r'\{\s*"answer"\s*:\s*"[^"]+"\s*\}',
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
