"""
Multi-thinker CoT generator for CoT-Evo framework.

This module implements the multi-thinker initialization strategy (Formula 1-2 from the paper).
"""

import asyncio
import logging
from typing import List, Optional
import re

from ..models.registry import ModelRegistry
from ..core.trajectory import Trajectory
from ..core.population import Population
from ..knowledge.hybrid import HybridKnowledgeAugmenter
from ..utils.answer_extractor import extract_cot_and_answer
from .prompts import (
    SYSTEM_PROMPTS,
    VANILLA_TEMPLATE,
    KNOWLEDGE_AUGMENTED_TEMPLATE,
    STOP_SEQUENCES,
)


logger = logging.getLogger(__name__)


class MultiThinkerGenerator:
    """
    Multi-thinker CoT generator implementing Formula 1-2 from the paper.

    Formula 1 (vanilla): t_i = l_i(x)
    Formula 2 (knowledge-augmented): t_j = l_j(x, K_x)

    This generator creates an initial diverse population by:
    1. Using multiple teacher models with different prompting strategies
    2. Augmenting some trajectories with domain knowledge
    """

    def __init__(
        self,
        model_registry: ModelRegistry,
        knowledge_augmenter: HybridKnowledgeAugmenter,
        dataset_name: str = "generic"
    ):
        """
        Initialize the multi-thinker generator.

        Args:
            model_registry: Registry of available teacher models
            knowledge_augmenter: Knowledge augmenter for Formula 2
            dataset_name: Name of the dataset (for selecting appropriate prompts)
        """
        self.registry = model_registry
        self.knowledge_augmenter = knowledge_augmenter
        self.dataset_name = dataset_name

        # Get system prompt for this dataset
        self.system_prompt = SYSTEM_PROMPTS.get(
            dataset_name,
            SYSTEM_PROMPTS["SciKnowEval"]  # Fallback to generic science prompt
        )

        # Get stop sequences for this dataset
        self.stop_sequences = STOP_SEQUENCES.get(dataset_name, [])

    async def generate_initial_pool(
        self,
        query: str,
        ground_truth: str,
        n_vanilla: int = 7,
        n_knowledge_augmented: int = 3,
        prompt_template: Optional[str] = None
    ) -> Population:
        """
        Generate initial candidate pool P = P^G ∪ P^K (Formula 1-2).

        Args:
            query: The input problem/question
            ground_truth: The correct answer (used for knowledge generation)
            n_vanilla: Number of vanilla-generated trajectories (Formula 1)
            n_knowledge_augmented: Number of knowledge-augmented trajectories (Formula 2)
            prompt_template: Optional custom prompt template

        Returns:
            Population containing all generated trajectories
        """
        logger.info(
            f"Generating initial pool: {n_vanilla} vanilla + "
            f"{n_knowledge_augmented} knowledge-augmented"
        )

        # Collect all generation tasks
        tasks = []

        # Formula 1: Generate vanilla trajectories
        for i in range(n_vanilla):
            model = self.registry.get_random_thinker()
            if model is None:
                logger.warning("No teacher models available for vanilla generation")
                continue
            tasks.append(self._generate_vanilla(model, query, prompt_template))

        # Formula 2: Generate knowledge-augmented trajectories
        # First generate knowledge once
        if n_knowledge_augmented > 0:
            try:
                knowledge = await self.knowledge_augmenter.generate_knowledge(query, ground_truth)
                logger.debug(f"Generated knowledge: {knowledge[:100]}...")

                # Use the same knowledge for all augmented trajectories
                for i in range(n_knowledge_augmented):
                    model = self.registry.get_random_thinker()
                    if model is None:
                        logger.warning("No teacher models available for knowledge generation")
                        continue
                    tasks.append(
                        self._generate_knowledge_augmented(
                            model, query, knowledge, prompt_template
                        )
                    )

            except Exception as e:
                logger.error(f"Failed to generate knowledge: {e}. Skipping knowledge-augmented trajectories.")

        # Execute all generation tasks in parallel
        if not tasks:
            logger.error("No trajectories generated!")
            return Population()

        trajectories = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out failed generations
        valid_trajectories = []
        for i, traj in enumerate(trajectories):
            if isinstance(traj, Exception):
                logger.warning(f"Trajectory {i} generation failed: {traj}")
            elif traj is not None:
                valid_trajectories.append(traj)

        logger.info(f"Successfully generated {len(valid_trajectories)} trajectories")

        return Population(valid_trajectories)

    async def _generate_vanilla(
        self,
        model,
        query: str,
        prompt_template: Optional[str] = None
    ) -> Trajectory:
        """
        Generate a vanilla CoT (Formula 1): t_i = l_i(x)

        Uses enhanced prompt with explicit <|think|>...<|answer|> format
        matching test_complete_evolution.py implementation.

        Args:
            model: Teacher model to use
            query: The problem/question
            prompt_template: Optional custom prompt

        Returns:
            Generated trajectory
        """
        # Use enhanced prompt (matching test_complete_evolution.py)
        if prompt_template is None:
            # Determine prompt based on dataset
            if self.dataset_name in ["ChemCoTDataset", "ChemCoTBench"]:
                prompt = f"""Please solve the following chemistry problem step by step.

Your response must follow this exact format:
<|think|>
Your step-by-step reasoning process goes here. Explain your thought process clearly.
<|answer|>
Your final answer in JSON format: {{"Major Product": "SMILES string"}}

Question: {query}

Remember to:
1. Think step by step
2. Show your work clearly
3. Provide your final answer in the specified JSON format"""
            elif self.dataset_name == "BioProBench":
                prompt = f"""Please solve the following biological problem step by step.

Your response must follow this exact format:
<|think|>
Your step-by-step reasoning process goes here. Explain your thought process clearly.
<|answer|>
Your final answer in this format: [ANSWER_START]answer here[ANSWER_END]

Question: {query}

Remember to:
1. Think step by step
2. Show your work clearly
3. Provide your final answer in the specified format"""
            else:
                # Generic fallback
                prompt = VANILLA_TEMPLATE.format(query=query)
        else:
            prompt = prompt_template.format(query=query)

        # Use dataset-specific system prompt
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt}
        ]

        # Generate full output
        full_output = await model.generate_async(
            prompt="",  # Empty prompt when using messages
            messages=messages,
            stop=self.stop_sequences if self.stop_sequences else None
        )

        # Extract reasoning and answer using enhanced extractor
        reasoning, answer = extract_cot_and_answer(full_output)

        return Trajectory(
            query=query,
            answer=answer,
            reasoning=reasoning,
            source_model=model.model_name,
            generation_method="vanilla"
        )

    async def _generate_knowledge_augmented(
        self,
        model,
        query: str,
        knowledge: str,
        prompt_template: Optional[str] = None
    ) -> Trajectory:
        """
        Generate a knowledge-augmented CoT (Formula 2): t_j = l_j(x, K_x)

        Uses enhanced prompt with explicit <|think|>...<|answer|> format
        matching test_complete_evolution.py implementation.

        Args:
            model: Teacher model to use
            query: The problem/question
            knowledge: Knowledge snippet to provide
            prompt_template: Optional custom prompt

        Returns:
            Generated trajectory
        """
        # Use enhanced prompt (matching test_complete_evolution.py)
        if prompt_template is None:
            # Determine prompt based on dataset
            if self.dataset_name in ["ChemCoTDataset", "ChemCoTBench"]:
                prompt = f"""Please solve the following chemistry problem step by step. Use the provided knowledge to help you.

Relevant knowledge:
{knowledge}

Your response must follow this exact format:
<|think|>
Your step-by-step reasoning process goes here. Make sure to use the provided knowledge effectively.
<|answer|>
Your final answer in JSON format: {{"Major Product": "SMILES string"}}

Question: {query}

Remember to:
1. Use the provided knowledge effectively
2. Think step by step
3. Show your work clearly
4. Provide your final answer in the specified JSON format"""
            elif self.dataset_name == "BioProBench":
                prompt = f"""Please solve the following biological problem step by step. Use the provided knowledge to help you.

Relevant knowledge:
{knowledge}

Your response must follow this exact format:
<|think|>
Your step-by-step reasoning process goes here. Make sure to use the provided knowledge effectively.
<|answer|>
Your final answer in this format: [ANSWER_START]answer here[ANSWER_END]

Question: {query}

Remember to:
1. Use the provided knowledge effectively
2. Think step by step
3. Show your work clearly
4. Provide your final answer in the specified format"""
            else:
                # Generic fallback
                prompt = KNOWLEDGE_AUGMENTED_TEMPLATE.format(
                    query=query,
                    knowledge=knowledge
                )
        else:
            # Assume template has both {query} and {knowledge} placeholders
            prompt = prompt_template.format(
                query=query,
                knowledge=knowledge
            )

        # Use dataset-specific system prompt
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt}
        ]

        # Generate full output
        full_output = await model.generate_async(
            prompt="",  # Empty prompt when using messages
            messages=messages,
            stop=self.stop_sequences if self.stop_sequences else None
        )

        # Extract reasoning and answer using enhanced extractor
        reasoning, answer = extract_cot_and_answer(full_output)

        return Trajectory(
            query=query,
            answer=answer,
            reasoning=reasoning,
            knowledge=knowledge,
            source_model=model.model_name,
            generation_method="knowledge_augmented"
        )

    def _extract_answer(self, reasoning: str) -> str:
        """
        Extract the final answer from reasoning.

        This is a simple implementation. For production, you should use
        the task-specific answer extractors from Supplementary_Material.

        Args:
            reasoning: Full reasoning text

        Returns:
            Extracted answer
        """
        # Try to find the last line as the answer
        lines = reasoning.strip().split('\n')

        # Remove empty lines
        lines = [l.strip() for l in lines if l.strip()]

        if not lines:
            return ""

        # Last non-empty line is typically the answer
        answer = lines[-1]

        # Clean up common prefixes
        for prefix in ["Answer:", "The answer is", "Result:", "Final answer:"]:
            if answer.lower().startswith(prefix.lower()):
                answer = answer[len(prefix):].strip()
                break

        return answer

    def _extract_answer_json(self, reasoning: str) -> str:
        """
        Extract answer from JSON format (for ChemCoT datasets).

        Args:
            reasoning: Full reasoning text

        Returns:
            Extracted answer
        """
        # Look for JSON pattern
        import json

        pattern = r'\{[^{}]*"result"\s*:\s*"[^"]*"\s*\}'
        matches = re.findall(pattern, reasoning)

        if matches:
            try:
                # Parse the last match
                result = json.loads(matches[-1])
                return result.get("result", "")
            except json.JSONDecodeError:
                pass

        # Fallback to simple extraction
        return self._extract_answer(reasoning)

    def _extract_answer_bio(self, reasoning: str) -> str:
        """
        Extract answer from BioProBench format [ANSWER_START]...[ANSWER_END].

        Args:
            reasoning: Full reasoning text

        Returns:
            Extracted answer
        """
        pattern = r'\[ANSWER_START\](.*?)\[ANSWER_END\]'
        match = re.search(pattern, reasoning, re.DOTALL)

        if match:
            return match.group(1).strip()

        # Fallback: look for last occurrence
        if "[ANSWER_START]" in reasoning:
            parts = reasoning.split("[ANSWER_START]")
            last_part = parts[-1]
            if "[ANSWER_END]" in last_part:
                answer = last_part.split("[ANSWER_END]")[0].strip()
                return answer

        # Final fallback
        return self._extract_answer(reasoning)
