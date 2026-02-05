"""
Knowledge generation module for CoT-Evo framework.

This module implements the generation-based knowledge augmentation,
which uses an LLM to extract domain knowledge from correct answers.
"""

import logging
from typing import Optional
from ..models.base import LLMProvider
from ..initialization.prompts import KNOWLEDGE_GENERATION_PROMPT


logger = logging.getLogger(__name__)


class KnowledgeGenerator:
    """
    Generate knowledge snippets using LLM.

    This implements the generation-based approach for knowledge augmentation,
    where an LLM analyzes the correct answer and extracts necessary domain knowledge.
    """

    def __init__(self, model: LLMProvider):
        """
        Initialize the knowledge generator.

        Args:
            model: LLM to use for knowledge generation (typically a high-capability model)
        """
        self.model = model

    async def generate_knowledge(
        self,
        query: str,
        ground_truth: str
    ) -> str:
        """
        Generate knowledge snippet from the ground truth answer.

        This implements the reflective reasoning process described in the paper:
        use the correct answer to identify necessary knowledge for solving the query.

        Args:
            query: The problem/question
            ground_truth: The correct answer

        Returns:
            Generated knowledge snippet
        """
        prompt = KNOWLEDGE_GENERATION_PROMPT.format(
            query=query,
            ground_truth=ground_truth
        )

        try:
            response = await self.model.generate_async(prompt)
            knowledge = self._extract_knowledge(response)
            return knowledge

        except Exception as e:
            logger.error(f"Failed to generate knowledge: {e}")
            raise

    def _extract_knowledge(self, response: str) -> str:
        """
        Extract knowledge from LLM response.

        Args:
            response: Full LLM response

        Returns:
            Extracted knowledge snippet
        """
        # Look for knowledge markers
        start_marker = "[KNOWLEDGE_START]"
        end_marker = "[KNOWLEDGE_END]"

        start_idx = response.find(start_marker)
        end_idx = response.find(end_marker)

        if start_idx != -1 and end_idx != -1:
            # Extract content between markers
            knowledge = response[start_idx + len(start_marker):end_idx].strip()
            return knowledge
        else:
            # Fallback: return entire response
            logger.warning("Knowledge markers not found, using full response")
            return response.strip()


class MockKnowledgeGenerator(KnowledgeGenerator):
    """
    Mock knowledge generator for testing.

    Returns simple placeholder knowledge instead of calling an LLM.
    """

    async def generate_knowledge(
        self,
        query: str,
        ground_truth: str
    ) -> str:
        """
        Generate mock knowledge for testing.

        Args:
            query: The problem/question
            ground_truth: The correct answer

        Returns:
            Mock knowledge snippet
        """
        # Return a simple placeholder
        return "Domain knowledge for solving the problem."
