"""
Hybrid knowledge augmentation module for CoT-Evo framework.

This module implements the hybrid knowledge augmentation strategy:
- Priority 1: RAG retrieval (if available and reliable)
- Priority 2: Generation-based (using LLM to extract knowledge from answers)
"""

import logging
from typing import Optional, List
from .generation import KnowledgeGenerator


logger = logging.getLogger(__name__)


class HybridKnowledgeAugmenter:
    """
    Hybrid knowledge augmenter combining RAG and generation-based approaches.

    Implements the mixed-mode knowledge augmentation strategy described in the paper:
    1. First tries to retrieve relevant knowledge from a knowledge base (RAG)
    2. Falls back to LLM-based generation if RAG is unavailable or unreliable
    """

    def __init__(
        self,
        generator: Optional[KnowledgeGenerator] = None,
        knowledge_base: Optional["KnowledgeBase"] = None,
        rag_threshold: float = 0.7
    ):
        """
        Initialize the hybrid knowledge augmenter.

        Args:
            generator: LLM-based knowledge generator
            knowledge_base: Optional RAG knowledge base for retrieval
            rag_threshold: Minimum confidence threshold for using RAG results
        """
        self.generator = generator
        self.knowledge_base = knowledge_base
        self.rag_threshold = rag_threshold

    async def generate_knowledge(
        self,
        query: str,
        ground_truth: str
    ) -> str:
        """
        Generate knowledge snippet using hybrid strategy.

        Args:
            query: The problem/question
            ground_truth: The correct answer

        Returns:
            Generated knowledge snippet

        Raises:
            RuntimeError: If both RAG and generation are unavailable
        """
        # Strategy 1: Try RAG retrieval
        if self.knowledge_base is not None:
            try:
                retrieved = await self._try_rag(query)
                if retrieved:
                    logger.debug(f"Using RAG-retrieved knowledge for query: {query[:50]}...")
                    return retrieved
            except Exception as e:
                logger.warning(f"RAG retrieval failed: {e}. Falling back to generation.")

        # Strategy 2: Fall back to LLM-based generation
        if self.generator is not None:
            try:
                logger.debug(f"Using LLM-based knowledge generation for query: {query[:50]}...")
                knowledge = await self.generator.generate_knowledge(query, ground_truth)
                return knowledge
            except Exception as e:
                logger.error(f"Knowledge generation failed: {e}")
                raise

        # No method available
        raise RuntimeError(
            "No knowledge generation method available. "
            "Please provide either a knowledge_base (RAG) or a generator (LLM)."
        )

    async def _try_rag(self, query: str) -> Optional[str]:
        """
        Try to retrieve knowledge using RAG.

        Args:
            query: The problem/question

        Returns:
            Retrieved knowledge if reliable, None otherwise
        """
        # Retrieve top-k documents
        results = await self.knowledge_base.retrieve(query, top_k=3)

        if not results:
            return None

        # Check if retrieval is confident enough
        if self._is_reliable(results):
            return self._format_knowledge(results)

        return None

    def _is_reliable(self, results: List) -> bool:
        """
        Check if RAG results are reliable enough to use.

        Args:
            results: List of retrieval results with scores

        Returns:
            True if results should be used, False otherwise
        """
        if not results:
            return False

        # Check if the top result meets the confidence threshold
        top_score = results[0].get("score", 0.0) if isinstance(results[0], dict) else 0.0

        return top_score >= self.rag_threshold

    def _format_knowledge(self, results: List) -> str:
        """
        Format RAG results into a knowledge snippet.

        Args:
            results: List of retrieval results

        Returns:
            Formatted knowledge snippet
        """
        knowledge_parts = []

        for i, result in enumerate(results[:3], 1):  # Use top 3
            if isinstance(result, dict):
                text = result.get("text", result.get("content", ""))
            else:
                text = str(result)

            knowledge_parts.append(f"{i}. {text}")

        return "\n\n".join(knowledge_parts)


class KnowledgeBase:
    """
    Abstract base class for RAG knowledge bases.
    """

    async def retrieve(self, query: str, top_k: int = 3) -> List:
        """
        Retrieve relevant knowledge for a query.

        Args:
            query: The problem/question
            top_k: Number of results to retrieve

        Returns:
            List of retrieval results with scores
        """
        raise NotImplementedError("Subclasses must implement retrieve()")


class MockKnowledgeBase(KnowledgeBase):
    """
    Mock knowledge base for testing.

    Returns simple placeholder knowledge without actual retrieval.
    """

    def __init__(self, knowledge_snippets: List[str]):
        """
        Initialize with mock knowledge snippets.

        Args:
            knowledge_snippets: List of mock knowledge snippets
        """
        self.knowledge_snippets = knowledge_snippets

    async def retrieve(self, query: str, top_k: int = 3) -> List:
        """
        Return mock knowledge results.

        Args:
            query: The problem/question (ignored)
            top_k: Number of results to return

        Returns:
            Mock results with high confidence scores
        """
        results = []
        for i, snippet in enumerate(self.knowledge_snippets[:top_k]):
            results.append({
                "text": snippet,
                "score": 0.9  # High confidence
            })

        return results
