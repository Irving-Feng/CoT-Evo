"""
Unit tests for hybrid knowledge augmenter module.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from src.knowledge.hybrid import HybridKnowledgeAugmenter
from src.knowledge.generation import KnowledgeGenerator


class TestHybridKnowledgeAugmenter:
    """Test suite for HybridKnowledgeAugmenter class."""

    @pytest.fixture
    def mock_generator(self):
        """Create a mock knowledge generator."""
        generator = MagicMock(spec=KnowledgeGenerator)
        generator.generate_knowledge = AsyncMock()
        return generator

    @pytest.fixture
    def mock_knowledge_base(self):
        """Create a mock knowledge base (RAG)."""
        kb = MagicMock()
        kb.retrieve = AsyncMock()
        return kb

    @pytest.fixture
    def augmenter_with_generator_only(self, mock_generator):
        """Create augmenter with only generator (no RAG)."""
        return HybridKnowledgeAugmenter(generator=mock_generator, knowledge_base=None)

    @pytest.fixture
    def augmenter_with_rag_only(self, mock_knowledge_base):
        """Create augmenter with only RAG (no generator)."""
        return HybridKnowledgeAugmenter(generator=None, knowledge_base=mock_knowledge_base)

    @pytest.fixture
    def augmenter_hybrid(self, mock_generator, mock_knowledge_base):
        """Create augmenter with both RAG and generator."""
        return HybridKnowledgeAugmenter(
            generator=mock_generator,
            knowledge_base=mock_knowledge_base
        )

    @pytest.mark.asyncio
    async def test_generate_knowledge_fallback_to_generator(self, augmenter_with_rag_only, mock_knowledge_base):
        """Test fallback to generator when RAG fails (actually should raise error)."""
        # RAG returns None
        mock_knowledge_base.retrieve.return_value = None

        query = "What is the molecular weight of water?"
        ground_truth = "18.015 g/mol"

        # Should raise error since no generator available
        with pytest.raises(RuntimeError, match="No knowledge generation method"):
            await augmenter_with_rag_only.generate_knowledge(query, ground_truth)

    @pytest.mark.asyncio
    async def test_generate_knowledge_rag_succeeds(self, augmenter_hybrid, mock_knowledge_base, mock_generator):
        """Test RAG retrieval succeeds."""
        # RAG returns valid knowledge
        rag_knowledge = "Water has molecular weight 18.015 g/mol (H2O: 2H + O)"
        mock_knowledge_base.retrieve.return_value = rag_knowledge

        query = "What is the molecular weight of water?"
        ground_truth = "18.015 g/mol"

        knowledge = await augmenter_hybrid.generate_knowledge(query, ground_truth)

        # Should use RAG result
        assert knowledge == rag_knowledge
        mock_knowledge_base.retrieve.assert_called_once()
        # Generator should not be called
        mock_generator.generate_knowledge.assert_not_called()

    @pytest.mark.asyncio
    async def test_generate_knowledge_rag_fails_generator_succeeds(self, augmenter_hybrid, mock_knowledge_base, mock_generator):
        """Test fallback to generator when RAG returns None."""
        # RAG returns None
        mock_knowledge_base.retrieve.return_value = None

        # Generator returns valid knowledge
        gen_knowledge = "Water (H2O) molecular weight = 2*1.008 + 15.999 = 18.015 g/mol"
        mock_generator.generate_knowledge.return_value = gen_knowledge

        query = "What is the molecular weight of water?"
        ground_truth = "18.015 g/mol"

        knowledge = await augmenter_hybrid.generate_knowledge(query, ground_truth)

        # Should use generator result
        assert knowledge == gen_knowledge
        mock_knowledge_base.retrieve.assert_called_once()
        mock_generator.generate_knowledge.assert_called_once_with(query, ground_truth)

    @pytest.mark.asyncio
    async def test_generate_knowledge_generator_only(self, augmenter_with_generator_only, mock_generator):
        """Test knowledge generation with only generator (no RAG)."""
        gen_knowledge = "Photosynthesis converts light energy to chemical energy."
        mock_generator.generate_knowledge.return_value = gen_knowledge

        query = "What is photosynthesis?"
        ground_truth = "Energy conversion process in plants"

        knowledge = await augmenter_with_generator_only.generate_knowledge(query, ground_truth)

        assert knowledge == gen_knowledge
        mock_generator.generate_knowledge.assert_called_once_with(query, ground_truth)

    @pytest.mark.asyncio
    async def test_no_knowledge_method_available(self):
        """Test error when neither RAG nor generator is available."""
        augmenter = HybridKnowledgeAugmenter(generator=None, knowledge_base=None)

        with pytest.raises(RuntimeError, match="No knowledge generation method available"):
            await augmenter.generate_knowledge("Query", "Answer")

    @pytest.mark.asyncio
    async def test_format_knowledge_single_item(self, augmenter_with_generator_only):
        """Test formatting single knowledge item."""
        retrieved = ["Single piece of knowledge"]

        formatted = augmenter_with_generator_only._format_knowledge(retrieved)

        assert formatted == "Single piece of knowledge"

    @pytest.mark.asyncio
    async def test_format_knowledge_multiple_items(self, augmenter_with_generator_only):
        """Test formatting multiple knowledge items."""
        retrieved = [
            "Knowledge point 1",
            "Knowledge point 2",
            "Knowledge point 3"
        ]

        formatted = augmenter_with_generator_only._format_knowledge(retrieved)

        assert "Knowledge point 1" in formatted
        assert "Knowledge point 2" in formatted
        assert "Knowledge point 3" in formatted

    @pytest.mark.asyncio
    async def test_try_rag_empty_retrieval(self, augmenter_with_rag_only, mock_knowledge_base):
        """Test RAG retrieval returns empty list."""
        mock_knowledge_base.retrieve.return_value = []

        result = await augmenter_with_rag_only._try_rag("Test query")

        assert result is None

    @pytest.mark.asyncio
    async def test_is_reliable_with_content(self, augmenter_hybrid):
        """Test _is_reliable with actual content."""
        retrieved = ["Substantial knowledge content here"]

        assert augmenter_hybrid._is_reliable(retrieved) is True

    @pytest.mark.asyncio
    async def test_is_reliable_empty(self, augmenter_hybrid):
        """Test _is_reliable with empty list."""
        assert augmenter_hybrid._is_reliable([]) is False

    @pytest.mark.asyncio
    async def test_is_reliable_none(self, augmenter_hybrid):
        """Test _is_reliable with None."""
        assert augmenter_hybrid._is_reliable(None) is False


class TestHybridKnowledgeAugmenterReliability:
    """Test knowledge reliability scoring."""

    @pytest.fixture
    def augmenter(self):
        """Create augmenter for reliability tests."""
        mock_generator = MagicMock()
        mock_kb = MagicMock()
        return HybridKnowledgeAugmenter(generator=mock_generator, knowledge_base=mock_kb)

    def test_is_reliable_with_content(self, augmenter):
        """Test _is_reliable with actual content."""
        retrieved = [
            {"text": "Substantial knowledge content here", "score": 0.8}
        ]

        assert augmenter._is_reliable(retrieved) is True

    def test_is_reliable_with_low_score(self, augmenter):
        """Test _is_reliable with low confidence score."""
        retrieved = [
            {"text": "Some content", "score": 0.5}
        ]

        # Default threshold is 0.7
        assert augmenter._is_reliable(retrieved) is False

    def test_is_reliable_empty(self, augmenter):
        """Test _is_reliable with empty list."""
        assert augmenter._is_reliable([]) is False

    def test_is_reliable_none(self, augmenter):
        """Test _is_reliable with None."""
        assert augmenter._is_reliable(None) is False
