"""
Unit tests for knowledge generation module.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from src.knowledge.generation import KnowledgeGenerator


class TestKnowledgeGenerator:
    """Test suite for KnowledgeGenerator class."""

    @pytest.fixture
    def mock_model(self):
        """Create a mock model."""
        model = MagicMock()
        model.generate_async = AsyncMock()
        return model

    @pytest.fixture
    def generator(self, mock_model):
        """Create a KnowledgeGenerator instance."""
        return KnowledgeGenerator(mock_model)

    @pytest.mark.asyncio
    async def test_generate_knowledge_success(self, generator, mock_model):
        """Test successful knowledge generation."""
        query = "What is the molecular weight of water?"
        ground_truth = "18.015 g/mol"

        # Mock the model response
        mock_response = """[RESULT_START]
Water (H2O) has a molecular structure consisting of two hydrogen atoms and one oxygen atom.
The atomic weights are: H = 1.008 u, O = 15.999 u.
Therefore, the molecular weight of H2O = 2(1.008) + 15.999 = 18.015 g/mol.
[RESULT_END]"""

        mock_model.generate_async.return_value = mock_response

        # Generate knowledge
        knowledge = await generator.generate_knowledge(query, ground_truth)

        # Verify
        assert "molecular weight" in knowledge.lower() or "H2O" in knowledge
        assert len(knowledge) > 0
        mock_model.generate_async.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_knowledge_no_markers(self, generator, mock_model):
        """Test knowledge generation when response has no markers."""
        query = "What is photosynthesis?"
        ground_truth = "The process by which plants convert light energy to chemical energy."

        # Mock response without markers
        mock_response = "Photosynthesis is the process plants use to convert light energy."
        mock_model.generate_async.return_value = mock_response

        # Generate knowledge
        knowledge = await generator.generate_knowledge(query, ground_truth)

        # Verify fallback behavior
        assert knowledge == mock_response.strip()

    @pytest.mark.asyncio
    async def test_extract_knowledge_with_markers(self, generator):
        """Test knowledge extraction with RESULT_START/END markers."""
        response = """Some prefix text.

[RESULT_START]
This is the extracted knowledge.
It should be returned as is.
[RESULT_END]

Some suffix text."""

        knowledge = generator._extract_knowledge(response)
        assert "This is the extracted knowledge" in knowledge
        assert "prefix" not in knowledge
        assert "suffix" not in knowledge

    @pytest.mark.asyncio
    async def test_extract_knowledge_without_markers(self, generator):
        """Test knowledge extraction without markers."""
        response = "Plain response without any markers."

        knowledge = generator._extract_knowledge(response)
        assert knowledge == "Plain response without any markers."

    @pytest.mark.asyncio
    async def test_extract_knowledge_empty_response(self, generator):
        """Test knowledge extraction with empty response."""
        knowledge = generator._extract_knowledge("")
        assert knowledge == ""

    @pytest.mark.asyncio
    async def test_generate_knowledge_model_error(self, generator, mock_model):
        """Test knowledge generation when model raises an error."""
        mock_model.generate_async.side_effect = Exception("API error")

        with pytest.raises(Exception, match="API error"):
            await generator.generate_knowledge("Query", "Answer")


class TestKnowledgeGeneratorIntegration:
    """Integration tests for KnowledgeGenerator with realistic scenarios."""

    @pytest.fixture
    def generator(self):
        """Create generator with a realistic mock."""
        mock_model = MagicMock()
        mock_model.generate_async = AsyncMock()
        return KnowledgeGenerator(mock_model)

    @pytest.mark.asyncio
    async def test_chemistry_knowledge_generation(self, generator):
        """Test knowledge generation for chemistry problems."""
        query = "Calculate the pH of a 0.1 M HCl solution."
        ground_truth = "pH = 1.0"

        # Realistic mock response
        mock_response = """[RESULT_START]
HCl is a strong acid that completely dissociates in water: HCl → H+ + Cl-
For a 0.1 M HCl solution, [H+] = 0.1 M
pH = -log10[H+] = -log10(0.1) = 1.0
Strong acids have pH = -log10(concentration)
[RESULT_END]"""

        generator.model.generate_async.return_value = mock_response

        knowledge = await generator.generate_knowledge(query, ground_truth)

        assert "HCl" in knowledge or "strong acid" in knowledge
        assert "pH" in knowledge

    @pytest.mark.asyncio
    async def test_biology_knowledge_generation(self, generator):
        """Test knowledge generation for biology problems."""
        query = "What is the role of mitochondria in a cell?"
        ground_truth = "Mitochondria are the powerhouse of the cell, producing ATP through cellular respiration."

        mock_response = """[RESULT_START]
Mitochondria are organelles responsible for ATP production through cellular respiration.
They have a double membrane structure with cristae to increase surface area.
Key processes: Krebs cycle, electron transport chain, oxidative phosphorylation.
ATP synthase uses proton gradient to produce ATP from ADP.
[RESULT_END]"""

        generator.model.generate_async.return_value = mock_response

        knowledge = await generator.generate_knowledge(query, ground_truth)

        assert "ATP" in knowledge or "energy" in knowledge
        assert "mitochondria" in knowledge.lower()
