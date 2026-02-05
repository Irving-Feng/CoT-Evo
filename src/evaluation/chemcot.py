"""
ChemCoT evaluation framework.

This module provides evaluation functions for chemistry reasoning tasks,
including molecule property prediction, molecule editing, and reaction prediction.
"""

import logging
import re
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Try to import RDKit, but handle gracefully if not available
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, AllChem, DataStructs
    from rdkit.Chem.Fingerprints import FingerprintMols
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    logger.warning("RDKit not available. ChemCoT evaluation will be limited.")


# Valid molecular groups for editing tasks
GROUP_SET = {
    "carbon", "nitrogen", "oxygen", "fluorine", "phosphorus", "sulfur",
    "chlorine", "bromine", "iodine", "boron", "silicon", "selenium",
    "tellurium", "arsenic", "antimony", "bismuth", "benzene_ring",
    "hydrogen", "HBA", "HBD", "rot_bonds", "ring_count"
}


@dataclass
class ChemCoTMetrics:
    """Metrics for ChemCoT evaluation."""
    exact_match: bool = False
    property_accuracy: float = 0.0
    molecular_similarity: float = 0.0
    edit_valid: bool = False
    smiles_valid: bool = False
    error_message: Optional[str] = None


class ChemCoTEvaluator:
    """
    Evaluator for ChemCoT (Chemistry Chain-of-Thought) tasks.

    Supports multiple task types:
    - Molecule understanding: Property prediction, molecular weight, etc.
    - Molecule editing: Add, delete, substitute functional groups
    - Molecule optimization: Property optimization
    - Reaction prediction: Product prediction

    Attributes:
        strict_mode: If True, requires exact SMILES match for molecule editing
    """

    def __init__(self, strict_mode: bool = False):
        """
        Initialize the ChemCoT evaluator.

        Args:
            strict_mode: Whether to require exact SMILES matches
        """
        self.strict_mode = strict_mode

        if not RDKIT_AVAILABLE:
            logger.warning("RDKit not available - only basic text-based evaluation will work")

    def extract_smiles(self, text: str) -> Optional[str]:
        """
        Extract SMILES from text using regex patterns.

        Args:
            text: Text that may contain a SMILES string

        Returns:
            Extracted SMILES or None if not found
        """
        # Common SMILES patterns (look for bracketed patterns and common atoms)
        smiles_patterns = [
            r'([CNOcnoFfPSpsBbBr?Ii][A-Za-z0-9@+\-\[\]\(\)\\=#$]+)',  # Generic SMILES
            r'([CNOcno].{0,100})',  # Simple organic molecule pattern
        ]

        for pattern in smiles_patterns:
            matches = re.findall(pattern, text)
            if matches:
                # Return the longest match (most likely to be complete SMILES)
                return max(matches, key=len)

        return None

    def extract_number(self, text: str) -> Optional[float]:
        """
        Extract a numeric answer from text.

        Args:
            text: Text that may contain a number

        Returns:
            Extracted number or None if not found
        """
        # Look for numbers with optional decimal points and units
        patterns = [
            r'(\d+\.\d+)',  # Decimal numbers
            r'(\d+)',  # Integers
            r'([\d.]+\s*(?:g/mol|Da|kDa|MW))',  # Numbers with molecular weight units
        ]

        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                try:
                    return float(match.group(1).split()[0])  # Handle units
                except (ValueError, IndexError):
                    continue

        return None

    def evaluate_molecular_weight(
        self,
        predicted: str,
        ground_truth: str,
        tolerance: float = 0.01
    ) -> ChemCoTMetrics:
        """
        Evaluate molecular weight prediction.

        Args:
            predicted: Predicted answer (text or number)
            ground_truth: Ground truth molecular weight
            tolerance: Relative tolerance for correctness

        Returns:
            ChemCoTMetrics with evaluation results
        """
        metrics = ChemCoTMetrics()

        # Extract numbers
        pred_num = self.extract_number(predicted)
        gt_num = self.extract_number(ground_truth)

        if pred_num is None:
            metrics.error_message = f"Could not extract number from prediction: {predicted}"
            return metrics

        if gt_num is None:
            try:
                gt_num = float(ground_truth)
            except ValueError:
                metrics.error_message = f"Invalid ground truth: {ground_truth}"
                return metrics

        # Check if within tolerance
        relative_error = abs(pred_num - gt_num) / gt_num if gt_num != 0 else float('inf')
        metrics.exact_match = relative_error <= tolerance
        metrics.property_accuracy = max(0.0, 1.0 - relative_error)

        return metrics

    def evaluate_property_prediction(
        self,
        predicted: str,
        ground_truth: str,
        property_name: str = "logp"
    ) -> ChemCoTMetrics:
        """
        Evaluate molecular property prediction.

        Args:
            predicted: Predicted property value
            ground_truth: Ground truth property value
            property_name: Name of the property (for error messages)

        Returns:
            ChemCoTMetrics with evaluation results
        """
        metrics = ChemCoTMetrics()

        # Extract numbers
        pred_num = self.extract_number(predicted)
        gt_num = self.extract_number(ground_truth)

        if pred_num is None:
            metrics.error_message = f"Could not extract number from prediction: {predicted}"
            return metrics

        if gt_num is None:
            metrics.error_message = f"Invalid ground truth: {ground_truth}"
            return metrics

        # Calculate relative error
        if gt_num != 0:
            relative_error = abs(pred_num - gt_num) / gt_num
            metrics.property_accuracy = max(0.0, 1.0 - relative_error)
            metrics.exact_match = relative_error < 0.05  # Within 5%
        else:
            metrics.exact_match = pred_num == 0
            metrics.property_accuracy = 1.0 if metrics.exact_match else 0.0

        return metrics

    def is_valid_smiles(self, smiles: str) -> bool:
        """
        Check if a SMILES string is valid.

        Args:
            smiles: SMILES string to validate

        Returns:
            True if valid, False otherwise
        """
        if not RDKIT_AVAILABLE:
            # Basic validation: check if it looks like a SMILES
            return bool(smiles and len(smiles) > 0)

        try:
            mol = Chem.MolFromSmiles(smiles)
            return mol is not None
        except Exception:
            return False

    def calculate_molecular_property(
        self,
        smiles: str,
        property_name: str
    ) -> Optional[float]:
        """
        Calculate a molecular property using RDKit.

        Args:
            smiles: SMILES string
            property_name: Name of the property to calculate

        Returns:
            Property value or None if calculation fails
        """
        if not RDKIT_AVAILABLE:
            logger.warning("RDKit not available for property calculation")
            return None

        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None

            # Map property names to RDKit functions
            property_map = {
                'logp': Descriptors.MolLogP,
                'weight': Descriptors.MolWt,
                'qed': Descriptors.qed,
                'TPSA': Descriptors.TPSA,
                'HBA': Descriptors.NumHAcceptors,
                'HBD': Descriptors.NumHDonors,
                'rot_bonds': Descriptors.NumRotatableBonds,
                'ring_count': Descriptors.RingCount,
                'mr': Descriptors.MolMR,
            }

            if property_name in property_map:
                return property_map[property_name](mol)
            else:
                logger.warning(f"Unknown property: {property_name}")
                return None

        except Exception as e:
            logger.error(f"Error calculating property {property_name}: {e}")
            return None

    def count_molecular_group(self, smiles: str, group: str) -> Optional[int]:
        """
        Count occurrences of a molecular group in a molecule.

        Args:
            smiles: SMILES string
            group: Group name (e.g., "carbon", "nitrogen", "benzene_ring")

        Returns:
            Count of the group or None if calculation fails
        """
        if not RDKIT_AVAILABLE:
            logger.warning("RDKit not available for group counting")
            return None

        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None

            # Normalize group name
            if group == "benzene":
                group = "benzene_ring"

            # Count atoms by atomic number
            atomic_numbers = {
                "carbon": 6, "nitrogen": 7, "oxygen": 8,
                "fluorine": 9, "phosphorus": 15, "sulfur": 16,
                "chlorine": 17, "bromine": 35, "iodine": 53,
                "boron": 5, "silicon": 14, "selenium": 34,
            }

            if group in atomic_numbers:
                return sum(atom.GetAtomicNum() == atomic_numbers[group] for atom in mol.GetAtoms())

            # Special cases
            if group == "benzene_ring":
                # Count aromatic rings
                return sum(mol.GetRingInfo().IsAtomInRingOfSize(atom.GetIdx(), 6)
                          for atom in mol.GetAtoms()
                          if atom.GetIsAromatic())

            if group == "hydrogen":
                # Count implicit hydrogens
                return sum(atom.GetTotalNumHs() for atom in mol.GetAtoms())

            logger.warning(f"Unknown group: {group}")
            return None

        except Exception as e:
            logger.error(f"Error counting group {group}: {e}")
            return None

    def evaluate_molecule_edit(
        self,
        source_smiles: str,
        predicted_smiles: str,
        target_smiles: str,
        edit_type: str,
        group: Optional[str] = None,
        remove_group: Optional[str] = None,
        add_group: Optional[str] = None
    ) -> ChemCoTMetrics:
        """
        Evaluate molecule editing task.

        Args:
            source_smiles: Source molecule SMILES
            predicted_smiles: Predicted edited molecule SMILES
            target_smiles: Target molecule SMILES
            edit_type: Type of edit ('add', 'delete', 'substitute')
            group: Group to add/delete (for add/delete edits)
            remove_group: Group to remove (for substitute edits)
            add_group: Group to add (for substitute edits)

        Returns:
            ChemCoTMetrics with evaluation results
        """
        metrics = ChemCoTMetrics()

        # Validate SMILES
        metrics.smiles_valid = self.is_valid_smiles(predicted_smiles)
        if not metrics.smiles_valid:
            metrics.error_message = f"Invalid predicted SMILES: {predicted_smiles}"
            return metrics

        if not RDKIT_AVAILABLE:
            # Basic text-based evaluation
            metrics.exact_match = predicted_smiles.strip() == target_smiles.strip()
            metrics.edit_valid = metrics.exact_match
            return metrics

        # Check edit validity
        if edit_type == "add":
            metrics.edit_valid = self._check_add_valid(
                source_smiles, predicted_smiles, group
            )
        elif edit_type == "delete":
            metrics.edit_valid = self._check_delete_valid(
                source_smiles, predicted_smiles, group
            )
        elif edit_type == "substitute":
            metrics.edit_valid = self._check_substitute_valid(
                source_smiles, predicted_smiles, remove_group, add_group
            )
        else:
            metrics.error_message = f"Unknown edit type: {edit_type}"
            return metrics

        # Calculate molecular similarity
        metrics.molecular_similarity = self._calculate_similarity(
            predicted_smiles, target_smiles
        )

        # Exact match check
        if self.strict_mode:
            metrics.exact_match = predicted_smiles == target_smiles
        else:
            # Consider it correct if edit is valid and similarity is high
            metrics.exact_match = metrics.edit_valid and metrics.molecular_similarity > 0.9

        return metrics

    def _check_add_valid(self, src: str, tgt: str, group: str) -> bool:
        """Check if add edit is valid."""
        if group == "benzene":
            group = "benzene_ring"

        if group not in GROUP_SET:
            logger.warning(f"Invalid group: {group}")
            return False

        src_count = self.count_molecular_group(src, group)
        tgt_count = self.count_molecular_group(tgt, group)

        if src_count is None or tgt_count is None:
            return False

        return tgt_count == src_count + 1

    def _check_delete_valid(self, src: str, tgt: str, group: str) -> bool:
        """Check if delete edit is valid."""
        if group == "benzene":
            group = "benzene_ring"

        if group not in GROUP_SET:
            logger.warning(f"Invalid group: {group}")
            return False

        src_count = self.count_molecular_group(src, group)
        tgt_count = self.count_molecular_group(tgt, group)

        if src_count is None or tgt_count is None:
            return False

        return tgt_count == src_count - 1

    def _check_substitute_valid(
        self,
        src: str,
        tgt: str,
        remove_group: str,
        add_group: str
    ) -> bool:
        """Check if substitute edit is valid."""
        if remove_group == "benzene":
            remove_group = "benzene_ring"
        if add_group == "benzene":
            add_group = "benzene_ring"

        if remove_group not in GROUP_SET or add_group not in GROUP_SET:
            logger.warning(f"Invalid groups: {remove_group}, {add_group}")
            return False

        remove_valid = self._check_delete_valid(src, tgt, remove_group)
        add_valid = self._check_add_valid(src, tgt, add_group)

        return remove_valid and add_valid

    def _calculate_similarity(
        self,
        mol1: str,
        mol2: str,
        fingerprint_type: str = 'Morgan',
        similarity_metric: str = 'Tanimoto'
    ) -> float:
        """
        Calculate molecular similarity using fingerprints.

        Args:
            mol1: First molecule SMILES
            mol2: Second molecule SMILES
            fingerprint_type: Type of fingerprint to use
            similarity_metric: Similarity metric to use

        Returns:
            Similarity score (0-1)
        """
        if not RDKIT_AVAILABLE:
            return 0.0

        try:
            m1 = Chem.MolFromSmiles(mol1)
            m2 = Chem.MolFromSmiles(mol2)

            if m1 is None or m2 is None:
                return 0.0

            # Generate fingerprints
            if fingerprint_type == 'Morgan':
                fp1 = AllChem.GetMorganFingerprintAsBitVect(m1, radius=2, nBits=2048)
                fp2 = AllChem.GetMorganFingerprintAsBitVect(m2, radius=2, nBits=2048)
            else:
                fp1 = FingerprintMols.FingerprintMol(m1)
                fp2 = FingerprintMols.FingerprintMol(m2)

            # Calculate similarity
            if similarity_metric == 'Tanimoto':
                return DataStructs.TanimotoSimilarity(fp1, fp2)
            else:
                logger.warning(f"Unsupported similarity metric: {similarity_metric}")
                return 0.0

        except Exception as e:
            logger.error(f"Error calculating similarity: {e}")
            return 0.0

    def evaluate_general_chemistry(
        self,
        predicted: str,
        ground_truth: str
    ) -> ChemCoTMetrics:
        """
        Evaluate general chemistry question (fallback).

        Uses text-based similarity and numeric extraction.

        Args:
            predicted: Predicted answer
            ground_truth: Ground truth answer

        Returns:
            ChemCoTMetrics with evaluation results
        """
        metrics = ChemCoTMetrics()

        # Try to extract and compare numbers
        pred_num = self.extract_number(predicted)
        gt_num = self.extract_number(ground_truth)

        if pred_num is not None and gt_num is not None:
            # Numeric comparison
            if gt_num != 0:
                relative_error = abs(pred_num - gt_num) / abs(gt_num)
                metrics.exact_match = relative_error < 0.05
                metrics.property_accuracy = max(0.0, 1.0 - relative_error)
            else:
                metrics.exact_match = pred_num == gt_num
                metrics.property_accuracy = 1.0 if metrics.exact_match else 0.0
        else:
            # Text-based comparison
            pred_lower = predicted.lower().strip()
            gt_lower = ground_truth.lower().strip()

            # Exact match
            if pred_lower == gt_lower:
                metrics.exact_match = True
                metrics.property_accuracy = 1.0
            # Contains match
            elif gt_lower in pred_lower or pred_lower in gt_lower:
                metrics.exact_match = True
                metrics.property_accuracy = 0.8
            else:
                metrics.exact_match = False
                metrics.property_accuracy = 0.0

        return metrics


def create_chemcot_evaluator(strict_mode: bool = False) -> ChemCoTEvaluator:
    """
    Convenience function to create a ChemCoT evaluator.

    Args:
        strict_mode: Whether to require exact SMILES matches

    Returns:
        Configured ChemCoTEvaluator instance
    """
    return ChemCoTEvaluator(strict_mode=strict_mode)
