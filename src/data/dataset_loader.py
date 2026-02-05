"""
Dataset loader for CoT-Evo framework.

This module provides utilities to load datasets and their configurations
from the datasets.yaml registry.
"""

import logging
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import yaml

logger = logging.getLogger(__name__)


@dataclass
class DatasetConfig:
    """
    Configuration for a dataset from datasets.yaml.

    Attributes:
        name: Dataset name (e.g., "ChemCoTDataset")
        description: Dataset description
        data_format: Format of data files (e.g., "json")
        train_path: Path to training data
        test_path: Path to test data
        length_percentiles: Dict with 'lower' and 'upper' bounds
        system_prompt: System prompt for the dataset
        tasks: Dict of task-specific configurations
    """
    name: str
    description: str
    data_format: str
    train_path: str
    test_path: str
    length_percentiles: Dict[str, int]
    system_prompt: str
    tasks: Dict[str, Dict[str, Any]]

    # Optional fields
    oracle_path: Optional[str] = None
    requires_exact_match: bool = True


class DatasetLoader:
    """
    Loads datasets and their configurations from datasets.yaml.
    """

    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize dataset loader.

        Args:
            config_path: Path to datasets.yaml (default: config/datasets.yaml)
        """
        if config_path is None:
            config_path = Path(__file__).parent.parent.parent / "config" / "datasets.yaml"

        self.config_path = Path(config_path)
        self._datasets_cache: Optional[Dict[str, Dict]] = None

    def _load_datasets_config(self) -> Dict[str, Dict]:
        """Load and cache the datasets.yaml file."""
        if self._datasets_cache is not None:
            return self._datasets_cache

        if not self.config_path.exists():
            raise FileNotFoundError(f"Datasets config not found: {self.config_path}")

        with open(self.config_path) as f:
            config = yaml.safe_load(f)

        self._datasets_cache = config.get('datasets', {})
        return self._datasets_cache

    def list_datasets(self) -> List[str]:
        """
        List all available dataset names.

        Returns:
            List of dataset names registered in datasets.yaml
        """
        config = self._load_datasets_config()
        return list(config.keys())

    def get_dataset_config(self, dataset_name: str) -> DatasetConfig:
        """
        Get configuration for a specific dataset.

        Args:
            dataset_name: Name of the dataset (e.g., "ChemCoTDataset")

        Returns:
            DatasetConfig object with all dataset parameters

        Raises:
            ValueError: If dataset_name is not found in datasets.yaml
        """
        config = self._load_datasets_config()

        if dataset_name not in config:
            available = ", ".join(config.keys())
            raise ValueError(
                f"Dataset '{dataset_name}' not found. "
                f"Available datasets: {available}"
            )

        dataset_dict = config[dataset_name]

        # Extract common fields
        train_path = dataset_dict.get('train_path', '')
        test_path = dataset_dict.get('test_path', '')
        length_percentiles = dataset_dict.get('length_percentiles', {'lower': 1000, 'upper': 4000})
        system_prompt = dataset_dict.get('system_prompt', '')
        tasks = dataset_dict.get('tasks', {})
        oracle_path = dataset_dict.get('oracle_path')

        return DatasetConfig(
            name=dataset_dict.get('name', dataset_name),
            description=dataset_dict.get('description', ''),
            data_format=dataset_dict.get('data_format', 'json'),
            train_path=train_path,
            test_path=test_path,
            length_percentiles=length_percentiles,
            system_prompt=system_prompt,
            tasks=tasks,
            oracle_path=oracle_path
        )

    def load_dataset_data(
        self,
        dataset_name: str,
        max_samples: int = -1,
        split: str = "train"
    ) -> List[Dict[str, Any]]:
        """
        Load dataset data from JSON file.

        Args:
            dataset_name: Name of the dataset
            max_samples: Maximum number of samples to load (-1 for all)
            split: Which split to load ("train" or "test")

        Returns:
            List of dataset samples (dictionaries) with normalized 'answer' field

        Raises:
            FileNotFoundError: If data file doesn't exist
            ValueError: If dataset_name is not found
        """
        import uuid

        config = self.get_dataset_config(dataset_name)

        # Determine file path
        if split == "train":
            file_path = Path(config.train_path)
        elif split == "test":
            file_path = Path(config.test_path)
        else:
            raise ValueError(f"Invalid split: {split}. Use 'train' or 'test'.")

        if not file_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {file_path}")

        # Load JSON data
        with open(file_path) as f:
            data = json.load(f)

        # Apply max_samples filter
        if max_samples >= 0:
            data = data[:max_samples]

        # Normalize data format to ensure 'answer' field exists
        normalized_data = []
        for sample in data:
            # Ensure required fields exist
            if 'query' not in sample:
                logger.warning(f"Sample missing 'query' field, skipping: {sample.get('id', 'unknown')}")
                continue

            # Extract ground truth from various possible fields
            # Priority: answer > meta['gt'] > meta['answer'] > ground_truth
            if 'answer' in sample:
                answer = sample['answer']
            elif 'meta' in sample:
                meta = sample['meta']
                if isinstance(meta, dict):
                    answer = meta.get('gt', meta.get('answer', ''))
                else:
                    # Try to parse as JSON string
                    try:
                        meta_dict = json.loads(meta) if isinstance(meta, str) else meta
                        answer = meta_dict.get('gt', meta_dict.get('answer', ''))
                    except:
                        answer = str(meta) if meta else ''
            else:
                answer = sample.get('ground_truth', '')

            normalized_data.append({
                'id': sample.get('id', str(uuid.uuid4())),
                'query': sample['query'],
                'answer': answer,
                'task': sample.get('task', ''),
                'subtask': sample.get('subtask', ''),
                'knowledge': sample.get('knowledge', None),
                'meta': sample.get('meta', {})
            })

        if max_samples >= 0:
            logger.info(f"Loaded {len(normalized_data)} samples (max_samples={max_samples})")
        else:
            logger.info(f"Loaded all {len(normalized_data)} samples")

        return normalized_data

    def get_length_percentiles(self, dataset_name: str) -> Dict[str, int]:
        """
        Get length percentiles for a dataset.

        Args:
            dataset_name: Name of the dataset

        Returns:
            Dict with 'lower' and 'upper' percentile values (in words/tokens)
        """
        config = self.get_dataset_config(dataset_name)
        return config.length_percentiles

    def get_system_prompt(self, dataset_name: str) -> str:
        """
        Get system prompt for a dataset.

        Args:
            dataset_name: Name of the dataset

        Returns:
            System prompt string
        """
        config = self.get_dataset_config(dataset_name)
        return config.system_prompt
