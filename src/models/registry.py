"""
Model registry for CoT-Evo framework.

This module manages all LLM models used in the framework, including teacher models,
embedding models, judge models, etc. It supports flexible model selection strategies
to enable both automatic and manual model assignment for different operations.
"""

import os
import yaml
from typing import Dict, List, Optional, Any
from pathlib import Path
import logging

from .base import LLMProvider, EmbeddingProvider, GenerationConfig
from .openai_provider import OpenAIProvider, OpenAIEmbeddingProvider
from ..core.trajectory import Trajectory


logger = logging.getLogger(__name__)


class ModelRegistry:
    """
    Registry for managing all LLM models in the CoT-Evo framework.

    This class handles:
    - Registration of multiple teacher models
    - Specialized models (embedding, judge, knowledge generator)
    - Model selection strategies (auto vs. global)
    - Configuration loading from YAML files
    """

    def __init__(self, config_path: Optional[str] = None, load_from_env: bool = False):
        """
        Initialize the model registry.

        Args:
            config_path: Optional path to models.yaml configuration file
            load_from_env: Whether to automatically load models from .env file (default: False)
                           NOTE: It is recommended to configure models directly in config/models.yaml
        """
        self.models: Dict[str, LLMProvider] = {}
        self.thinkers: List[str] = []  # List of teacher model names
        self.embedding_model: Optional[EmbeddingProvider] = None
        self.judge_model: Optional[LLMProvider] = None
        self.knowledge_generator_model: Optional[LLMProvider] = None
        self.global_operator_model: Optional[LLMProvider] = None

        # Model selection mode: "auto" or "global"
        self.selection_mode: str = "auto"

        # Auto-load from environment variables if requested (not recommended)
        if load_from_env:
            logger.warning("Loading models from environment variables is not recommended. "
                          "Please use config/models.yaml instead.")
            self.load_from_env()

        # Load from config file if provided
        if config_path:
            self.load_from_config(config_path)

    def load_from_config(self, config_path: str) -> None:
        """
        Load models from a YAML configuration file.

        This method loads model configurations directly from the YAML file.
        All model credentials should be configured directly in the YAML file.

        Args:
            config_path: Path to models.yaml file
        """
        config_file = Path(config_path)

        if not config_file.exists():
            raise FileNotFoundError(f"Model config file not found: {config_path}")

        with open(config_file) as f:
            config = yaml.safe_load(f)

        # Register teacher models (thinkers)
        if "models" in config and "thinkers" in config["models"]:
            for thinker_config in config["models"]["thinkers"]:
                self._register_thinker_from_config(thinker_config)

        # Register embedding model
        if "models" in config and "embedding" in config["models"]:
            self._register_embedding_from_config(config["models"]["embedding"])

        # Register judge model
        if "models" in config and "judge" in config["models"]:
            self._register_judge_from_config(config["models"]["judge"])

        # Register knowledge generator model
        if "models" in config and "knowledge_generator" in config["models"]:
            self._register_knowledge_generator_from_config(
                config["models"]["knowledge_generator"]
            )

        # Set global operator model
        if "models" in config and "global_operator" in config["models"]:
            global_op_name = config["models"]["global_operator"]
            if global_op_name and global_op_name in self.models:
                self.global_operator_model = self.models[global_op_name]
                self.selection_mode = "global"
                logger.info(f"Using global operator model: {global_op_name}")

        # Set selection mode from config
        if "selection_strategy" in config and "operation" in config["selection_strategy"]:
            self.selection_mode = config["selection_strategy"]["operation"]

        logger.info(f"Loaded {len(self.thinkers)} teacher models from {config_path}")
        logger.info(f"Model selection mode: {self.selection_mode}")

    def _register_thinker_from_config(self, config: Dict[str, Any]) -> None:
        """
        Register a teacher model from configuration.

        Args:
            config: Model configuration dictionary containing:
                    - name: Unique identifier for the model
                    - base_url: API endpoint URL
                    - api_key: API key for authentication
                    - model_name: Actual model name for API calls
                    - provider: Provider type (default: "openai")
        """
        name = config["name"]
        provider_type = config.get("provider", "openai")

        # Read values directly from config (no environment variable substitution)
        base_url = config["base_url"]
        api_key = config["api_key"]
        model_name = config.get("model_name", name)

        if provider_type == "openai":
            provider = OpenAIProvider(
                model_name=model_name,
                base_url=base_url,
                api_key=api_key,
                timeout=config.get("timeout", 120),
                max_retries=config.get("max_retries", 3),
            )
            self.register_model(name, provider, role="thinker")
            logger.info(f"Registered teacher model: {name} (model_name={model_name})")
        else:
            raise ValueError(f"Unknown provider type: {provider_type}")

    def _register_embedding_from_config(self, config: Dict[str, Any]) -> None:
        """
        Register embedding model from configuration.

        Args:
            config: Model configuration dictionary
        """
        provider_type = config.get("provider", "openai")

        # Read values directly from config
        base_url = config["base_url"]
        api_key = config["api_key"]
        model_name = config.get("model_name", config["name"])
        embedding_dim = config.get("embedding_dim", 1536)

        if provider_type == "openai":
            self.embedding_model = OpenAIEmbeddingProvider(
                model_name=model_name,
                embedding_dim=embedding_dim,
                base_url=base_url,
                api_key=api_key,
            )
            logger.info(f"Registered embedding model: {model_name} (dim={embedding_dim})")
        else:
            raise ValueError(f"Unknown provider type: {provider_type}")

    def _register_judge_from_config(self, config: Dict[str, Any]) -> None:
        """
        Register judge model from configuration.

        Args:
            config: Model configuration dictionary
        """
        provider_type = config.get("provider", "openai")

        # Read values directly from config
        base_url = config["base_url"]
        api_key = config["api_key"]
        model_name = config.get("model_name", config["name"])

        if provider_type == "openai":
            self.judge_model = OpenAIProvider(
                model_name=model_name,
                base_url=base_url,
                api_key=api_key,
            )
            logger.info(f"Registered judge model: {model_name}")
        else:
            raise ValueError(f"Unknown provider type: {provider_type}")

    def _register_knowledge_generator_from_config(self, config: Dict[str, Any]) -> None:
        """
        Register knowledge generator model from configuration.

        Args:
            config: Model configuration dictionary
        """
        provider_type = config.get("provider", "openai")

        # Read values directly from config
        base_url = config["base_url"]
        api_key = config["api_key"]
        model_name = config.get("model_name", config["name"])

        if provider_type == "openai":
            self.knowledge_generator_model = OpenAIProvider(
                model_name=model_name,
                base_url=base_url,
                api_key=api_key,
            )
            logger.info(f"Registered knowledge generator model: {model_name}")
        else:
            raise ValueError(f"Unknown provider type: {provider_type}")

    def load_from_env(self) -> None:
        """
        Automatically load models from environment variables.

        This method scans for environment variables following the pattern:
        - MODEL_<NAME>_BASE_URL
        - MODEL_<NAME>_API_KEY
        - MODEL_<NAME>_MODEL_NAME

        And loads them as teacher models. Also loads specialized models:
        - EMBEDDING_MODEL_*: Embedding model
        - JUDGE_MODEL_*: Judge model
        - KNOWLEDGE_GEN_MODEL_*: Knowledge generator model
        - GLOBAL_OPERATOR_MODEL: Global operator model (if set)

        This allows dynamic configuration of any number of models via .env file.
        """
        import os
        env_vars = os.environ.copy()

        # Find all teacher models by looking for MODEL_*_BASE_URL pattern
        teacher_models = set()
        for key in env_vars.keys():
            if key.startswith("MODEL_") and key.endswith("_BASE_URL"):
                # Extract model name: MODEL_DEEPSEEK_R1_BASE_URL -> DEEPSEEK_R1
                model_name = key[6:-10]  # Remove "MODEL_" prefix and "_BASE_URL" suffix
                if model_name:  # Ensure non-empty
                    teacher_models.add(model_name)

        # Register each teacher model
        for model_name in teacher_models:
            try:
                base_url = env_vars.get(f"MODEL_{model_name}_BASE_URL")
                api_key = env_vars.get(f"MODEL_{model_name}_API_KEY")
                model_id = env_vars.get(f"MODEL_{model_name}_MODEL_NAME", model_name.lower().replace("_", "-"))

                if base_url and api_key:
                    provider = OpenAIProvider(
                        model_name=model_id,
                        base_url=base_url,
                        api_key=api_key,
                        timeout=120,
                        max_retries=3,
                    )
                    # Use model_id as the registry name (more user-friendly)
                    self.register_model(model_id, provider, role="thinker")
                    logger.info(f"Auto-loaded teacher model: {model_id} ({model_name})")
                else:
                    logger.warning(f"Incomplete configuration for MODEL_{model_name}, skipping")
            except Exception as e:
                logger.error(f"Failed to load model {model_name}: {e}")

        # Load specialized models
        self._load_specialized_model_from_env(env_vars, "EMBEDDING_MODEL", "embedding")
        self._load_specialized_model_from_env(env_vars, "JUDGE_MODEL", "judge")
        self._load_specialized_model_from_env(env_vars, "KNOWLEDGE_GEN_MODEL", "knowledge_generator")

        # Load global operator model if set
        global_op_name = env_vars.get("GLOBAL_OPERATOR_MODEL")
        if global_op_name:
            self.set_global_operator(global_op_name)

        logger.info(f"Auto-loaded {len(self.thinkers)} teacher models from environment")

    def _load_specialized_model_from_env(
        self,
        env_vars: Dict[str, str],
        prefix: str,
        role: str
    ) -> None:
        """
        Load a specialized model from environment variables.

        Args:
            env_vars: Dictionary of environment variables
            prefix: Environment variable prefix (e.g., "EMBEDDING_MODEL")
            role: Role identifier ("embedding", "judge", or "knowledge_generator")
        """
        base_url = env_vars.get(f"{prefix}_BASE_URL")
        api_key = env_vars.get(f"{prefix}_API_KEY")
        model_name = env_vars.get(f"{prefix}_MODEL_NAME", "default-model")

        if base_url and api_key:
            try:
                if role == "embedding":
                    embedding_dim = int(env_vars.get(f"{prefix}_DIM", "1536"))
                    self.embedding_model = OpenAIEmbeddingProvider(
                        model_name=model_name,
                        embedding_dim=embedding_dim,
                        base_url=base_url,
                        api_key=api_key,
                    )
                    logger.info(f"Loaded embedding model: {model_name}")
                else:
                    provider = OpenAIProvider(
                        model_name=model_name,
                        base_url=base_url,
                        api_key=api_key,
                    )
                    if role == "judge":
                        self.judge_model = provider
                    elif role == "knowledge_generator":
                        self.knowledge_generator_model = provider
                    logger.info(f"Loaded {role} model: {model_name}")
            except Exception as e:
                logger.warning(f"Failed to load {role} model: {e}")

    def _substitute_env(self, value: str, env_vars: Dict[str, str]) -> str:
        """
        Substitute environment variables in a string.

        NOTE: This method is deprecated and kept for backward compatibility only.
        It is recommended to configure all values directly in config/models.yaml
        instead of using environment variable placeholders.
        """
        if not value:
            return value

        # Handle ${VAR_NAME} format
        if "${" in value and "}" in value:
            start = value.find("${")
            end = value.find("}", start)
            var_name = value[start + 2:end]
            env_value = env_vars.get(var_name, "")
            if env_value:
                logger.debug(f"Substituting environment variable: {var_name}")
            return value[:start] + env_value + value[end + 1:]

        return value

    def register_model(
        self,
        name: str,
        provider: LLMProvider,
        role: str = "thinker"
    ) -> None:
        """
        Register a model.

        Args:
            name: Name/identifier for the model
            provider: LLMProvider instance
            role: Role of the model (thinker, judge, etc.)
        """
        self.models[name] = provider

        if role == "thinker" and name not in self.thinkers:
            self.thinkers.append(name)

        logger.debug(f"Registered model: {name} ({provider.provider_type})")

    def get_model(self, name: str) -> Optional[LLMProvider]:
        """
        Get a model by name.

        Args:
            name: Model name

        Returns:
            LLMProvider if found, None otherwise
        """
        return self.models.get(name)

    def get_thinker_models(self) -> List[LLMProvider]:
        """Get all registered teacher models."""
        return [self.models[name] for name in self.thinkers if name in self.models]

    def get_random_thinker(self) -> Optional[LLMProvider]:
        """Get a random teacher model."""
        import random

        if not self.thinkers:
            return None

        name = random.choice(self.thinkers)
        return self.models.get(name)

    def get_operator_model(self, trajectory: Optional[Trajectory] = None) -> Optional[LLMProvider]:
        """
        Get the appropriate operator model for crossover/mutation operations.

        Implements the two-mode model selection strategy:
        - Mode A (auto): Use the original generator model of the trajectory
        - Mode B (global): Use a globally specified operator model

        Args:
            trajectory: Optional trajectory to get the generator model from

        Returns:
            LLMProvider to use for operations
        """
        # Mode B: Use global operator model
        if self.selection_mode == "global" and self.global_operator_model:
            return self.global_operator_model

        # Mode A: Use trajectory's original generator model
        if trajectory and trajectory.source_model:
            model = self.models.get(trajectory.source_model)
            if model:
                return model

        # Fallback: use a random teacher model
        return self.get_random_thinker()

    def set_global_operator(self, model_name: str) -> None:
        """
        Set a global operator model (Mode B).

        Args:
            model_name: Name of the model to use as global operator
        """
        if model_name not in self.models:
            raise ValueError(f"Model not found: {model_name}")

        self.global_operator_model = self.models[model_name]
        self.selection_mode = "global"
        logger.info(f"Set global operator model to: {model_name}")

    def set_auto_mode(self) -> None:
        """Set model selection to auto mode (Mode A)."""
        self.selection_mode = "auto"
        self.global_operator_model = None
        logger.info("Set model selection to auto mode")

    def get_embedding_model(self) -> Optional[EmbeddingProvider]:
        """Get the embedding model."""
        return self.embedding_model

    def get_judge_model(self) -> Optional[LLMProvider]:
        """Get the judge model."""
        return self.judge_model

    def get_knowledge_generator_model(self) -> Optional[LLMProvider]:
        """Get the knowledge generator model."""
        return self.knowledge_generator_model

    def list_models(self) -> List[str]:
        """List all registered model names."""
        return list(self.models.keys())

    def __repr__(self) -> str:
        """String representation of the registry."""
        return (
            f"ModelRegistry("
            f"thinkers={len(self.thinkers)}, "
            f"mode={self.selection_mode}, "
            f"global_op={self.global_operator_model.model_name if self.global_operator_model else None})"
        )
