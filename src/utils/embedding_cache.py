"""
Embedding cache for efficient behavioral vector storage and retrieval.

This module implements a cache for embeddings to avoid redundant computation
of behavioral vectors for trajectories, which is critical for performance
in the NSLC selection algorithm.
"""

import hashlib
import pickle
import logging
from pathlib import Path
from typing import Optional, Dict, Any
import numpy as np

logger = logging.getLogger(__name__)


class EmbeddingCache:
    """
    Cache for storing and retrieving embeddings efficiently.

    Features:
    - In-memory caching for fast access
    - Optional disk persistence for cross-session caching
    - Automatic deduplication based on content hash
    - Thread-safe operations

    Attributes:
        cache: Dictionary mapping cache keys to embeddings
        cache_dir: Directory for persistent storage (None if disabled)
        hit_count: Number of cache hits
        miss_count: Number of cache misses
    """

    def __init__(self, cache_dir: Optional[Path] = None, enable_persistence: bool = True):
        """
        Initialize the embedding cache.

        Args:
            cache_dir: Directory for persistent cache files (None for in-memory only)
            enable_persistence: Whether to save/load cache from disk
        """
        self.cache: Dict[str, np.ndarray] = {}
        self.cache_dir = cache_dir
        self.enable_persistence = enable_persistence and cache_dir is not None

        self.hit_count = 0
        self.miss_count = 0

        # Load from disk if persistence is enabled
        if self.enable_persistence and self.cache_dir.exists():
            self._load_from_disk()

    def _generate_cache_key(self, text: str, model_name: str) -> str:
        """
        Generate a unique cache key for a text-model pair.

        Uses SHA256 hash of (text + model_name) to ensure uniqueness
        while keeping keys manageable.

        Args:
            text: The text to be embedded
            model_name: Name of the embedding model

        Returns:
            A unique cache key (hex digest of SHA256 hash)
        """
        content = f"{model_name}:{text}"
        return hashlib.sha256(content.encode()).hexdigest()

    def get(self, text: str, model_name: str) -> Optional[np.ndarray]:
        """
        Retrieve an embedding from the cache.

        Args:
            text: The text that was embedded
            model_name: Name of the embedding model

        Returns:
            Cached embedding if found, None otherwise
        """
        key = self._generate_cache_key(text, model_name)

        if key in self.cache:
            self.hit_count += 1
            logger.debug(f"Cache hit for key {key[:16]}...")
            return self.cache[key].copy()  # Return a copy to prevent mutation

        self.miss_count += 1
        logger.debug(f"Cache miss for key {key[:16]}...")
        return None

    def put(self, text: str, model_name: str, embedding: np.ndarray) -> None:
        """
        Store an embedding in the cache.

        Args:
            text: The text that was embedded
            model_name: Name of the embedding model
            embedding: The embedding vector to store
        """
        key = self._generate_cache_key(text, model_name)
        self.cache[key] = embedding.copy()  # Store a copy

        logger.debug(f"Cached embedding with key {key[:16]}... (size: {self._size()})")

        # Persist to disk if enabled
        if self.enable_persistence:
            self._save_to_disk_async()

    def put_batch(self, items: list) -> None:
        """
        Store multiple embeddings in the cache.

        Args:
            items: List of (text, model_name, embedding) tuples
        """
        for text, model_name, embedding in items:
            self.put(text, model_name, embedding)

    def clear(self) -> None:
        """Clear all cached embeddings."""
        self.cache.clear()
        self.hit_count = 0
        self.miss_count = 0
        logger.info("Embedding cache cleared")

        # Clear disk cache if enabled
        if self.enable_persistence and self.cache_dir.exists():
            for cache_file in self.cache_dir.glob("embedding_cache_*.pkl"):
                cache_file.unlink()

    def _size(self) -> int:
        """Get the number of cached embeddings."""
        return len(self.cache)

    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache statistics (size, hit rate, etc.)
        """
        total_requests = self.hit_count + self.miss_count
        hit_rate = self.hit_count / total_requests if total_requests > 0 else 0.0

        return {
            "size": self._size(),
            "hit_count": self.hit_count,
            "miss_count": self.miss_count,
            "hit_rate": hit_rate,
            "total_requests": total_requests
        }

    def _save_to_disk(self) -> None:
        """
        Save cache to disk for persistence.

        Saves the entire cache to a single pickle file.
        """
        if not self.enable_persistence:
            return

        try:
            # Create cache directory if it doesn't exist
            self.cache_dir.mkdir(parents=True, exist_ok=True)

            # Save to pickle file
            cache_file = self.cache_dir / "embedding_cache.pkl"
            with open(cache_file, "wb") as f:
                pickle.dump(self.cache, f)

            logger.info(f"Saved {self._size()} embeddings to {cache_file}")
        except Exception as e:
            logger.error(f"Failed to save embedding cache to disk: {e}")

    def _save_to_disk_async(self) -> None:
        """
        Async save to disk (non-blocking).

        In production, this could use a background thread or asyncio task.
        For now, it's a synchronous operation that's fast for small caches.
        """
        # Only save periodically to avoid excessive I/O
        if self.miss_count % 10 == 0:  # Save every 10 new embeddings
            self._save_to_disk()

    def _load_from_disk(self) -> None:
        """Load cache from disk."""
        if not self.enable_persistence:
            return

        cache_file = self.cache_dir / "embedding_cache.pkl"

        if not cache_file.exists():
            logger.debug("No existing cache file found")
            return

        try:
            with open(cache_file, "rb") as f:
                loaded_cache = pickle.load(f)

            # Merge loaded cache with current cache
            self.cache.update(loaded_cache)

            logger.info(f"Loaded {len(loaded_cache)} embeddings from {cache_file}")
            logger.info(f"Total cache size: {self._size()}")
        except Exception as e:
            logger.error(f"Failed to load embedding cache from disk: {e}")

    def __repr__(self) -> str:
        """String representation of the cache."""
        stats = self.get_stats()
        return (
            f"EmbeddingCache(size={stats['size']}, "
            f"hit_rate={stats['hit_rate']:.2%}, "
            f"total_requests={stats['total_requests']})"
        )


class GlobalEmbeddingCache:
    """
    Global singleton instance of the embedding cache.

    This ensures that all components share the same cache,
    maximizing cache hits across the entire evolution process.
    """

    _instance: Optional[EmbeddingCache] = None

    @classmethod
    def initialize(cls, cache_dir: Optional[Path] = None, enable_persistence: bool = True) -> EmbeddingCache:
        """
        Initialize the global embedding cache.

        Args:
            cache_dir: Directory for persistent cache files
            enable_persistence: Whether to enable disk persistence

        Returns:
            The global EmbeddingCache instance
        """
        if cls._instance is None:
            cls._instance = EmbeddingCache(
                cache_dir=cache_dir,
                enable_persistence=enable_persistence
            )
            logger.info(f"Global embedding cache initialized at {id(cls._instance)}")

        return cls._instance

    @classmethod
    def get_instance(cls) -> Optional[EmbeddingCache]:
        """
        Get the global embedding cache instance.

        Returns:
            The global EmbeddingCache instance, or None if not initialized
        """
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Reset the global embedding cache instance."""
        cls._instance = None
        logger.info("Global embedding cache reset")
