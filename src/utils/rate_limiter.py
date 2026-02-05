"""
Rate limiter for LLM API calls.

This module implements rate limiting to prevent hitting API rate limits
and optimize throughput by managing concurrent requests efficiently.
"""

import asyncio
import time
import logging
from typing import Dict, Optional, Callable
from collections import deque
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class RateLimit:
    """Rate limit configuration for an API."""
    requests_per_minute: int
    max_concurrent: int


class RateLimiter:
    """
    Token bucket rate limiter for API calls.

    Implements the token bucket algorithm to rate limit API calls
    while allowing bursts up to a maximum capacity.

    Attributes:
        rate_limits: Dictionary mapping model names to their rate limits
        buckets: Dictionary of token buckets for each model
        semaphores: Dictionary of semaphores for concurrent request control
    """

    def __init__(self, default_rate_limit: RateLimit):
        """
        Initialize the rate limiter.

        Args:
            default_rate_limit: Default rate limit for all models
        """
        self.default_rate_limit = default_rate_limit
        self.rate_limits: Dict[str, RateLimit] = {}
        self.buckets: Dict[str, TokenBucket] = {}
        self.semaphores: Dict[str, asyncio.Semaphore] = {}

        logger.info(f"Rate limiter initialized with default: {default_rate_limit.requests_per_minute} req/min, {default_rate_limit.max_concurrent} concurrent")

    def set_rate_limit(self, model_name: str, rate_limit: RateLimit) -> None:
        """
        Set a custom rate limit for a specific model.

        Args:
            model_name: Name of the model
            rate_limit: Rate limit configuration
        """
        self.rate_limits[model_name] = rate_limit

        # Create or update token bucket
        if model_name not in self.buckets:
            self.buckets[model_name] = TokenBucket(
                rate=rate_limit.requests_per_minute / 60.0,  # Convert to per-second
                capacity=rate_limit.requests_per_minute  # Allow burst up to 1 minute worth
            )

        # Create or update semaphore
        self.semaphores[model_name] = asyncio.Semaphore(rate_limit.max_concurrent)

        logger.info(f"Rate limit set for {model_name}: {rate_limit.requests_per_minute} req/min, {rate_limit.max_concurrent} concurrent")

    async def acquire(self, model_name: str, timeout: float = 60.0) -> bool:
        """
        Acquire permission to make an API call to the given model.

        This will:
        1. Wait for a token from the token bucket (rate limiting)
        2. Acquire a semaphore slot (concurrent request limiting)

        Args:
            model_name: Name of the model to call
            timeout: Maximum time to wait for permission (seconds)

        Returns:
            True if permission was acquired, False if timeout occurred
        """
        # Get or create rate limit for this model
        rate_limit = self.rate_limits.get(model_name, self.default_rate_limit)

        # Initialize bucket and semaphore if needed
        if model_name not in self.buckets:
            self.set_rate_limit(model_name, rate_limit)

        bucket = self.buckets[model_name]
        semaphore = self.semaphores[model_name]

        # Try to acquire a token
        start_time = time.time()
        while True:
            if bucket.try_consume():
                break  # Got a token

            # Wait before retrying
            elapsed = time.time() - start_time
            if elapsed >= timeout:
                logger.warning(f"Rate limiter timeout for {model_name} after {elapsed:.1f}s")
                return False

            # Calculate wait time
            wait_time = bucket.time_until_next_token()
            wait_time = min(wait_time, timeout - elapsed)
            await asyncio.sleep(wait_time)

        # Acquire semaphore for concurrent request control
        await semaphore.acquire()

        logger.debug(f"Rate limiter: acquired permission for {model_name}")
        return True

    def release(self, model_name: str) -> None:
        """
        Release permission after API call completes.

        Args:
            model_name: Name of the model that was called
        """
        if model_name in self.semaphores:
            self.semaphores[model_name].release()
            logger.debug(f"Rate limiter: released permission for {model_name}")

    async def call_with_limit(
        self,
        model_name: str,
        func: Callable,
        *args,
        timeout: float = 60.0,
        **kwargs
    ):
        """
        Execute a function with rate limiting.

        Args:
            model_name: Name of the model being called
            func: The async function to execute
            *args: Positional arguments for the function
            timeout: Maximum time to wait for rate limiter
            **kwargs: Keyword arguments for the function

        Returns:
            The return value of the function

        Raises:
            TimeoutError: If rate limiter timeout occurs
        """
        # Acquire permission
        if not await self.acquire(model_name, timeout=timeout):
            raise TimeoutError(f"Rate limiter timeout for {model_name}")

        try:
            # Execute the function
            result = await func(*args, **kwargs)
            return result
        finally:
            # Always release permission
            self.release(model_name)

    def get_stats(self) -> Dict[str, any]:
        """
        Get statistics about rate limiter state.

        Returns:
            Dictionary with rate limiter statistics
        """
        stats = {
            "models": len(self.buckets),
            "model_stats": {}
        }

        for model_name, bucket in self.buckets.items():
            semaphore = self.semaphores[model_name]
            stats["model_stats"][model_name] = {
                "tokens_available": bucket.tokens,
                "capacity": bucket.capacity,
                "semaphore_available": semaphore._value,
                "semaphore_max": semaphore._value + semaphore._lock._count() if hasattr(semaphore, '_lock') else 'N/A'
            }

        return stats


class TokenBucket:
    """
    Token bucket implementation for rate limiting.

    The token bucket algorithm allows for bursts of requests
    while maintaining a long-term rate limit.

    Attributes:
        rate: Rate of token generation (tokens per second)
        capacity: Maximum number of tokens in the bucket
        tokens: Current number of tokens in the bucket
        last_update: Timestamp of last token generation
    """

    def __init__(self, rate: float, capacity: int):
        """
        Initialize the token bucket.

        Args:
            rate: Token generation rate (tokens per second)
            capacity: Maximum bucket capacity (tokens)
        """
        self.rate = rate
        self.capacity = capacity
        self.tokens = float(capacity)  # Start with full bucket
        self.last_update = time.time()

    def _refill(self) -> None:
        """Refill tokens based on elapsed time."""
        now = time.time()
        elapsed = now - self.last_update

        # Add new tokens based on elapsed time
        new_tokens = elapsed * self.rate
        self.tokens = min(self.capacity, self.tokens + new_tokens)
        self.last_update = now

    def try_consume(self, tokens: int = 1) -> bool:
        """
        Try to consume tokens from the bucket.

        Args:
            tokens: Number of tokens to consume

        Returns:
            True if tokens were consumed, False if not enough tokens
        """
        self._refill()

        if self.tokens >= tokens:
            self.tokens -= tokens
            return True
        return False

    def time_until_next_token(self) -> float:
        """
        Calculate time until next token is available.

        Returns:
            Time in seconds until next token (0 if tokens available now)
        """
        self._refill()

        if self.tokens >= 1:
            return 0.0

        # Calculate time needed to generate 1 token
        tokens_needed = 1 - self.tokens
        return tokens_needed / self.rate


class ConnectionPool:
    """
    Connection pool for managing multiple LLM API connections.

    This pool helps manage resources and avoid creating too many
    simultaneous connections.

    Attributes:
        max_connections: Maximum number of concurrent connections
        active_connections: Current number of active connections
        lock: Async lock for thread-safe operations
    """

    def __init__(self, max_connections: int = 100):
        """
        Initialize the connection pool.

        Args:
            max_connections: Maximum concurrent connections
        """
        self.max_connections = max_connections
        self.active_connections = 0
        self.lock = asyncio.Lock()

        logger.info(f"Connection pool initialized with max {max_connections} connections")

    async def acquire(self, timeout: float = 30.0) -> bool:
        """
        Acquire a connection from the pool.

        Args:
            timeout: Maximum time to wait (seconds)

        Returns:
            True if connection was acquired, False if timeout
        """
        start_time = time.time()

        while True:
            async with self.lock:
                if self.active_connections < self.max_connections:
                    self.active_connections += 1
                    logger.debug(f"Connection acquired: {self.active_connections}/{self.max_connections} active")
                    return True

            # Wait before retrying
            elapsed = time.time() - start_time
            if elapsed >= timeout:
                logger.warning(f"Connection pool timeout after {elapsed:.1f}s")
                return False

            await asyncio.sleep(0.1)  # Small delay before retry

    def release(self) -> None:
        """Release a connection back to the pool."""
        # Note: We're not using async here because we want immediate release
        # In production, this should be made thread-safe
        self.active_connections = max(0, self.active_connections - 1)
        logger.debug(f"Connection released: {self.active_connections}/{self.max_connections} active")

    def get_stats(self) -> Dict[str, any]:
        """Get connection pool statistics."""
        return {
            "active_connections": self.active_connections,
            "max_connections": self.max_connections,
            "available_connections": self.max_connections - self.active_connections
        }


class GlobalRateLimiter:
    """
    Global singleton instance of the rate limiter.

    This ensures consistent rate limiting across all components.
    """

    _instance: Optional[RateLimiter] = None
    _connection_pool: Optional[ConnectionPool] = None

    @classmethod
    def initialize(
        cls,
        default_rate_limit: RateLimit,
        max_connections: int = 100
    ) -> RateLimiter:
        """
        Initialize the global rate limiter.

        Args:
            default_rate_limit: Default rate limit for all models
            max_connections: Maximum concurrent connections

        Returns:
            The global RateLimiter instance
        """
        if cls._instance is None:
            cls._instance = RateLimiter(default_rate_limit)
            cls._connection_pool = ConnectionPool(max_connections)
            logger.info("Global rate limiter initialized")

        return cls._instance

    @classmethod
    def get_instance(cls) -> Optional[RateLimiter]:
        """Get the global rate limiter instance."""
        return cls._instance

    @classmethod
    def get_connection_pool(cls) -> Optional[ConnectionPool]:
        """Get the global connection pool."""
        return cls._connection_pool

    @classmethod
    def reset(cls) -> None:
        """Reset the global rate limiter."""
        cls._instance = None
        cls._connection_pool = None
        logger.info("Global rate limiter reset")
