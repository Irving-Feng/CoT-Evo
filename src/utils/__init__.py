"""
Utility modules for CoT-Evo framework.

This package provides utility classes for caching, rate limiting,
and other cross-cutting concerns.
"""

from .embedding_cache import (
    EmbeddingCache,
    GlobalEmbeddingCache
)

from .rate_limiter import (
    RateLimiter,
    RateLimit,
    TokenBucket,
    ConnectionPool,
    GlobalRateLimiter
)

__all__ = [
    # Embedding cache
    "EmbeddingCache",
    "GlobalEmbeddingCache",

    # Rate limiting
    "RateLimiter",
    "RateLimit",
    "TokenBucket",
    "ConnectionPool",
    "GlobalRateLimiter",
]
