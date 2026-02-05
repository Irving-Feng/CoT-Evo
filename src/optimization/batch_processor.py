"""
Optimized batch processor for high-throughput CoT evolution.

This module provides an efficient batch processor that can handle 20-50
concurrent samples with rate limiting, caching, and intelligent resource management.
"""

import asyncio
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass
from datetime import datetime
import time

from .engine import EvolutionEngine
from .config import EvolutionEngineConfig
from ..core.trajectory import Trajectory
from ..utils.rate_limiter import GlobalRateLimiter, RateLimit
from ..utils.embedding_cache import GlobalEmbeddingCache

logger = logging.getLogger(__name__)


@dataclass
class BatchConfig:
    """Configuration for batch processing."""
    max_concurrent: int = 30  # Max concurrent samples
    sample_timeout: float = 300.0  # 5 minutes per sample
    enable_checkpointing: bool = True
    checkpoint_dir: Optional[Path] = None
    progress_update_interval: float = 10.0  # Seconds between progress updates


@dataclass
class BatchResult:
    """Result of batch processing."""
    total_samples: int
    successful: int
    failed: int
    results: List[Trajectory]
    errors: List[Dict[str, Any]]
    total_time: float
    samples_per_minute: float

    def get_success_rate(self) -> float:
        """Get the success rate as a percentage."""
        if self.total_samples == 0:
            return 0.0
        return (self.successful / self.total_samples) * 100.0


class BatchProcessor:
    """
    High-throughput batch processor for CoT evolution.

    Features:
    - Configurable concurrency (20-50 samples)
    - Rate limiting per model to avoid API limits
    - Embedding caching to reduce redundant computation
    - Automatic checkpointing and resumption
    - Progress tracking and statistics
    - Error handling and retry logic
    """

    def __init__(
        self,
        engine: EvolutionEngine,
        config: BatchConfig,
        rate_limits: Optional[Dict[str, RateLimit]] = None
    ):
        """
        Initialize the batch processor.

        Args:
            engine: Evolution engine instance
            config: Batch configuration
            rate_limits: Optional per-model rate limits
        """
        self.engine = engine
        self.config = config

        # Set up rate limits if provided
        if rate_limits:
            rate_limiter = GlobalRateLimiter.get_instance()
            if rate_limiter is None:
                # Initialize with default rate limit
                default_limit = RateLimit(
                    requests_per_minute=120,  # Conservative default
                    max_concurrent=10
                )
                rate_limiter = GlobalRateLimiter.initialize(default_limit)

            for model_name, limit in rate_limits.items():
                rate_limiter.set_rate_limit(model_name, limit)

            logger.info(f"Set rate limits for {len(rate_limits)} models")

        # Initialize embedding cache if not already done
        cache = GlobalEmbeddingCache.get_instance()
        if cache is None and self.config.checkpoint_dir:
            cache_dir = self.config.checkpoint_dir / "cache"
            cache = GlobalEmbeddingCache.initialize(
                cache_dir=cache_dir,
                enable_persistence=True
            )
            logger.info(f"Initialized embedding cache at {cache_dir}")

    async def process_batch(
        self,
        samples: List[Dict[str, str]],
        progress_callback: Optional[Callable] = None
    ) -> BatchResult:
        """
        Process a batch of samples with optimized concurrency.

        Args:
            samples: List of samples with 'query' and 'ground_truth' keys
            progress_callback: Optional callback for progress updates

        Returns:
            BatchResult with statistics and results
        """
        start_time = time.time()
        total_samples = len(samples)

        logger.info(f"Starting batch processing of {total_samples} samples")
        logger.info(f"Max concurrency: {self.config.max_concurrent}")
        logger.info(f"Sample timeout: {self.config.sample_timeout}s")

        # Create semaphore for concurrency control
        semaphore = asyncio.Semaphore(self.config.max_concurrent)

        # Track results
        results = []
        errors = []
        completed_count = [0]  # Use list for mutable reference in nested function

        # Progress tracking task
        progress_task = None
        if progress_callback:
            progress_task = asyncio.create_task(
                self._progress_tracker(
                    total_samples,
                    completed_count,
                    progress_callback
                )
            )

        # Process samples concurrently
        tasks = []
        for idx, sample in enumerate(samples):
            task = self._process_single_sample(
                sample=sample,
                idx=idx,
                semaphore=semaphore,
                completed_count=completed_count
            )
            tasks.append(task)

        # Wait for all tasks to complete
        task_results = await asyncio.gather(*tasks, return_exceptions=True)

        # Cancel progress task
        if progress_task:
            progress_task.cancel()
            try:
                await progress_task
            except asyncio.CancelledError:
                pass

        # Process results
        for i, result in enumerate(task_results):
            if isinstance(result, Exception):
                errors.append({
                    "sample_index": i,
                    "sample": samples[i],
                    "error": str(result)
                })
                logger.error(f"Sample {i} failed: {result}")
            elif isinstance(result, Trajectory):
                results.append(result)
            else:
                logger.warning(f"Unexpected result type for sample {i}: {type(result)}")

        # Calculate statistics
        total_time = time.time() - start_time
        successful = len(results)
        failed = len(errors)
        samples_per_minute = (successful / total_time) * 60.0 if total_time > 0 else 0.0

        batch_result = BatchResult(
            total_samples=total_samples,
            successful=successful,
            failed=failed,
            results=results,
            errors=errors,
            total_time=total_time,
            samples_per_minute=samples_per_minute
        )

        logger.info(f"Batch processing complete:")
        logger.info(f"  Successful: {successful}/{total_samples} ({batch_result.get_success_rate():.1f}%)")
        logger.info(f"  Failed: {failed}/{total_samples}")
        logger.info(f"  Total time: {total_time:.1f}s")
        logger.info(f"  Throughput: {samples_per_minute:.1f} samples/min")

        # Log cache statistics
        cache = GlobalEmbeddingCache.get_instance()
        if cache:
            stats = cache.get_stats()
            logger.info(f"Embedding cache stats: {stats}")

        return batch_result

    async def _process_single_sample(
        self,
        sample: Dict[str, str],
        idx: int,
        semaphore: asyncio.Semaphore,
        completed_count: list
    ) -> Optional[Trajectory]:
        """
        Process a single sample with rate limiting and error handling.

        Args:
            sample: Sample dictionary with 'query' and 'ground_truth'
            idx: Sample index
            semaphore: Semaphore for concurrency control
            completed_count: Mutable counter for completed samples

        Returns:
            Trajectory if successful, None otherwise
        """
        async with semaphore:
            query = sample.get("query", "")
            ground_truth = sample.get("ground_truth", "")
            reference_knowledge = sample.get("reference_knowledge")

            logger.info(f"Processing sample {idx}: {query[:50]}...")

            try:
                # Determine checkpoint path
                checkpoint_path = None
                if self.config.enable_checkpointing and self.config.checkpoint_dir:
                    checkpoint_path = self.config.checkpoint_dir / f"sample_{idx}.pkl"

                # Run evolution
                result = await asyncio.wait_for(
                    self.engine.evolve(
                        query=query,
                        ground_truth=ground_truth,
                        reference_knowledge=reference_knowledge,
                        checkpoint_path=checkpoint_path
                    ),
                    timeout=self.config.sample_timeout
                )

                completed_count[0] += 1
                logger.info(f"Sample {idx} complete (fitness: {result.fitness_score:.3f})")

                return result

            except asyncio.TimeoutError:
                completed_count[0] += 1
                logger.error(f"Sample {idx} timed out after {self.config.sample_timeout}s")
                return None

            except Exception as e:
                completed_count[0] += 1
                logger.error(f"Sample {idx} failed: {e}")
                raise

    async def _progress_tracker(
        self,
        total_samples: int,
        completed_count: list,
        callback: Callable
    ) -> None:
        """
        Track and report progress periodically.

        Args:
            total_samples: Total number of samples to process
            completed_count: Mutable counter for completed samples
            callback: Callback function for progress updates
        """
        last_completed = 0

        while True:
            await asyncio.sleep(self.config.progress_update_interval)

            completed = completed_count[0]
            if completed != last_completed:
                progress = {
                    "total": total_samples,
                    "completed": completed,
                    "remaining": total_samples - completed,
                    "progress_percent": (completed / total_samples) * 100,
                    "timestamp": datetime.now().isoformat()
                }

                try:
                    callback(progress)
                except Exception as e:
                    logger.error(f"Progress callback error: {e}")

                last_completed = completed

            # Check if done
            if completed >= total_samples:
                break

    def save_results(self, batch_result: BatchResult, output_path: Path) -> None:
        """
        Save batch results to a file.

        Args:
            batch_result: Batch processing result
            output_path: Path to save results
        """
        import json

        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Prepare data for JSON serialization
        data = {
            "statistics": {
                "total_samples": batch_result.total_samples,
                "successful": batch_result.successful,
                "failed": batch_result.failed,
                "success_rate": batch_result.get_success_rate(),
                "total_time": batch_result.total_time,
                "samples_per_minute": batch_result.samples_per_minute
            },
            "results": [
                {
                    "query": r.query,
                    "answer": r.answer,
                    "reasoning": r.reasoning,
                    "fitness_score": r.fitness_score,
                    "source_model": r.source_model,
                    "generation_method": r.generation_method
                }
                for r in batch_result.results
            ],
            "errors": batch_result.errors
        }

        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Results saved to {output_path}")


def create_batch_processor(
    engine: EvolutionEngine,
    max_concurrent: int = 30,
    checkpoint_dir: Optional[Path] = None,
    rate_limits: Optional[Dict[str, RateLimit]] = None
) -> BatchProcessor:
    """
    Convenience function to create a batch processor.

    Args:
        engine: Evolution engine instance
        max_concurrent: Maximum concurrent samples (20-50 recommended)
        checkpoint_dir: Directory for checkpoints
        rate_limits: Optional per-model rate limits

    Returns:
        Configured BatchProcessor instance
    """
    config = BatchConfig(
        max_concurrent=max_concurrent,
        enable_checkpointing=checkpoint_dir is not None,
        checkpoint_dir=checkpoint_dir
    )

    return BatchProcessor(
        engine=engine,
        config=config,
        rate_limits=rate_limits
    )
