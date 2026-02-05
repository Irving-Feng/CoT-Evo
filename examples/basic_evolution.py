#!/usr/bin/env python3
"""
Basic example of using the CoT-Evo Evolution Engine.

This script demonstrates how to:
1. Set up the model registry with multiple teacher models
2. Configure the evolution engine
3. Run evolution on a single query
4. Process multiple queries in batch

IMPORTANT: Before running this script, make sure to configure your models in:
    config/models.yaml

All model credentials (base_url, api_key, model_name) should be configured
directly in the YAML file. Environment variables are no longer required.
"""

import asyncio
import logging
from pathlib import Path

from src.models.registry import ModelRegistry
from src.initialization.generators import MultiThinkerGenerator
from src.selection.nslc import NSLCSelector
from src.variation.crossover import ReflectiveCrossover
from src.variation.mutation import ReflectiveMutation
from src.core.fitness import FitnessEvaluator
from src.optimization.engine import EvolutionEngine
from src.optimization.config import EvolutionEngineConfig
from src.knowledge.hybrid import HybridKnowledgeAugmenter
from src.knowledge.generation import KnowledgeGenerator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def setup_components():
    """Set up all components needed for the evolution engine."""

    # 1. Initialize model registry (loads from config/models.yaml)
    logger.info("Initializing model registry from config/models.yaml...")
    registry = ModelRegistry(config_path="config/models.yaml")

    # 2. Create knowledge augmenter (for knowledge-augmented generation)
    logger.info("Setting up knowledge augmenter...")
    # Use the knowledge generator model configured in models.yaml
    knowledge_generator = KnowledgeGenerator(
        model_registry=registry,
        model_name="gpt-4o-knowledge"  # Must match the name in models.yaml
    )
    knowledge_augmenter = HybridKnowledgeAugmenter(
        generator=knowledge_generator,
        knowledge_base=None  # No RAG for this example
    )

    # 3. Create multi-thinker generator
    logger.info("Setting up multi-thinker generator...")
    generator = MultiThinkerGenerator(
        model_registry=registry,
        knowledge_augmenter=knowledge_augmenter,
        dataset_name="ChemCoTDataset"  # For prompt templates (must match config/datasets.yaml)
    )

    # 4. Create NSLC selector
    logger.info("Setting up NSLC selector...")
    selector = NSLCSelector(
        model_registry=registry,
        n_neighbors=5,
        epsilon=0.1
    )

    # 5. Create crossover and mutation operators
    logger.info("Setting up variation operators...")
    crossover = ReflectiveCrossover(model_registry=registry)
    mutation = ReflectiveMutation(model_registry=registry)

    # 6. Create fitness evaluator
    logger.info("Setting up fitness evaluator...")
    fitness_evaluator = FitnessEvaluator(
        lambda_length=0.3,
        lambda_knowledge=0.5
    )

    # Configure evaluator components (these would need to be set up properly)
    # For now, we'll use simple implementations
    # TODO: Set up exact_match_evaluator, length_evaluator, knowledge_evaluator

    return registry, generator, selector, crossover, mutation, fitness_evaluator


async def evolve_single_query():
    """Example: Evolve CoT for a single query."""

    logger.info("=" * 80)
    logger.info("EXAMPLE 1: Single Query Evolution")
    logger.info("=" * 80)

    # Setup components
    registry, generator, selector, crossover, mutation, fitness_evaluator = await setup_components()

    # Configure evolution
    config = EvolutionEngineConfig(
        n_generations=5,
        population_size=10,
        n_vanilla=7,
        n_knowledge_augmented=3,
        convergence_threshold=1.0,
        crossover_prob=0.4,
        mutation_prob=0.6,
        elitism_ratio=0.5,
        save_checkpoints=True,
        log_level="INFO"
    )

    # Create engine
    engine = EvolutionEngine(
        model_registry=registry,
        generator=generator,
        selector=selector,
        crossover=crossover,
        mutation=mutation,
        fitness_evaluator=fitness_evaluator,
        config=config
    )

    # Define query
    query = "What is the molecular weight of glucose (C6H12O6)?"
    ground_truth = "180.16 g/mol"

    logger.info(f"Query: {query}")
    logger.info(f"Ground Truth: {ground_truth}")

    # Run evolution
    checkpoint_path = Path("outputs/checkpoints/glucose_example.pkl")
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    best_trajectory = await engine.evolve(
        query=query,
        ground_truth=ground_truth,
        checkpoint_path=checkpoint_path
    )

    # Display results
    logger.info("\n" + "=" * 80)
    logger.info("EVOLUTION COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Best Fitness Score: {best_trajectory.fitness_score:.4f}")
    logger.info(f"Source Model: {best_trajectory.source_model}")
    logger.info(f"Generation Method: {best_trajectory.generation_method}")
    logger.info(f"\nFinal Answer: {best_trajectory.answer}")
    logger.info(f"\nReasoning:\n{best_trajectory.reasoning}")

    if best_trajectory.knowledge:
        logger.info(f"\nKnowledge Used:\n{best_trajectory.knowledge}")

    # Print generation statistics
    logger.info("\n" + "-" * 80)
    logger.info("GENERATION STATISTICS")
    logger.info("-" * 80)
    for history in engine.history:
        logger.info(
            f"Gen {history.generation}: "
            f"fitness={history.avg_fitness:.3f}/{history.best_fitness:.3f}, "
            f"accuracy={history.accuracy:.2%}, "
            f"pareto_size={history.pareto_front_size}"
        )


async def evolve_batch():
    """Example: Evolve CoT for multiple queries in batch."""

    logger.info("\n\n" + "=" * 80)
    logger.info("EXAMPLE 2: Batch Processing")
    logger.info("=" * 80)

    # Setup components
    registry, generator, selector, crossover, mutation, fitness_evaluator = await setup_components()

    # Configure evolution (lighter for batch processing)
    config = EvolutionEngineConfig(
        n_generations=3,
        population_size=8,
        n_vanilla=5,
        n_knowledge_augmented=3,
        convergence_threshold=0.95,
        save_checkpoints=True,
        log_level="INFO"
    )

    # Create engine
    engine = EvolutionEngine(
        model_registry=registry,
        generator=generator,
        selector=selector,
        crossover=crossover,
        mutation=mutation,
        fitness_evaluator=fitness_evaluator,
        config=config
    )

    # Define sample queries
    samples = [
        {
            "query": "What is the molecular weight of water (H2O)?",
            "ground_truth": "18.015 g/mol"
        },
        {
            "query": "Calculate the molecular weight of ethanol (C2H5OH).",
            "ground_truth": "46.07 g/mol"
        },
        {
            "query": "What is the molecular weight of acetic acid (CH3COOH)?",
            "ground_truth": "60.05 g/mol"
        }
    ]

    logger.info(f"Processing {len(samples)} samples with max concurrency of 2")

    # Run batch evolution
    checkpoint_dir = Path("outputs/checkpoints/batch_example")
    results = await engine.evolve_batch(
        samples=samples,
        max_concurrent=2,
        checkpoint_dir=checkpoint_dir
    )

    # Display results
    logger.info("\n" + "=" * 80)
    logger.info("BATCH EVOLUTION COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Successfully processed: {len(results)}/{len(samples)} samples")

    for i, (sample, result) in enumerate(zip(samples, results)):
        logger.info(f"\nSample {i + 1}:")
        logger.info(f"  Query: {sample['query']}")
        logger.info(f"  Best Fitness: {result.fitness_score:.4f}")
        logger.info(f"  Answer: {result.answer}")
        logger.info(f"  Model: {result.source_model}")


async def main():
    """Main entry point."""

    try:
        # Example 1: Single query evolution
        await evolve_single_query()

        # Example 2: Batch processing (commented out to save API calls)
        # await evolve_batch()

    except Exception as e:
        logger.error(f"Error during evolution: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    # Run async main
    asyncio.run(main())
