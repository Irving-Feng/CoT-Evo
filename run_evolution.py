#!/usr/bin/env python3
"""
CoT-Evo: Evolutionary Distillation of Chain-of-Thought

Main entry point for running CoT-Evo evolution on a dataset.

This script loads a dataset from datasets.yaml, initializes the evolution engine,
and runs evolutionary optimization on each sample.

Usage:
    python run_evolution.py --dataset ChemCoTDataset --max-samples 10
    python run_evolution.py --dataset BioProBench --max-samples -1 --config config/evolution.yaml
"""

import asyncio
import argparse
import logging
import sys
import yaml
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.data.dataset_loader import DatasetLoader
from src.optimization.engine import EvolutionEngine
from src.optimization.config import EvolutionEngineConfig
from src.models.registry import ModelRegistry
from src.initialization.generators import MultiThinkerGenerator
from src.selection.nslc import NSLCSelector
from src.variation.crossover import ReflectiveCrossover
from src.variation.mutation import ReflectiveMutation
from src.core.fitness import (
    FitnessEvaluator,
    ExactMatchEvaluator,
    LengthEvaluator,
    KnowledgeJudgeEvaluator
)
from src.knowledge.generation import KnowledgeGenerator
from src.knowledge.hybrid import HybridKnowledgeAugmenter
from src.utils.answer_extractor import extract_cot_and_answer, clean_answer

# Helper function to calculate reasoning length (matching test_complete_evolution.py)
def calculate_reasoning_length(reasoning: str) -> int:
    """
    Calculate reasoning length in words, excluding <|think|> and <|answer|> markers.

    Args:
        reasoning: Full reasoning string potentially containing markers

    Returns:
        Word count of pure reasoning content
    """
    if not reasoning:
        return 0

    # Extract content between <|think|> and <|answer|>
    if "<|think|>" in reasoning:
        start = reasoning.find("<|think|>") + 9
        end = reasoning.find("<|answer|>")
        if end != -1:
            reasoning = reasoning[start:end].strip()

    # Return word count (split by whitespace)
    return len(reasoning.split())

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def load_default_config(config_path: str = "config/default.yaml") -> Dict[str, Any]:
    """
    Load default configuration from default.yaml.

    Args:
        config_path: Path to default.yaml

    Returns:
        Configuration dictionary
    """
    config_file = Path(config_path)
    if not config_file.exists():
        logger.warning(f"Default config not found: {config_path}, using built-in defaults")
        return {
            'parallel': {'batch_size': 20, 'max_concurrent_requests': 100}
        }

    with open(config_file) as f:
        config = yaml.safe_load(f)

    logger.info(f"Loaded default config from: {config_path}")
    return config


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run CoT-Evo evolutionary distillation on a dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run on ChemCoT dataset with 10 samples
  python run_evolution.py --dataset ChemCoTDataset --max-samples 10

  # Run on all BioProBench samples
  python run_evolution.py --dataset BioProBench --max-samples -1

  # Use custom evolution config
  python run_evolution.py --dataset ChemCoTDataset --config config/my_evolution.yaml
        """
    )

    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name from config/datasets.yaml (e.g., ChemCoTDataset, BioProBench, generic)"
    )

    parser.add_argument(
        "--max-samples",
        type=int,
        default=-1,
        help="Maximum number of samples to process (-1 for all samples, N for max(N, data_len))"
    )

    parser.add_argument(
        "--config",
        type=str,
        default="config/evolution.yaml",
        help="Path to evolution engine config YAML (default: config/evolution.yaml)"
    )

    parser.add_argument(
        "--default-config",
        type=str,
        default="config/default.yaml",
        help="Path to default config YAML (default: config/default.yaml)"
    )

    parser.add_argument(
        "--models",
        type=str,
        default="config/models.yaml",
        help="Path to models config YAML (default: config/models.yaml)"
    )

    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "test"],
        help="Which dataset split to use (default: train)"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: outputs/{dataset}/)"
    )

    parser.add_argument(
        "--use-knowledge-judge",
        action="store_true",
        default=False,
        help="Enable LLM-as-a-Judge for knowledge evaluation (slower but more accurate)"
    )

    parser.add_argument(
        "--lambda-length",
        type=float,
        default=0.3,
        help="Weight for length score (λ1, default: 0.3)"
    )

    parser.add_argument(
        "--lambda-knowledge",
        type=float,
        default=0.5,
        help="Weight for knowledge score (λ2, default: 0.5)"
    )

    return parser.parse_args()


async def run_evolution_on_dataset(
    dataset_name: str,
    max_samples: int,
    config: EvolutionEngineConfig,
    registry: ModelRegistry,
    dataset_loader: DatasetLoader,
    use_knowledge_judge: bool = False,
    lambda_length: float = 0.3,
    lambda_knowledge: float = 0.5,
    split: str = "train",
    output_dir: Path = None,
    batch_size: int = 20
) -> List[Dict[str, Any]]:
    """
    Run evolution on all samples in a dataset.

    Args:
        dataset_name: Name of the dataset
        max_samples: Maximum samples to process (-1 for all)
        config: Evolution engine configuration
        registry: Model registry
        dataset_loader: Dataset loader instance
        use_knowledge_judge: Whether to use LLM judge
        lambda_length: Weight for length score
        lambda_knowledge: Weight for knowledge score
        split: Which split to use
        output_dir: Output directory path
        batch_size: Maximum concurrent samples to process

    Returns:
        List of result dictionaries
    """
    # Load dataset data
    logger.info(f"Loading dataset: {dataset_name}")
    data = dataset_loader.load_dataset_data(dataset_name, max_samples=max_samples, split=split)

    if not data:
        logger.error(f"No data loaded for {dataset_name}")
        return []

    # Get dataset config
    dataset_config = dataset_loader.get_dataset_config(dataset_name)

    # Set up output directory
    if output_dir is None:
        output_dir = Path("outputs") / dataset_name / datetime.now().strftime("%Y%m%d_%H%M%S")
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

    # Initialize fitness evaluator components
    exact_match_evaluator = ExactMatchEvaluator(strict=True)

    # Get length percentiles from dataset config
    length_percentiles = dataset_config.length_percentiles
    length_evaluator = LengthEvaluator(
        lower_percentile=length_percentiles['lower'],
        upper_percentile=length_percentiles['upper']
    )

    # Knowledge judge (optional)
    knowledge_evaluator = None
    if use_knowledge_judge:
        judge_model = registry.get_judge_model()
        knowledge_evaluator = KnowledgeJudgeEvaluator(judge_model)
        logger.info("LLM-as-a-Judge ENABLED for knowledge evaluation")
    else:
        logger.info("LLM-as-a-Judge DISABLED (using binary knowledge check)")

    # Create main fitness evaluator
    fitness_evaluator = FitnessEvaluator(
        lambda_length=lambda_length,
        lambda_knowledge=lambda_knowledge,
        use_exact_match=True,
        use_length_score=True,
        use_knowledge_score=True
    )

    fitness_evaluator.set_exact_match_evaluator(exact_match_evaluator)
    fitness_evaluator.set_length_evaluator(length_evaluator)
    if knowledge_evaluator:
        fitness_evaluator.set_knowledge_evaluator(knowledge_evaluator)

    # Initialize knowledge augmentation components
    # Use knowledge generator model for knowledge generation, fallback to judge model
    kg_model = registry.get_knowledge_generator_model()
    if kg_model is None:
        kg_model = registry.get_judge_model()

    if kg_model:
        knowledge_generator = KnowledgeGenerator(model=kg_model)
        knowledge_augmenter = HybridKnowledgeAugmenter(generator=knowledge_generator)
        logger.info("Knowledge augmentation enabled (using LLM-based generation)")
    else:
        # Create a simple knowledge augmenter without generation (will skip knowledge augmentation)
        knowledge_augmenter = HybridKnowledgeAugmenter(generator=None)
        logger.warning("No knowledge generator model available. Knowledge augmentation will be skipped.")

    # Note: Evolution components (generator, selector, crossover, mutation, engine)
    # will be created separately for each sample to avoid state conflicts in parallel execution

    # Define async function to process a single sample
    async def process_single_sample(idx: int, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single sample through evolution."""
        sample_id = sample.get('id', f"sample_{idx}")

        logger.info(f"[Sample {idx}] Starting: {sample_id}")

        try:
            # Extract query and answer
            query = sample['query']
            ground_truth = sample['answer']

            # Reference knowledge (if available)
            reference_knowledge = sample.get('knowledge', None)

            # Create a new engine instance for this sample (to avoid shared state issues)
            sample_generator = MultiThinkerGenerator(
                model_registry=registry,
                knowledge_augmenter=knowledge_augmenter,
                dataset_name=dataset_name
            )

            sample_selector = NSLCSelector(
                model_registry=registry,
                n_neighbors=config.k_neighbors,
                epsilon=config.epsilon
            )

            sample_crossover = ReflectiveCrossover(model_registry=registry)
            sample_mutation = ReflectiveMutation(model_registry=registry)

            sample_engine = EvolutionEngine(
                model_registry=registry,
                generator=sample_generator,
                selector=sample_selector,
                crossover=sample_crossover,
                mutation=sample_mutation,
                fitness_evaluator=fitness_evaluator,
                config=config,
                output_dir=output_dir,  # Enable detailed logging
                sample_id=sample_id     # Enable detailed logging
            )

            # Run evolution
            best_trajectory = await sample_engine.evolve(
                query=query,
                ground_truth=ground_truth,
                reference_knowledge=reference_knowledge
            )

            # Extract answer using clean_answer (matching test_complete_evolution.py)
            # best_trajectory.answer is already extracted, just need to clean it
            if best_trajectory and best_trajectory.answer:
                clean_ans = clean_answer(best_trajectory.answer)
                extracted_reasoning = best_trajectory.reasoning if best_trajectory.reasoning else ""
            else:
                clean_ans = ""
                extracted_reasoning = ""

            # Save individual result (matching test_complete_evolution.py format)
            sample_result_file = output_dir / f"sample_{idx}_{sample_id[:8]}.json"
            import json
            with open(sample_result_file, 'w') as f:
                json.dump({
                    "sample_id": sample_id,
                    "task": sample.get('task', ''),
                    "query": query[:200] + "..." if len(query) > 200 else query,
                    "ground_truth": ground_truth,
                    "best_answer": clean_ans,
                    "best_fitness": float(best_trajectory.fitness_score) if best_trajectory and best_trajectory.fitness_score is not None else None,
                    "best_reasoning_length": len(best_trajectory.reasoning.split()) if best_trajectory and best_trajectory.reasoning else 0,
                    "source_model": best_trajectory.source_model if best_trajectory else "",
                    "generation_method": best_trajectory.generation_method if best_trajectory else "",
                    "full_reasoning": best_trajectory.reasoning if best_trajectory else "",
                    "full_answer": best_trajectory.answer if best_trajectory else "",
                    "generations": sample_engine.generation,
                    "timestamp": datetime.now().isoformat()
                }, f, indent=2)

            logger.info(f"[Sample {idx}] ✅ Complete - Fitness: {best_trajectory.fitness_score if best_trajectory else 0:.3f}")

            return {
                "sample_id": sample_id,
                "task": sample.get('task', ''),
                "query": query[:200] + "..." if len(query) > 200 else query,
                "ground_truth": ground_truth,
                "best_answer": clean_ans,
                "best_fitness": float(best_trajectory.fitness_score) if best_trajectory and best_trajectory.fitness_score is not None else None,
                "best_reasoning_length": calculate_reasoning_length(best_trajectory.reasoning) if best_trajectory and best_trajectory.reasoning else 0,
                "source_model": best_trajectory.source_model if best_trajectory else "",
                "generation_method": best_trajectory.generation_method if best_trajectory else "",
                "full_reasoning": best_trajectory.reasoning if best_trajectory else "",
                "full_answer": best_trajectory.answer if best_trajectory else "",
                "generations": sample_engine.generation,
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"[Sample {idx}] ❌ Failed: {e}")
            return {
                "sample_id": sample_id,
                "query": sample.get('query', ''),
                "ground_truth": sample.get('answer', ''),
                "best_answer": None,
                "best_reasoning": None,
                "best_fitness": None,
                "generations": 0,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

    # Run evolution on all samples IN PARALLEL with batch_size limit
    total_samples = len(data)

    logger.info(f"\n{'='*70}")
    logger.info(f"Starting PARALLEL evolution on {total_samples} samples (batch_size={batch_size})")
    logger.info(f"{'='*70}\n")

    # Create a semaphore to limit concurrent tasks
    semaphore = asyncio.Semaphore(batch_size)

    async def process_with_semaphore(idx: int, sample: Dict[str, Any]):
        """Process sample with semaphore-controlled concurrency."""
        async with semaphore:
            return await process_single_sample(idx, sample)

    # Create tasks for all samples
    tasks = [
        process_with_semaphore(idx, sample)
        for idx, sample in enumerate(data, 1)
    ]

    # Run all tasks in parallel using asyncio.gather (controlled by semaphore)
    logger.info(f"Launching {total_samples} evolution tasks with max {batch_size} concurrent...")
    results_list = await asyncio.gather(*tasks, return_exceptions=True)

    # Filter out exceptions and collect results
    results = []
    for i, result in enumerate(results_list, 1):
        if isinstance(result, Exception):
            logger.error(f"[Sample {i}] Exception: {result}")
        elif result is not None:
            results.append(result)

    # Save results
    results_file = output_dir / "results.json"
    import json
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"\n{'='*70}")
    logger.info(f"Results saved to: {results_file}")
    logger.info(f"{'='*70}\n")

    # Print summary
    successful = [r for r in results if r.get('best_fitness') is not None]
    failed = len(results) - len(successful)

    if successful:
        avg_fitness = sum(r['best_fitness'] for r in successful) / len(successful)
        max_fitness = max(r['best_fitness'] for r in successful)

        logger.info(f"Summary:")
        logger.info(f"  Total samples: {len(results)}")
        logger.info(f"  Successful: {len(successful)}")
        logger.info(f"  Failed: {failed}")
        logger.info(f"  Average fitness: {avg_fitness:.3f}")
        logger.info(f"  Max fitness: {max_fitness:.3f}")

    return results


async def main():
    """Main entry point."""
    args = parse_arguments()

    # Load default config
    default_config = load_default_config(args.default_config)
    batch_size = default_config.get('parallel', {}).get('batch_size', 20)

    # Extract fitness parameters from default.yaml (with fallback to CLI args)
    fitness_config = default_config.get('fitness', {})
    lambda_length = args.lambda_length if args.lambda_length != 0.3 else fitness_config.get('lambda_length', 0.3)
    lambda_knowledge = args.lambda_knowledge if args.lambda_knowledge != 0.5 else fitness_config.get('lambda_knowledge', 0.5)

    logger.info("="*70)
    logger.info("CoT-Evo: Evolutionary Distillation of Chain-of-Thought")
    logger.info("="*70)
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Max samples: {args.max_samples}")
    logger.info(f"Split: {args.split}")
    logger.info(f"Config: {args.config}")
    logger.info(f"Default config: {args.default_config}")
    logger.info(f"Models: {args.models}")
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Knowledge judge: {'Enabled' if args.use_knowledge_judge else 'Disabled'}")
    logger.info(f"λ1 (length): {lambda_length}")
    logger.info(f"λ2 (knowledge): {lambda_knowledge}")
    logger.info("="*70 + "\n")

    try:
        # Load dataset loader
        dataset_loader = DatasetLoader()

        # Check if dataset exists
        available_datasets = dataset_loader.list_datasets()
        if args.dataset not in available_datasets:
            logger.error(f"Dataset '{args.dataset}' not found.")
            logger.error(f"Available datasets: {', '.join(available_datasets)}")
            sys.exit(1)

        # Load evolution config
        config_path = Path(args.config)
        if config_path.exists():
            config = EvolutionEngineConfig.from_yaml(config_path)
            logger.info(f"Loaded evolution config from: {config_path}")
        else:
            logger.warning(f"Config file not found: {config_path}, using defaults")
            config = EvolutionEngineConfig()

        # Load model registry
        registry = ModelRegistry(config_path=args.models)
        logger.info(f"Loaded {len(registry.thinkers)} teacher models")

        # Run evolution
        results = await run_evolution_on_dataset(
            dataset_name=args.dataset,
            max_samples=args.max_samples,
            config=config,
            registry=registry,
            batch_size=batch_size,
            dataset_loader=dataset_loader,
            use_knowledge_judge=args.use_knowledge_judge,
            lambda_length=lambda_length,
            lambda_knowledge=lambda_knowledge,
            split=args.split,
            output_dir=Path(args.output_dir) if args.output_dir else None
        )

        logger.info("\n✅ All done!")
        sys.exit(0)

    except Exception as e:
        logger.error(f"\n❌ Error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
