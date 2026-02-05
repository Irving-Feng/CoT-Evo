# CoT-Evo Examples

This directory contains practical examples demonstrating how to use the CoT-Evo framework.

## 📁 Examples

### 1. `basic_evolution.py`

**Purpose**: Demonstrates the basic usage of the EvolutionEngine for both single-query and batch processing.

**What it shows**:
- Setting up the model registry with multiple teacher models
- Configuring all components (generator, selector, crossover, mutation, fitness evaluator)
- Running evolution on a single chemistry query
- Processing multiple queries in parallel with concurrency control
- Saving and loading checkpoints
- Displaying generation statistics and results

**How to run**:
```bash
cd /path/to/CoT-Evo
python examples/basic_evolution.py
```

**Requirements**:
- Valid API keys in `.env` file for all configured models
- `config/models.yaml` properly set up
- Sufficient API quota (evolution can require many LLM calls)

**Expected output**:
- Best evolved trajectory for each query
- Fitness scores and generation statistics
- Checkpoint files saved to `outputs/checkpoints/`

### 2. `custom_config.py` (TODO)

**Purpose**: Shows how to use custom YAML configurations for different scenarios.

**What it will show**:
- Loading configurations from YAML files
- Comparing different hyperparameter settings
- Customizing mutation mode distributions
- Adjusting convergence thresholds

### 3. `dataset_evolution.py` (TODO)

**Purpose**: Demonstrates evolving CoTs for an entire dataset.

**What it will show**:
- Loading BioProBench or ChemCoT datasets
- Running evolution on full dataset with progress tracking
- Exporting evolved CoTs in various formats (JSON, JSONL)
- Computing aggregate statistics

### 4. `ablation_study.py` (TODO)

**Purpose**: Implements ablation studies to understand component contributions.

**What it will show**:
- Running evolution with different components disabled
- Comparing NSLC vs random selection
- Impact of different mutation modes
- Effect of knowledge augmentation

## 📊 Example Queries

The examples use chemistry and biology reasoning questions similar to:

**Chemistry**:
```
Query: "What is the molecular weight of glucose (C6H12O6)?"
Ground Truth: "180.16 g/mol"
```

**Biology**:
```
Query: "Design a protocol for protein purification using affinity chromatography."
Ground Truth: [Detailed protocol steps]
```

## 💡 Tips for Running Examples

1. **Start with Small Settings**:
   ```python
   config = EvolutionEngineConfig(
       n_generations=2,  # Start small
       population_size=6,  # Minimize API calls
       n_vanilla=4,
       n_knowledge_augmented=2
   )
   ```

2. **Use Checkpoints**:
   ```python
   best = await engine.evolve(
       query,
       answer,
       checkpoint_path=Path("checkpoints/my_query.pkl")  # Enable resume
   )
   ```

3. **Monitor Progress**:
   - Check logs for generation statistics
   - Review intermediate checkpoints
   - Adjust hyperparameters based on convergence

4. **Control Costs**:
   - Use `max_concurrent` to limit parallel API calls
   - Start with smaller population sizes
   - Consider using faster/cheaper models for initial testing

## 🐛 Troubleshooting

**Issue**: ModuleNotFoundError
- **Solution**: Make sure you're running from the project root directory
- **Install**: `pip install -r requirements.txt`

**Issue**: API key errors
- **Solution**: Check your `.env` file has valid keys
- **Verify**: Run `python -c "import os; print(os.getenv('MODEL_DEEPSEEK_R1_API_KEY'))"`

**Issue**: Slow evolution
- **Solution**: Reduce `n_generations` and `population_size`
- **Optimize**: Use `max_concurrent=1` if hitting rate limits

## 📚 Next Steps

After running the examples:
1. Review the generated CoTs in `outputs/`
2. Experiment with different hyperparameters
3. Try your own queries from your domain
4. Integrate into your own training pipeline

## 🤝 Contributing Examples

Have a useful example? Please contribute!
1. Place your script in this directory
2. Add clear documentation in this README
3. Include expected outputs and requirements
4. Follow the naming convention: `category_description.py`
