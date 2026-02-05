"""
Prompt templates for CoT-Evo initialization module.

This module contains system prompts and templates for multi-thinker CoT generation.
"""

# ============================================================================
# System Prompts for Different Datasets
# ============================================================================

SYSTEM_PROMPTS = {
    "ChemCoTDataset": """You are a helpful assistant for answering chemistry-related questions. Given a chemical query, your task is to solve it thoroughly and explicitly provide the final result. For tasks such as Molecule Editing, Molecule Optimization, or Reaction Prediction, you should output the final molecule in SMILES format using a JSON structure in the last line of your response. Unless the user specifies a different output format, the default JSON format should be used: {"result": "final SMILES or answer here"}""",

    "ChemCoTBench": """You are a helpful assistant for answering chemistry-related questions. Given a chemical query, your task is to solve it thoroughly and explicitly provide the final result. For tasks such as Molecule Editing, Molecule Optimization, or Reaction Prediction, you should output the final molecule in SMILES format using a JSON structure in the last line of your response. Unless the user specifies a different output format, the default JSON format should be used: {"result": "final SMILES or answer here"}""",

    "BioProBench": """You are a helpful assistant for answering biological protocol-related questions. Given a biological query, your task is to solve it thoroughly and explicitly provide the final result. You should output the final answer in the last line of your response. Unless the user specifies a different output format, the default format should be used: [ANSWER_START]final answer here[ANSWER_END]""",

    "SciKnowEval": """You are a helpful assistant for answering science-related questions. Given a scientific query, your task is to solve it thoroughly and explicitly provide the final result. You should output the final answer using a JSON structure in the last line of your response. Unless the user specifies a different output format, the default JSON format should be used: {"result": "final answer here"}"""
}

# ============================================================================
# Knowledge Generation Prompts
# ============================================================================

KNOWLEDGE_GENERATION_PROMPT = """You are a scientific reasoning expert. Your task is to identify and extract the necessary knowledge required to solve a given scientific problem.

[Problem]
{query}

[Correct Answer]
{ground_truth}

### Instructions

Analyze the correct answer and identify the key domain knowledge, concepts, formulas, or facts that are necessary to solve this problem. Extract knowledge that:

1. Is essential for reaching the correct answer
2. Might not be commonly known
3. Can be stated as general, context-independent principles

Format your response as clear, concise knowledge snippets that:
- Are accurate and verifiable
- General enough to be useful for similar problems
- Specific enough to be actionable

Output the knowledge in the following format:

[KNOWLEDGE_START]
Your extracted knowledge here
[KNOWLEDGE_END]
"""

# ============================================================================
# Knowledge-Augmented Generation Template
# ============================================================================

KNOWLEDGE_AUGMENTED_TEMPLATE = """You are a helpful assistant for solving scientific problems. You will be provided with additional knowledge to help you reason through the problem.

[Problem]
{query}

[Additional Knowledge]
{knowledge}

### Instructions

Use the provided knowledge to solve the problem. Make sure to:
1. Explicitly reference the knowledge when relevant
2. Show your reasoning clearly
3. Provide the final answer in the specified format

Your response:
"""

# ============================================================================
# Vanilla Generation Template (without knowledge)
# ============================================================================

VANILLA_TEMPLATE = """You are a helpful assistant for solving scientific problems.

[Problem]
{query}

### Instructions

Solve the problem step by step, showing your reasoning clearly. Provide the final answer in the specified format.

Your response:
"""

# ============================================================================
# Stop Sequences for Different Tasks
# ============================================================================

STOP_SEQUENCES = {
    "BioProBench": ["[ANSWER_END]", "\n\n"],
    "ChemCoTDataset": [],
    "ChemCoTBench": [],
    "SciKnowEval": []
}
