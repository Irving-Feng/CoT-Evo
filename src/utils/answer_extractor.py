"""
Answer extraction utilities for CoT-Evo framework.

This module provides functions to extract answers and reasoning from model outputs,
properly handling the Supplementary Material format: <|think|> and <|answer|>
"""

import re
import logging
from typing import Tuple, Optional
import json

logger = logging.getLogger(__name__)


def extract_cot_and_answer(model_output: str) -> Tuple[str, str]:
    """
    Extract CoT reasoning and final answer from model output.

    Primary format (Supplementary Material):
    <|think|>
    {reasoning}
    <|answer|>
    {answer}

    Fallback formats:
    - "reasoning...\n\nfinal result: <answer>"
    - "reasoning...\n\nThe answer is: <answer>"
    - JSON format: {"Major Product": "..."}

    Args:
        model_output: Full model output

    Returns:
        Tuple of (reasoning, answer)
    """
    # Handle None or empty input
    if model_output is None:
        logger.warning("Model output is None, returning empty reasoning and answer")
        return "", ""

    if not isinstance(model_output, str):
        logger.warning(f"Model output is not a string (type: {type(model_output)}), converting to string")
        model_output = str(model_output)

    # Priority 1: Try <|think|>...<|answer|>... format (Supplementary Material)
    think_start = model_output.find("<|think|>")
    think_end = model_output.find("<|answer|>")

    if think_start != -1 and think_end != -1:
        reasoning = model_output[think_start + 9:think_end].strip()
        answer = model_output[think_end + 10:].strip()
        logger.debug(f"Extracted reasoning ({len(reasoning)} chars) and answer using <|think|>/<|answer|> format")
        return reasoning, answer

    # Priority 2: Try JSON format (ChemCoTDataset common)
    # Look for {"Major Product": "..."} or {"result": "..."}
    json_patterns = [
        r'\{\s*"Major Product"\s*:\s*"[^"]+)"\s*\}',
        r'\{\s*"result"\s*:\s*"[^"]+)"\s*\}',
        r'\{\s*"answer"\s*:\s*"[^"]+)"\s*\}',
    ]

    for pattern in json_patterns:
        match = re.search(pattern, model_output)
        if match:
            json_str = match.group(0)
            try:
                data = json.loads(json_str)
                # Extract the value
                for key in ["Major Product", "result", "answer"]:
                    if key in data:
                        answer = json.dumps({key: data[key]})
                        # Find reasoning before the JSON
                        reasoning = model_output[:match.start()].strip()
                        logger.debug(f"Extracted reasoning ({len(reasoning)} chars) and answer using JSON format")
                        return reasoning, answer
            except json.JSONDecodeError:
                continue

    # Priority 3: Try "final result:" or similar patterns
    patterns = [
        r'\n\nfinal result\s*:?\s*(.+?)\s*$',
        r'\n\nThe answer is\s*:?\s*(.+?)\s*$',
        r'\n\nAnswer\s*:?\s*(.+?)\s*$',
        r'\n\nThus, the answer is\s*:?\s*(.+?)\s*$',
    ]

    for pattern in patterns:
        match = re.search(pattern, model_output, re.IGNORECASE | re.DOTALL)
        if match:
            answer = match.group(1).strip()
            reasoning = model_output[:match.start()].strip()
            logger.debug(f"Extracted reasoning ({len(reasoning)} chars) and answer using pattern matching")
            return reasoning, answer

    # Priority 4: Fallback - treat last line as answer
    lines = model_output.split('\n')
    if len(lines) > 1:
        # Last non-empty line is likely the answer
        for i in range(len(lines) - 1, -1, -1):
            if lines[i].strip():
                answer = lines[i].strip()
                reasoning = '\n'.join(lines[:i]).strip()
                logger.debug(f"Fallback extraction: answer={answer}")
                return reasoning, answer

    # Last resort: return everything as reasoning, empty answer
    logger.warning("Could not extract answer, treating all as reasoning")
    return model_output.strip(), ""


def extract_cot_from_markers(text: str) -> Tuple[str, str]:
    """
    Extract CoT from <|think|>...<|answer|>... markers.

    This is a specialized function for use in crossover and mutation operations
    where the input is expected to be in the marker format.

    Args:
        text: Text containing <|think|>...<|answer|>... markers

    Returns:
        Tuple of (reasoning, answer)

    Raises:
        ValueError: If markers are not found
    """
    think_start = text.find("<|think|>")
    think_end = text.find("<|answer|>")

    if think_start == -1:
        raise ValueError("<|think|> marker not found in text")
    if think_end == -1:
        raise ValueError("<|answer|> marker not found in text")
    if think_end <= think_start:
        raise ValueError("<|answer|> appears before <|think|>")

    reasoning = text[think_start + 9:think_end].strip()
    answer = text[think_end + 10:].strip()

    return reasoning, answer


def extract_answer_from_seed_pair(seed_pair: dict) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Extract reasoning and answer from a dataset seed pair.

    Args:
        seed_pair: A seed pair from the dataset with 'response' field

    Returns:
        Tuple of (reasoning, answer, model_name)
    """
    response = seed_pair.get('response', '')
    model_name = seed_pair.get('model', 'unknown')

    reasoning, answer = extract_cot_and_answer(response)

    if not reasoning:
        return None, None, None

    return reasoning, answer, model_name


def combine_cot_and_answer(reasoning: str, answer: str) -> str:
    """
    Combine reasoning and answer into Supplementary Material format.

    Args:
        reasoning: The chain-of-thought reasoning
        answer: The final answer

    Returns:
        Combined string in <|think|>/<|answer|> format
    """
    return f"<|think|>\n{reasoning}\n<|answer|>\n{answer}"


def clean_answer(answer: str) -> str:
    """
    Clean answer string for comparison.

    Handles various formats:
    - JSON: {"Major Product": "SMILES"}
    - <|answer|> marker content
    - Plain text

    Args:
        answer: Raw answer string

    Returns:
        Cleaned answer (extracted SMILES if JSON, or plain text)
    """
    if not answer:
        return ""

    # Remove <|think|> and <|answer|> markers if present
    answer = answer.replace("<|think|>", "").replace("<|answer|>", "").strip()

    # If it's JSON format, extract the actual value
    try:
        # Try parsing as JSON
        data = json.loads(answer)
        if isinstance(data, dict):
            # Look for common keys
            for key in ["Major Product", "result", "answer", "By Product"]:
                if key in data:
                    value = data[key]
                    if isinstance(value, str):
                        return value.strip()
            # If dict but no known key, return the JSON string
            return answer
    except json.JSONDecodeError:
        pass

    # Remove common prefixes
    answer = re.sub(r'^(final result|answer|the answer is)\s*:?\s*', '', answer, flags=re.IGNORECASE)

    # Remove extra whitespace
    answer = ' '.join(answer.split())

    return answer.strip()


def format_final_result(answer: str) -> str:
    """
    Format an answer in the <|think|>/<|answer|> format.

    Args:
        answer: The answer to format

    Returns:
        Formatted string with markers
    """
    # Try to detect if answer is already in JSON format
    try:
        data = json.loads(answer)
        if isinstance(data, dict):
            answer = json.dumps(data)
    except json.JSONDecodeError:
        pass

    return f"<|think|>\nYour reasoning goes here\n<|answer|>\n{answer}"
