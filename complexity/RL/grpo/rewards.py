"""
Reward functions for GRPO training.

QCM (Multiple Choice) exact match is the primary reward:
- Clean binary signal (0 or 1), no reward hacking
- Perfect for GRPO's group-relative advantage estimation
"""

import re
from typing import List, Optional


def extract_answer(text: str) -> Optional[str]:
    """
    Extract a single-letter answer (A/B/C/D) from model output.

    Handles common formats:
      - "The answer is B"
      - "Answer: B"
      - "B)"
      - "**B**"
      - Just "B" alone on a line
    """
    # Pattern 1: "the answer is X" or "answer: X"
    m = re.search(r'(?:the\s+)?answer\s*(?:is|:)\s*([A-Da-d])\b', text, re.IGNORECASE)
    if m:
        return m.group(1).upper()

    # Pattern 2: boxed answer like \boxed{B}
    m = re.search(r'\\boxed\{([A-Da-d])\}', text)
    if m:
        return m.group(1).upper()

    # Pattern 3: "X)" or "X." at start of a line
    m = re.search(r'^([A-Da-d])[).\s]', text.strip(), re.MULTILINE)
    if m:
        return m.group(1).upper()

    # Pattern 4: bold/starred like **B**
    m = re.search(r'\*\*([A-Da-d])\*\*', text)
    if m:
        return m.group(1).upper()

    # Pattern 5: last single letter A-D in the text
    matches = re.findall(r'\b([A-Da-d])\b', text)
    if matches:
        return matches[-1].upper()

    return None


def mcq_exact_match_reward(completions: List[str], answer: str) -> List[float]:
    """
    Binary exact-match reward for multiple choice questions.

    Args:
        completions: List of G model completions for one prompt.
        answer: Ground truth answer letter (A/B/C/D).

    Returns:
        List of rewards (1.0 for correct, 0.0 for wrong).
    """
    answer = answer.strip().upper()
    rewards = []
    for completion in completions:
        extracted = extract_answer(completion)
        rewards.append(1.0 if extracted == answer else 0.0)
    return rewards


def format_reward(completions: List[str]) -> List[float]:
    """
    Reward for following the expected output format.

    Returns 0.5 for well-formatted output ("The answer is X"), 0.0 otherwise.
    """
    rewards = []
    for completion in completions:
        has_format = bool(re.search(
            r'(?:the\s+)?answer\s*(?:is|:)\s*[A-Da-d]\b',
            completion,
            re.IGNORECASE,
        ))
        rewards.append(0.5 if has_format else 0.0)
    return rewards


def combined_reward(
    completions: List[str],
    answer: str,
    format_weight: float = 0.1,
) -> List[float]:
    """
    Combined reward: correctness (0/1) + format bonus (0/0.5).
    """
    correct = mcq_exact_match_reward(completions, answer)
    fmt = format_reward(completions)
    return [c + format_weight * f for c, f in zip(correct, fmt)]
