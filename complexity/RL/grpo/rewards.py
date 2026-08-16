"""
Reward functions for DAPO/GRPO training.

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


# ── Language Quality Rewards (rule-based) ────────────────────────────

def _repetition_score(text: str) -> float:
    """Penalize repetitive text. Returns 0.0 (all repeated) to 1.0 (no repetition)."""
    words = text.lower().split()
    if len(words) < 5:
        return 0.0
    # Trigram repetition ratio
    trigrams = [tuple(words[i:i+3]) for i in range(len(words) - 2)]
    if not trigrams:
        return 0.0
    unique_ratio = len(set(trigrams)) / len(trigrams)
    return min(1.0, unique_ratio)


def _length_score(text: str, target_min: int = 20, target_max: int = 200) -> float:
    """Reward appropriate length. Too short or too long = penalty."""
    words = text.split()
    n = len(words)
    if n < 5:
        return 0.0
    if target_min <= n <= target_max:
        return 1.0
    if n < target_min:
        return n / target_min
    # Soft penalty for too long
    return max(0.0, 1.0 - (n - target_max) / target_max)


def _coherence_score(text: str) -> float:
    """
    Basic coherence heuristics:
    - Has proper sentence structure (periods, capitals)
    - No garbage characters
    - Vocabulary diversity
    """
    score = 0.0

    # Has at least one sentence-ending punctuation
    if re.search(r'[.!?]', text):
        score += 0.3

    # Has capitalized sentence starts
    sentences = re.split(r'[.!?]\s+', text)
    if sentences and any(s and s[0].isupper() for s in sentences if s.strip()):
        score += 0.2

    # Low garbage ratio (non-alphanumeric, non-punctuation)
    if text:
        clean = re.sub(r'[a-zA-Z0-9\s.,!?;:\'"()\-]', '', text)
        garbage_ratio = len(clean) / len(text)
        if garbage_ratio < 0.1:
            score += 0.3

    # Word diversity (unique words / total words)
    words = text.lower().split()
    if len(words) >= 5:
        diversity = len(set(words)) / len(words)
        if diversity > 0.4:
            score += 0.2

    return score


def _topic_drift_score(prompt: str, completion: str) -> float:
    """
    Measure if completion stays on topic with the prompt.

    Compares vocabulary overlap between prompt and chunks of the completion.
    If later chunks share fewer words with the prompt → drift detected.

    Returns 0.0 (total drift) to 1.0 (stays on topic).
    """
    # Extract content words (>3 chars, no stopwords)
    stopwords = {
        "the", "and", "that", "this", "with", "from", "have", "has",
        "been", "were", "was", "are", "for", "not", "but", "what",
        "all", "can", "had", "her", "his", "one", "our", "out",
        "their", "there", "they", "which", "will", "would", "your",
        "about", "also", "into", "more", "some", "than", "them",
        "then", "these", "when", "who", "how", "its", "may", "each",
    }
    def content_words(text):
        words = set(re.findall(r'[a-z]{4,}', text.lower()))
        return words - stopwords

    prompt_words = content_words(prompt)
    if not prompt_words:
        return 1.0  # Can't measure drift without prompt content

    comp_words = completion.split()
    if len(comp_words) < 20:
        return 1.0  # Too short to drift

    # Split completion into first half and second half
    mid = len(comp_words) // 2
    first_half = content_words(" ".join(comp_words[:mid]))
    second_half = content_words(" ".join(comp_words[mid:]))

    # Overlap with prompt
    first_overlap = len(first_half & prompt_words) / max(1, len(prompt_words))
    second_overlap = len(second_half & prompt_words) / max(1, len(prompt_words))

    # If second half loses topic compared to first half → drift
    if first_overlap > 0:
        drift_ratio = second_overlap / first_overlap
        return min(1.0, drift_ratio)

    # Both halves have low overlap — might be on a subtopic
    return min(1.0, second_overlap * 5)


def language_quality_reward(
    completions: List[str],
    prompt: str = "",
    **kwargs,
) -> List[float]:
    """
    Rule-based language quality reward for text generation.

    Scores 5 components (each 0-1, weighted):
      - Repetition:   0.25  (no repeated trigrams)
      - Length:        0.15  (appropriate length)
      - Coherence:    0.25  (punctuation, structure, no garbage)
      - Topic drift:  0.20  (stays on topic with prompt)
      - Non-empty:    0.15  (produced meaningful output)

    Args:
        completions: List of G model completions.
        prompt: The input prompt (used for topic drift detection).

    Returns list of scores in [0.0, 1.0].
    """
    rewards = []
    for text in completions:
        text = text.strip()
        if len(text) < 3:
            rewards.append(0.0)
            continue

        rep = _repetition_score(text)
        length = _length_score(text)
        coh = _coherence_score(text)
        drift = _topic_drift_score(prompt, text) if prompt else 1.0
        non_empty = 1.0 if len(text.split()) >= 5 else 0.0

        score = (0.25 * rep + 0.15 * length + 0.25 * coh
                 + 0.20 * drift + 0.15 * non_empty)
        rewards.append(round(score, 4))
    return rewards
