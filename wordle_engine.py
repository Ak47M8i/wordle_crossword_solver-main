"""Wordle solver logic, Shannon entropy calculation, candidate filtering, and simulation engine."""

import math
from collections import Counter
from typing import Dict, List, Optional, Tuple, Union

try:
    from data.wordle_words import (
        ADDITIONAL_ALLOWED,
        LETTER_FREQUENCIES,
        OPTIMAL_OPENERS,
        WORDLE_TARGETS,
        get_all_valid_words,
        get_target_words,
    )
except ImportError:
    from .data.wordle_words import (
        ADDITIONAL_ALLOWED,
        LETTER_FREQUENCIES,
        OPTIMAL_OPENERS,
        WORDLE_TARGETS,
        get_all_valid_words,
        get_target_words,
    )

# Feedback constants
GRAY = 0    # Letter not in word or extra occurrence
YELLOW = 1  # Letter in word but wrong position
GREEN = 2   # Letter in word and exact position

COLOR_NAMES = {GRAY: "Gray", YELLOW: "Yellow", GREEN: "Green"}
COLOR_ICONS = {GRAY: "⬛", YELLOW: "🟨", GREEN: "🟩"}
HEX_COLORS = {GRAY: "#3a3a3c", YELLOW: "#b59f3b", GREEN: "#538d4e"}


def compute_feedback(guess: str, secret: str) -> Tuple[int, int, int, int, int]:
    """Computes standard Wordle feedback pattern (0=Gray, 1=Yellow, 2=Green).
    
    Correctly accounts for duplicate letters according to official Wordle rules.
    """
    g = guess.upper().strip()
    s = secret.upper().strip()
    if len(g) != 5 or len(s) != 5:
        return (GRAY, GRAY, GRAY, GRAY, GRAY)

    result = [GRAY] * 5
    secret_counts: Dict[str, int] = {}

    # Pass 1: Exact matches (Green)
    for i in range(5):
        if g[i] == s[i]:
            result[i] = GREEN
        else:
            secret_counts[s[i]] = secret_counts.get(s[i], 0) + 1

    # Pass 2: Partial matches (Yellow)
    for i in range(5):
        if result[i] != GREEN:
            char = g[i]
            if secret_counts.get(char, 0) > 0:
                result[i] = YELLOW
                secret_counts[char] -= 1

    return (result[0], result[1], result[2], result[3], result[4])


def pattern_to_int(pattern: Tuple[int, ...]) -> int:
    """Encodes a 5-tuple feedback pattern into an integer in [0, 242]."""
    val = 0
    multiplier = 1
    for p in pattern:
        val += p * multiplier
        multiplier *= 3
    return val


def int_to_pattern(code: int) -> Tuple[int, int, int, int, int]:
    """Decodes an integer in [0, 242] back into a 5-tuple feedback pattern."""
    p = []
    curr = code
    for _ in range(5):
        p.append(curr % 3)
        curr //= 3
    return (p[0], p[1], p[2], p[3], p[4])


def filter_candidates(candidates: List[str], guess: str, pattern: Tuple[int, ...]) -> List[str]:
    """Filters candidate secret words that produce the exact feedback pattern when evaluated against guess."""
    g = guess.upper().strip()
    return [word for word in candidates if compute_feedback(g, word) == pattern]


def calculate_entropy(guess: str, candidate_secrets: List[str]) -> Tuple[float, float]:
    """Calculates the Shannon Entropy (expected information gain in bits) and expected remaining candidates.
    
    Formula:
    H(guess) = sum_{p} P(p) * log2(1 / P(p))
    E[Remaining] = sum_{p} P(p) * Count(p)
    """
    total = len(candidate_secrets)
    if total <= 1:
        return 0.0, float(total)

    g = guess.upper().strip()
    bucket_counts: Dict[Tuple[int, int, int, int, int], int] = Counter()

    for secret in candidate_secrets:
        pattern = compute_feedback(g, secret)
        bucket_counts[pattern] += 1

    entropy = 0.0
    expected_remaining = 0.0

    for count in bucket_counts.values():
        p = count / total
        entropy -= p * math.log2(p)
        expected_remaining += p * count

    return round(entropy, 3), round(expected_remaining, 2)


def calculate_frequency_score(word: str) -> float:
    """Calculates normalized letter frequency heuristic score for a word based on standard English letter frequencies."""
    w = word.upper().strip()
    unique_chars = set(w)
    score = sum(LETTER_FREQUENCIES.get(c, 0.5) for c in unique_chars)
    # Give slight bonus for unique letters (more information)
    if len(unique_chars) == 5:
        score *= 1.1
    return round(score, 2)


def rank_next_guesses(
    candidate_secrets: List[str],
    allowed_vocab: Optional[List[str]] = None,
    top_n: int = 15,
    prioritize_possible: bool = True
) -> List[Dict[str, Union[str, float, bool, int]]]:
    """Ranks optimal next guesses by Shannon entropy and letter frequencies.
    
    Returns structured list of candidate recommendations.
    """
    if not candidate_secrets:
        return []

    total_candidates = len(candidate_secrets)

    # If only 1 or 2 words remain, pick the remaining secret directly
    if total_candidates == 1:
        word = candidate_secrets[0]
        return [{
            "word": word,
            "entropy": 0.0,
            "expected_remaining": 1.0,
            "freq_score": calculate_frequency_score(word),
            "is_possible": True,
            "win_probability": 100.0
        }]

    # When opening with all 2,309 words, use precomputed top openers for instant response
    if total_candidates == len(WORDLE_TARGETS):
        openers = []
        for item in OPTIMAL_OPENERS[:top_n]:
            openers.append({
                "word": item["word"],
                "entropy": item["entropy"],
                "expected_remaining": item["expected_remaining"],
                "freq_score": calculate_frequency_score(item["word"]),
                "is_possible": item["is_possible"],
                "win_probability": round((1.0 / total_candidates) * 100, 2) if item["is_possible"] else 0.0
            })
        return openers

    # Determine which words to evaluate as potential guesses
    secret_set = set(candidate_secrets)
    
    # If remaining candidates are small, evaluate candidates first, plus top burner words if candidates <= 4
    if total_candidates <= 10:
        words_to_evaluate = list(candidate_secrets)
        if allowed_vocab and total_candidates > 2:
            # Also consider top frequency words that could act as informative burners
            sorted_burners = sorted(
                [w for w in allowed_vocab if w not in secret_set],
                key=calculate_frequency_score,
                reverse=True
            )[:30]
            words_to_evaluate.extend(sorted_burners)
    elif total_candidates <= 100:
        words_to_evaluate = list(candidate_secrets)
        if allowed_vocab:
            # Include top 50 frequency words
            sorted_burners = sorted(
                allowed_vocab,
                key=calculate_frequency_score,
                reverse=True
            )[:50]
            for w in sorted_burners:
                if w not in words_to_evaluate:
                    words_to_evaluate.append(w)
    else:
        # Pre-filter evaluate candidates by frequency to keep calculation responsive
        candidate_sample = sorted(
            candidate_secrets,
            key=calculate_frequency_score,
            reverse=True
        )[:150]
        words_to_evaluate = candidate_sample
        if allowed_vocab:
            top_vocab = sorted(
                allowed_vocab,
                key=calculate_frequency_score,
                reverse=True
            )[:100]
            for w in top_vocab:
                if w not in words_to_evaluate:
                    words_to_evaluate.append(w)

    scored_list = []
    for word in words_to_evaluate:
        entropy, exp_rem = calculate_entropy(word, candidate_secrets)
        is_poss = word in secret_set
        freq = calculate_frequency_score(word)
        win_prob = round((1.0 / total_candidates) * 100, 2) if is_poss else 0.0

        scored_list.append({
            "word": word,
            "entropy": entropy,
            "expected_remaining": exp_rem,
            "freq_score": freq,
            "is_possible": is_poss,
            "win_probability": win_prob
        })

    # Sort primarily by Shannon entropy (descending)
    # Tie-break by whether it's a possible secret (gives instant win opportunity), then letter frequency
    def sort_key(item):
        poss_bonus = 0.15 if (prioritize_possible and item["is_possible"]) else 0.0
        return (item["entropy"] + poss_bonus, item["is_possible"], item["freq_score"])

    scored_list.sort(key=sort_key, reverse=True)
    return scored_list[:top_n]


def simulate_game(
    secret_word: str,
    first_guess: str = "SOARE",
    max_turns: int = 6
) -> Dict[str, Union[bool, int, str, List[Dict]]]:
    """Simulates an autonomous AI Wordle game step-by-step using Shannon entropy.
    
    Returns full turn-by-turn history including guesses, patterns, remaining candidate counts,
    and entropy values.
    """
    secret = secret_word.upper().strip()
    target_words = get_target_words()

    if secret not in target_words:
        # If user provides a valid 5-letter word not in targets, add it as valid
        valid_words = get_all_valid_words()
        if secret in valid_words:
            candidates = list(target_words) + [secret]
        else:
            candidates = list(target_words)
    else:
        candidates = list(target_words)

    history = []
    current_guess = first_guess.upper().strip()
    won = False
    all_vocab = list(target_words)

    for turn in range(1, max_turns + 1):
        rem_before = len(candidates)
        feedback = compute_feedback(current_guess, secret)
        
        # Calculate entropy of this guess against candidates before filtering
        entropy, exp_rem = calculate_entropy(current_guess, candidates)
        
        # Filter candidates based on feedback
        candidates = filter_candidates(candidates, current_guess, feedback)
        rem_after = len(candidates)

        history.append({
            "turn": turn,
            "guess": current_guess,
            "feedback": feedback,
            "entropy": entropy,
            "remaining_before": rem_before,
            "remaining_after": rem_after,
            "remaining_sample": candidates[:10],
            "is_correct": feedback == (GREEN, GREEN, GREEN, GREEN, GREEN)
        })

        if feedback == (GREEN, GREEN, GREEN, GREEN, GREEN):
            won = True
            break

        # Calculate best next guess using entropy
        if candidates:
            recommendations = rank_next_guesses(
                candidate_secrets=candidates,
                allowed_vocab=all_vocab,
                top_n=5,
                prioritize_possible=True
            )
            if recommendations:
                current_guess = recommendations[0]["word"]
            else:
                current_guess = candidates[0]
        else:
            break

    return {
        "won": won,
        "turns_taken": len(history),
        "secret_word": secret,
        "history": history
    }
