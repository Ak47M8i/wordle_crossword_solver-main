"""Crossword Clue Matcher Engine: pattern parsing, regex filtering, and TF-IDF semantic clue matching."""

import re
from typing import Any, Dict, List, Optional, Tuple

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    from data.crossword_data import get_all_crossword_entries
except ImportError:
    from .data.crossword_data import get_all_crossword_entries


class CrosswordEngine:
    """Engine for pattern matching and semantic clue matching against a crossword database."""

    def __init__(self):
        self.entries = get_all_crossword_entries()
        # Unique list of vocabulary words for fast pattern matching
        self.unique_answers = sorted(list({entry["answer"] for entry in self.entries}))
        self.corpus_clues = [entry["clue"] for entry in self.entries]
        
        self.vectorizer = None
        self.tfidf_matrix = None
        if SKLEARN_AVAILABLE and self.corpus_clues:
            try:
                self.vectorizer = TfidfVectorizer(
                    ngram_range=(1, 2),
                    stop_words="english",
                    token_pattern=r"(?u)\b\w+\b"
                )
                self.tfidf_matrix = self.vectorizer.fit_transform(self.corpus_clues)
            except Exception:
                self.vectorizer = None
                self.tfidf_matrix = None

    @staticmethod
    def parse_pattern_to_regex(pattern_str: str, length_constraint: Optional[int] = None) -> Tuple[Optional[re.Pattern], int]:
        """Parses user input patterns like 'C _ _ T', 'C..T', 'c??t', or 'C___' into a compiled regex.
        
        Returns (compiled_regex, detected_length).
        """
        raw = pattern_str.strip()
        if not raw:
            if length_constraint and length_constraint > 0:
                pattern = f"^[A-Z]{{{length_constraint}}}$"
                return re.compile(pattern, re.IGNORECASE), length_constraint
            return None, 0

        # Check if user entered advanced regex directly (contains ^, $, or brackets)
        if raw.startswith("^") or raw.endswith("$") or ("[" in raw and "]" in raw):
            try:
                clean_regex = raw
                if not clean_regex.startswith("^"):
                    clean_regex = "^" + clean_regex
                if not clean_regex.endswith("$"):
                    clean_regex = clean_regex + "$"
                compiled = re.compile(clean_regex, re.IGNORECASE)
                return compiled, 0
            except re.error:
                pass  # Fall through to standard parsing

        # Standard crossword pattern formatting:
        # User might use spaces between letters, e.g., 'C _ _ T' or 'P A R I S' or 'C . . T'
        # Normalize: replace multiple whitespace with nothing if they separate letters/symbols
        tokens = []
        for ch in raw:
            if ch.isalpha():
                tokens.append(ch.upper())
            elif ch in ['_', '.', '?', '*']:
                tokens.append('.')
            elif ch.isspace():
                continue
            else:
                # Keep other chars or treat as wildcard
                tokens.append('.')

        pattern_length = len(tokens)
        if length_constraint and length_constraint > 0:
            if pattern_length < length_constraint:
                # Pad with wildcards
                tokens.extend(['.'] * (length_constraint - pattern_length))
            elif pattern_length > length_constraint:
                tokens = tokens[:length_constraint]
            target_length = length_constraint
        else:
            target_length = pattern_length

        regex_str = "^" + "".join(tokens) + "$"
        try:
            compiled = re.compile(regex_str, re.IGNORECASE)
            return compiled, target_length
        except re.error:
            return None, target_length

    def match_words_by_pattern(self, compiled_regex: Optional[re.Pattern], length_limit: Optional[int] = None) -> List[str]:
        """Filters candidate answers from the database that match the regex pattern and length."""
        if not compiled_regex:
            if length_limit and length_limit > 0:
                return [w for w in self.unique_answers if len(w) == length_limit]
            return list(self.unique_answers)

        matches = []
        for word in self.unique_answers:
            if length_limit and len(word) != length_limit:
                continue
            if compiled_regex.match(word):
                matches.append(word)
        return matches

    def _fallback_keyword_similarity(self, user_clue: str, entry_clue: str) -> float:
        """Simple Jaccard token similarity fallback when TF-IDF is unavailable."""
        user_tokens = set(re.findall(r"\w+", user_clue.lower()))
        entry_tokens = set(re.findall(r"\w+", entry_clue.lower()))
        if not user_tokens or not entry_tokens:
            return 0.0
        intersection = user_tokens.intersection(entry_tokens)
        union = user_tokens.union(entry_tokens)
        return len(intersection) / len(union)

    def search_candidates(
        self,
        clue: str = "",
        pattern: str = "",
        length: Optional[int] = None,
        top_k: int = 25
    ) -> List[Dict[str, Any]]:
        """Searches and ranks crossword answers using regex pattern matching and TF-IDF semantic scoring."""
        compiled_regex, detected_len = self.parse_pattern_to_regex(pattern, length)
        effective_len = length if (length and length > 0) else (detected_len if detected_len > 0 else None)

        # 1. Filter answers by pattern
        pattern_matches = set(self.match_words_by_pattern(compiled_regex, effective_len))
        if not pattern_matches:
            return []

        cleaned_clue = clue.strip()
        has_clue = len(cleaned_clue) > 0

        # 2. Compute semantic similarity if clue is provided
        clue_similarities = {}
        if has_clue and SKLEARN_AVAILABLE and self.vectorizer and self.tfidf_matrix is not None:
            try:
                clue_vec = self.vectorizer.transform([cleaned_clue])
                sim_matrix = cosine_similarity(clue_vec, self.tfidf_matrix)[0]
                for idx, entry in enumerate(self.entries):
                    sim = float(sim_matrix[idx])
                    ans = entry["answer"]
                    if ans in pattern_matches:
                        if ans not in clue_similarities or sim > clue_similarities[ans]["max_sim"]:
                            clue_similarities[ans] = {
                                "max_sim": sim,
                                "best_clue": entry["clue"],
                                "category": entry["category"],
                                "frequency": entry["frequency"],
                                "source": entry["source"]
                            }
            except Exception:
                clue_similarities = {}

        # Fallback or supplementary check
        if has_clue and not clue_similarities:
            for entry in self.entries:
                ans = entry["answer"]
                if ans in pattern_matches:
                    sim = self._fallback_keyword_similarity(cleaned_clue, entry["clue"])
                    if ans not in clue_similarities or sim > clue_similarities[ans]["max_sim"]:
                        clue_similarities[ans] = {
                            "max_sim": sim,
                            "best_clue": entry["clue"],
                            "category": entry["category"],
                            "frequency": entry["frequency"],
                            "source": entry["source"]
                        }

        # 3. Assemble and rank candidates
        results = []
        for word in pattern_matches:
            entry_info = clue_similarities.get(word)
            if not entry_info:
                # Find any entry for this word to get default clue/category
                default_entry = next((e for e in self.entries if e["answer"] == word), None)
                category = default_entry["category"] if default_entry else "General"
                best_clue = default_entry["clue"] if default_entry else "Standard vocabulary term"
                freq = default_entry["frequency"] if default_entry else 5
                source = default_entry["source"] if default_entry else "Dictionary"
                max_sim = 0.0
            else:
                category = entry_info["category"]
                best_clue = entry_info["best_clue"]
                freq = entry_info["frequency"]
                source = entry_info["source"]
                max_sim = entry_info["max_sim"]

            # Compute combined score
            # If clue is provided: 65% semantic similarity, 25% word frequency, 10% exact keyword presence
            if has_clue:
                # Check for exact token containment in clue
                clue_lower = cleaned_clue.lower()
                direct_keyword_bonus = 0.3 if (word.lower() in clue_lower or any(tok in clue_lower for tok in best_clue.lower().split())) else 0.0
                
                # Semantic similarity normalized (often 0.1 - 0.9)
                raw_score = (max_sim * 60.0) + (freq * 2.5) + (direct_keyword_bonus * 15.0)
                # Boost if similarity is exceptionally high
                if max_sim > 0.4:
                    raw_score += 20.0
                final_score = min(100.0, max(5.0, round(raw_score, 1)))
            else:
                # Rank primarily by frequency and word commonality when no clue provided
                final_score = round(min(100.0, freq * 10.0), 1)

            results.append({
                "answer": word,
                "length": len(word),
                "score": final_score,
                "semantic_similarity": round(max_sim * 100, 1) if has_clue else None,
                "best_clue": best_clue,
                "category": category,
                "frequency": freq,
                "source": source
            })

        # Sort descending by score, then answer alphabetically
        results.sort(key=lambda item: (-item["score"], item["answer"]))
        return results[:top_k]
