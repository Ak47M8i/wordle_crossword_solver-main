"""Unit tests for Wordle solver engine and Crossword clue matcher."""

import unittest
from wordle_engine import (
    compute_feedback,
    filter_candidates,
    calculate_entropy,
    rank_next_guesses,
    simulate_game,
    GREEN,
    YELLOW,
    GRAY
)
from crossword_engine import CrosswordEngine


class TestWordleEngine(unittest.TestCase):
    def test_feedback_exact_match(self):
        feedback = compute_feedback("CRANE", "CRANE")
        self.assertEqual(feedback, (GREEN, GREEN, GREEN, GREEN, GREEN))

    def test_feedback_disjoint(self):
        feedback = compute_feedback("AUDIO", "THYME")
        self.assertEqual(feedback, (GRAY, GRAY, GRAY, GRAY, GRAY))

    def test_feedback_duplicate_letters(self):
        # Guess "ROBOT", Secret "SPOON"
        # Secret has two 'O's (pos 2, 3). Guess has two 'O's (pos 1, 3).
        # Pos 3: O == O -> GREEN
        # Pos 1: O in secret -> YELLOW
        feedback = compute_feedback("ROBOT", "SPOON")
        self.assertEqual(feedback[3], GREEN)   # O == O at pos 3
        self.assertEqual(feedback[1], YELLOW)  # O in secret at pos 1

        # Guess "GEESE", Secret "SHEEP"
        # SHEEP has two 'E's (pos 2, 3) and one 'S' (pos 0).
        # In GEESE:
        # Pos 2 'E' is GREEN (exact match)
        # Pos 1 'E' is YELLOW (second 'E' match)
        # Pos 4 'E' is GRAY (excess 'E')
        # Pos 3 'S' is YELLOW (in SHEEP)
        # Pos 0 'G' is GRAY (not in SHEEP)
        feedback = compute_feedback("GEESE", "SHEEP")
        self.assertEqual(feedback, (GRAY, YELLOW, GREEN, YELLOW, GRAY))

    def test_filter_candidates(self):
        candidates = ["CRANE", "TRACE", "SLATE", "AUDIO"]
        feedback = compute_feedback("CRANE", "TRACE")
        filtered = filter_candidates(candidates, "CRANE", feedback)
        self.assertIn("TRACE", filtered)
        self.assertNotIn("AUDIO", filtered)

    def test_calculate_entropy(self):
        candidates = ["CRANE", "TRACE", "SLATE", "AUDIO", "CRAZY"]
        entropy, exp_rem = calculate_entropy("CRANE", candidates)
        self.assertGreaterEqual(entropy, 0.0)
        self.assertLessEqual(entropy, 8.0)
        self.assertGreaterEqual(exp_rem, 1.0)

    def test_rank_next_guesses(self):
        candidates = ["CRANE", "TRACE", "SLATE"]
        ranked = rank_next_guesses(candidates, top_n=3)
        self.assertGreater(len(ranked), 0)
        self.assertIn("word", ranked[0])
        self.assertIn("entropy", ranked[0])

    def test_simulate_game_win(self):
        # AI plays against "SLATE"
        result = simulate_game(secret_word="SLATE", first_guess="CRANE", max_turns=6)
        self.assertTrue(result["won"])
        self.assertLessEqual(result["turns_taken"], 6)
        self.assertEqual(result["secret_word"], "SLATE")


class TestCrosswordEngine(unittest.TestCase):
    def setUp(self):
        self.engine = CrosswordEngine()

    def test_pattern_parsing(self):
        regex, length = self.engine.parse_pattern_to_regex("C _ _ T", 4)
        self.assertIsNotNone(regex)
        self.assertEqual(length, 4)
        self.assertTrue(bool(regex.match("COAT")))
        self.assertTrue(bool(regex.match("CART")))
        self.assertFalse(bool(regex.match("DOG")))

    def test_crossword_clue_paris(self):
        results = self.engine.search_candidates(
            clue="Capital of France",
            pattern="P _ _ _ S",
            length=5
        )
        self.assertGreater(len(results), 0)
        self.assertEqual(results[0]["answer"], "PARIS")
        self.assertGreaterEqual(results[0]["score"], 80.0)

    def test_crossword_clue_cat(self):
        results = self.engine.search_candidates(
            clue="Feline companion",
            pattern="C _ _",
            length=3
        )
        self.assertGreater(len(results), 0)
        top_answers = [r["answer"] for r in results]
        self.assertIn("CAT", top_answers)

    def test_crossword_clue_cleo(self):
        results = self.engine.search_candidates(
            clue="Egyptian queen for short",
            pattern="C . . O",
            length=4
        )
        self.assertGreater(len(results), 0)
        self.assertEqual(results[0]["answer"], "CLEO")

    def test_empty_matches_handled(self):
        results = self.engine.search_candidates(
            clue="Impossible test",
            pattern="ZZZZZ",
            length=5
        )
        self.assertEqual(len(results), 0)


if __name__ == "__main__":
    unittest.main()
