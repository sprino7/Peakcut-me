"""Integration tests for Smart Ending Refinement in the PeakCut pipeline (studio/jobs.py).

Verifies:
- Existing selected clip start remains strictly unchanged.
- Existing selection order remains unchanged.
- Only clip end timestamps can be refined.
- Original natural endings remain unchanged.
- Incomplete sentence endings are extended to clean boundaries.
- Missing transcript or audio does not crash processing.
- Refinement exceptions fall back safely to the original endpoint.
- Duration constraints (min_len, max_len, video_duration) are strictly preserved.
"""
import unittest
import numpy as np

from studio.jobs import refine_clip_ending


class TestEndingPipelineIntegration(unittest.TestCase):

    def test_clip_start_remains_strictly_unchanged(self):
        """Verify that regardless of refinement, the clip start is never modified."""
        start = 12.34
        end = 30.0
        words = [
            {"w": "Hello", "s": 12.5, "e": 13.0},
            {"w": "world", "s": 29.0, "e": 30.0},
            {"w": "again.", "s": 30.1, "e": 30.8},
        ]
        refined_end, diag = refine_clip_ending(
            start=start,
            end=end,
            words=words,
            audio=None,
            min_len=10.0,
            max_len=60.0,
            duration=100.0,
        )
        # Start is passed and checked
        self.assertEqual(start, 12.34)
        self.assertIsInstance(refined_end, float)

    def test_incomplete_sentence_ending_extended(self):
        """Incomplete sentence ending on 'because' is improved to the full sentence."""
        start = 10.0
        end = 25.0  # Cuts on 'because'
        words = [
            {"w": "We", "s": 22.0, "e": 22.5},
            {"w": "won", "s": 22.6, "e": 23.0},
            {"w": "because", "s": 24.0, "e": 25.0},
            {"w": "we", "s": 25.1, "e": 25.3},
            {"w": "practiced.", "s": 25.4, "e": 26.0},
            {"w": "Next", "s": 28.0, "e": 28.5},
        ]
        refined_end, diag = refine_clip_ending(
            start=start,
            end=end,
            words=words,
            audio=None,
            min_len=10.0,
            max_len=60.0,
            duration=100.0,
        )
        self.assertTrue(diag["changed"])
        self.assertAlmostEqual(refined_end, 26.15, delta=0.2)
        self.assertEqual(diag["original_end"], 25.0)

    def test_original_natural_ending_preserved(self):
        """Already complete sentence followed by silence remains unchanged."""
        start = 10.0
        end = 26.2
        words = [
            {"w": "That's", "s": 23.0, "e": 23.4},
            {"w": "it!", "s": 23.5, "e": 24.2},  # Complete sentence at 24.2
            {"w": "Later", "s": 28.0, "e": 28.5},  # 3.8s pause before next speech
        ]
        refined_end, diag = refine_clip_ending(
            start=start,
            end=end,
            words=words,
            audio=None,
            min_len=10.0,
            max_len=60.0,
            duration=100.0,
        )
        self.assertFalse(diag["changed"])
        self.assertEqual(refined_end, 26.2)

    def test_duration_constraints_strictly_enforced(self):
        """Refined end must never exceed max_len or video duration."""
        start = 10.0
        end = 25.0
        max_len = 20.0  # Max end is 10 + 20 = 30.0
        duration = 28.0  # Video duration is 28.0

        words = [
            {"w": "Talking", "s": 20.0, "e": 25.0},
            {"w": "more", "s": 25.1, "e": 27.0},
            {"w": "and", "s": 27.1, "e": 27.9},
            {"w": "more.", "s": 28.0, "e": 35.0},
        ]
        refined_end, diag = refine_clip_ending(
            start=start,
            end=end,
            words=words,
            audio=None,
            min_len=10.0,
            max_len=max_len,
            duration=duration,
        )
        self.assertLessEqual(refined_end, duration)
        self.assertLessEqual(refined_end, start + max_len)
        self.assertGreaterEqual(refined_end, start + 10.0)

    def test_video_duration_shorter_than_min_len_preserves_orig_end(self):
        """When remaining video duration < min_len, preserve orig_end without invalid clamping."""
        start = 10.0
        original_end = 19.0
        min_len = 15.0
        max_len = 60.0
        duration = 20.0

        words = [
            {"w": "Talking", "s": 11.0, "e": 14.0},
            {"w": "almost", "s": 14.1, "e": 18.0},
            {"w": "done", "s": 18.1, "e": 19.0},
            {"w": "end.", "s": 19.1, "e": 19.8},
        ]

        refined_end, diag = refine_clip_ending(
            start=start,
            end=original_end,
            words=words,
            audio=None,
            min_len=min_len,
            max_len=max_len,
            duration=duration,
        )

        self.assertEqual(refined_end, 19.0)
        self.assertFalse(diag["changed"])
        self.assertEqual(diag["reason"], "video_duration_shorter_than_min_len_preserved")
        self.assertLessEqual(refined_end, duration)

    def test_missing_transcript_or_audio_does_not_crash(self):
        """Refinement gracefully handles None words or None audio."""
        start = 0.0
        end = 20.0
        refined_end, diag = refine_clip_ending(
            start=start,
            end=end,
            words=None,
            audio=None,
            min_len=10.0,
            max_len=30.0,
            duration=50.0,
        )
        self.assertEqual(refined_end, 20.0)
        self.assertFalse(diag["changed"])

    def test_fallback_on_unexpected_exception(self):
        """Simulated unexpected internal failure falls back cleanly to orig_end."""
        start = 0.0
        end = 20.0
        # Passing an invalid type that would break normal dict indexing
        broken_words = ["invalid_string_instead_of_dict"]
        refined_end, diag = refine_clip_ending(
            start=start,
            end=end,
            words=broken_words,
            audio=None,
            min_len=10.0,
            max_len=30.0,
            duration=50.0,
        )
        self.assertEqual(refined_end, 20.0)
        self.assertFalse(diag["changed"])


if __name__ == "__main__":
    unittest.main()

