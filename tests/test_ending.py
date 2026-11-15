"""Unit tests for the Smart Ending Refinement Engine (studio/ending.py).

Covers:
1. Original timestamp falling between two word boundaries.
2. Original timestamp cutting inside a word (not matching word ending).
3. Audio waveform with a non-zero source offset.
4. Stereo audio (N, 2).
5. Multi-channel audio (4 channels, and channels-first 2, N).
6. Audio unavailable (None).
7. Audio context insufficient (timestamp beyond buffer).
8. Candidate selection with competing sentence boundaries (distance penalty check).
9. Already natural ending with no unnecessary extension.
10. Incorrect or unordered transcript timestamps.
"""
import unittest
import numpy as np

from studio.ending import (
    evaluate_audio_pause,
    evaluate_baseline_endpoint,
    refine_clip_endpoint,
    score_candidate_boundary,
    RefinementResult,
)


class TestSmartEndingRefinement(unittest.TestCase):

    def setUp(self):
        self.sample_rate = 16000

    def _create_synthetic_audio(self, duration_sec, noise_level=0.001, tone_freq=None, tone_segments=None, channels=1, channels_first=False):
        """Create synthetic audio array at 16kHz with optional channels and tones."""
        n_samples = int(duration_sec * self.sample_rate)
        if channels == 1:
            audio = np.random.normal(0, noise_level, n_samples).astype(np.float32)
        else:
            if channels_first:
                audio = np.random.normal(0, noise_level, (channels, n_samples)).astype(np.float32)
            else:
                audio = np.random.normal(0, noise_level, (n_samples, channels)).astype(np.float32)

        if tone_segments and tone_freq:
            t = np.arange(n_samples) / self.sample_rate
            sine = 0.5 * np.sin(2 * np.pi * tone_freq * t).astype(np.float32)
            for start_s, end_s in tone_segments:
                i0 = int(start_s * self.sample_rate)
                i1 = min(n_samples, int(end_s * self.sample_rate))
                if channels == 1:
                    audio[i0:i1] += sine[i0:i1]
                elif channels_first:
                    audio[:, i0:i1] += sine[i0:i1]
                else:
                    for ch in range(channels):
                        audio[i0:i1, ch] += sine[i0:i1]

        return np.clip(audio, -1.0, 1.0)

    def test_original_timestamp_in_natural_gap_preserved(self):
        """1. Original timestamp falling between two word boundaries in a natural pause is preserved."""
        start = 0.0
        original_end = 20.0  # Sits at 20.0, cleanly in the 1.5s gap between 19.3 and 20.8

        words = [
            {"w": "Great", "s": 18.0, "e": 18.4},
            {"w": "shot.", "s": 18.5, "e": 19.3},  # sentence completes at 19.3
            {"w": "Next", "s": 20.8, "e": 21.2},   # next topic starts at 20.8
        ]

        res = refine_clip_endpoint(
            start=start,
            end=original_end,
            words=words,
            min_len=10.0,
            max_len=30.0,
            video_duration=60.0,
        )

        self.assertFalse(res.changed)
        self.assertEqual(res.end, 20.0)
        self.assertIn("preserved_original", res.reason)

    def test_original_timestamp_cutting_midword_is_corrected(self):
        """2. Original timestamp cuts in the middle of an active word ('running')."""
        start = 0.0
        original_end = 18.3  # Cuts right in the middle of "running" (18.0 to 18.6)

        words = [
            {"w": "He", "s": 17.5, "e": 17.9},
            {"w": "kept", "s": 18.0, "e": 18.2},
            {"w": "running.", "s": 18.2, "e": 18.8},  # word spans across 18.3
            {"w": "Then", "s": 20.0, "e": 20.4},
        ]

        res = refine_clip_endpoint(
            start=start,
            end=original_end,
            words=words,
            min_len=10.0,
            max_len=30.0,
            video_duration=60.0,
        )

        self.assertTrue(res.changed)
        # Should cleanly align to completion of "running." (18.8 + pad)
        self.assertAlmostEqual(res.end, 18.95, delta=0.15)
        self.assertIn("running", res.reason)

    def test_audio_waveform_with_source_offset(self):
        """3. Audio waveform represents an extracted segment starting at audio_offset=50.0."""
        start = 50.0
        original_end = 68.0  # Video time 68.0 = 18.0s into the audio segment

        words = [
            {"w": "Nice", "s": 65.0, "e": 65.5},
            {"w": "try", "s": 65.6, "e": 66.0},
            {"w": "because", "s": 66.1, "e": 67.0},  # hanging conjunction at 67.0
            {"w": "we", "s": 67.1, "e": 67.4},
            {"w": "scored.", "s": 67.5, "e": 68.2},   # completion at 68.2
            {"w": "Anyway", "s": 70.0, "e": 70.5},
        ]

        # Audio has 25 seconds of sound, corresponding to video timeline 50.0 to 75.0s
        # Tone active from 65.0 to 68.2 (relative: 15.0 to 18.2s in audio)
        audio = self._create_synthetic_audio(
            duration_sec=25.0,
            noise_level=0.001,
            tone_freq=440,
            tone_segments=[(15.0, 18.2)],
        )

        res = refine_clip_endpoint(
            start=start,
            end=original_end,
            words=words,
            audio=audio,
            audio_offset=50.0,
            min_len=10.0,
            max_len=30.0,
            video_duration=100.0,
        )

        self.assertTrue(res.changed)
        self.assertEqual(res.audio_status, "validated")
        self.assertTrue(res.audio_validated)
        self.assertAlmostEqual(res.end, 68.35, delta=0.15)

    def test_stereo_audio_handling(self):
        """4. Audio is 2-channel stereo with shape (N, 2)."""
        audio_stereo = self._create_synthetic_audio(
            duration_sec=20.0,
            noise_level=0.002,
            tone_freq=300,
            tone_segments=[(15.0, 18.0)],
            channels=2,
            channels_first=False,
        )
        self.assertEqual(audio_stereo.ndim, 2)
        self.assertEqual(audio_stereo.shape[1], 2)

        score, status, reason = evaluate_audio_pause(audio_stereo, timestamp=18.0)
        self.assertIn(status, ("validated", "inconclusive"))
        self.assertIsInstance(score, float)

    def test_multichannel_and_channels_first_audio(self):
        """5. Multi-channel audio (4 channels) and channels-first (2, N)."""
        audio_4ch = self._create_synthetic_audio(
            duration_sec=10.0,
            channels=4,
            channels_first=False,
        )
        score_4ch, status_4ch, _ = evaluate_audio_pause(audio_4ch, timestamp=5.0)
        self.assertIsInstance(score_4ch, float)

        audio_2xN = self._create_synthetic_audio(
            duration_sec=10.0,
            channels=2,
            channels_first=True,
        )
        score_2xN, status_2xN, _ = evaluate_audio_pause(audio_2xN, timestamp=5.0)
        self.assertIsInstance(score_2xN, float)

    def test_audio_unavailable(self):
        """6. When audio is None, audio_status is 'unavailable' and audio_validated is False."""
        words = [
            {"w": "Hello", "s": 10.0, "e": 10.5},
            {"w": "world.", "s": 10.6, "e": 11.0},
        ]

        res = refine_clip_endpoint(
            start=0.0,
            end=11.0,
            words=words,
            audio=None,
            min_len=5.0,
            max_len=30.0,
        )

        self.assertEqual(res.audio_status, "unavailable")
        self.assertFalse(res.audio_validated)

    def test_audio_context_insufficient(self):
        """7. Audio buffer ends before timestamp -> status is 'insufficient_context'."""
        short_audio = self._create_synthetic_audio(duration_sec=5.0)

        score, status, reason = evaluate_audio_pause(short_audio, timestamp=10.0)
        self.assertEqual(status, "insufficient_context")
        self.assertEqual(score, 0.0)

    def test_competing_sentence_boundaries_distance_penalty(self):
        """8. A closer complete sentence wins over a distant sentence despite both having periods."""
        start = 0.0
        original_end = 20.0

        words = [
            {"w": "We", "s": 18.0, "e": 18.3},
            {"w": "won.", "s": 18.4, "e": 18.8},  # Sentence 1: +0.2s from orig_end (ends at 18.8, gap of 1.2s)
            {"w": "The", "s": 20.0, "e": 20.3},
            {"w": "crowd", "s": 20.4, "e": 20.8},
            {"w": "cheered", "s": 20.9, "e": 21.5},
            {"w": "loudly", "s": 21.6, "e": 22.0},
            {"w": "for", "s": 22.1, "e": 22.4},
            {"w": "hours.", "s": 22.5, "e": 25.5},  # Sentence 2: 5.5s away from orig_end
        ]

        res = refine_clip_endpoint(
            start=start,
            end=original_end,
            words=words,
            min_len=10.0,
            max_len=30.0,
            video_duration=60.0,
        )

        # The closer clean sentence "won." should win over the distant "hours."
        self.assertIn("won", res.details["word"])
        self.assertLess(res.end, 20.5)

    def test_already_natural_ending_unchanged(self):
        """9. An already natural ending in silence after a terminal punctuation is unchanged."""
        start = 0.0
        original_end = 20.5

        words = [
            {"w": "That's", "s": 18.0, "e": 18.4},
            {"w": "all", "s": 18.5, "e": 18.8},
            {"w": "folks!", "s": 18.9, "e": 19.5},  # Clean complete sentence, 1s pause before 20.5
            {"w": "See", "s": 22.0, "e": 22.4},     # Next speaker after 2.5s pause
        ]

        res = refine_clip_endpoint(
            start=start,
            end=original_end,
            words=words,
            min_len=10.0,
            max_len=30.0,
            video_duration=60.0,
        )

        self.assertFalse(res.changed)
        self.assertEqual(res.end, 20.5)

    def test_unordered_and_invalid_transcript_timestamps(self):
        """10. Unordered, negative, or inverted timestamps are sanitized without crashing."""
        words_messy = [
            {"w": "second", "s": 15.0, "e": 15.5},
            {"w": "first", "s": 14.0, "e": 14.5},    # Out of order
            {"w": "corrupt", "s": 16.0, "e": 15.0},  # Inverted (e < s)
            {"w": "invalid", "s": "bad", "e": 18.0}, # Non-numeric
            {"w": "third.", "s": 15.6, "e": 16.2},
        ]

        res = refine_clip_endpoint(
            start=0.0,
            end=16.0,
            words=words_messy,
            min_len=10.0,
            max_len=30.0,
            video_duration=60.0,
        )

        # Successfully processes sanitized words
        self.assertIsInstance(res.end, float)
        self.assertGreaterEqual(res.end, 10.0)
        self.assertIn("third", res.details["word"])


if __name__ == "__main__":
    unittest.main()
