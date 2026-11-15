"""Smart Ending Refinement Engine for PeakCut.

Finds context-aware, natural clip boundaries by combining semantic transcript
analysis, word-level timestamps, inter-word silence, and acoustic energy.
Keeps existing highlight detection untouched.
"""
import math
import re
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

# Punctuation classifications
TERMINAL_PUNCT = {".", "?", "!"}
CLAUSE_PUNCT = {",", ";", ":", "—", "-"}

# Words that indicate an incomplete thought or dangling clause
HANGING_WORDS = {
    # Coordinating & subordinating conjunctions
    "and", "but", "or", "nor", "for", "yet", "so",
    "because", "although", "though", "while", "whereas",
    "if", "unless", "since", "as", "that", "whether",
    # Prepositions
    "in", "on", "at", "to", "for", "with", "by", "from",
    "about", "into", "through", "during", "before", "after",
    "above", "below", "between", "under", "without", "against",
    # Question / relative words
    "who", "whom", "whose", "which", "what", "where", "when", "why", "how",
    # Auxiliaries / copulae indicating ongoing predicate
    "is", "are", "was", "were", "am", "be", "been", "being",
    "have", "has", "had", "do", "does", "did",
    "can", "could", "will", "would", "shall", "should", "may", "might", "must",
    # Articles & demonstratives
    "a", "an", "the", "this", "these", "those",
    # Contractions suggesting incomplete clause
    "it's", "that's", "there's", "he's", "she's", "we're", "they're", "i'm", "you're",
}

# Configurable search window defaults (in seconds)
DEFAULT_PRE_WINDOW = 4.0
DEFAULT_POST_WINDOW = 6.0

# Audio analysis constants
AUDIO_SR = 16000
FRAME_MS = 30  # 30 ms analysis frame
FRAME_SAMPLES = int(AUDIO_SR * (FRAME_MS / 1000.0))  # 480 samples at 16kHz
SILENCE_RMS_FLOOR = 0.005


@dataclass
class RefinementResult:
    end: float
    original_end: float
    changed: bool
    score: float
    reason: str
    audio_status: str  # "unavailable", "insufficient_context", "inconclusive", "validated"
    audio_validated: bool
    details: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _clean_word_text(text: str) -> str:
    """Strip leading/trailing spaces and symbols for lexical checking."""
    return re.sub(r"[^\w'’\-]", "", text.strip().lower())


def _extract_punctuation(text: str) -> Tuple[str, bool, bool]:
    """Inspect trailing punctuation of a word token.
    Returns (cleaned_word, is_terminal, is_clause)."""
    t = str(text or "").strip()
    if not t:
        return "", False, False
    last_char = t[-1]
    is_term = last_char in TERMINAL_PUNCT
    is_clause = last_char in CLAUSE_PUNCT
    clean = _clean_word_text(t)
    return clean, is_term, is_clause


def _sanitize_words(words: Optional[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """Validate, clean, and sort words chronologically."""
    if not words:
        return []
    cleaned = []
    for w in words:
        if isinstance(w, dict) and "s" in w and "e" in w:
            try:
                s, e = float(w["s"]), float(w["e"])
                if math.isfinite(s) and math.isfinite(e) and e >= s:
                    cleaned.append({**w, "s": s, "e": e})
            except (ValueError, TypeError):
                continue
    cleaned.sort(key=lambda w: w["s"])
    return cleaned


def _to_mono_slice(audio: np.ndarray, start_idx: int, end_idx: int) -> np.ndarray:
    """Extract a slice and convert to 1D float32 normalized to [-1.0, 1.0].
    Handles mono, stereo (N, 2 or 2, N), and multi-channel arrays without full buffer copies."""
    if audio.ndim == 1:
        seg = audio[start_idx:end_idx]
    elif audio.ndim == 2:
        if audio.shape[0] <= 8 and audio.shape[0] < audio.shape[1]:
            # Channels-first: (channels, samples)
            seg = np.mean(audio[:, start_idx:end_idx], axis=0)
        else:
            # Channels-last: (samples, channels)
            seg = np.mean(audio[start_idx:end_idx, :], axis=-1)
    else:
        flat = audio.reshape(audio.shape[0], -1)
        seg = np.mean(flat[start_idx:end_idx, :], axis=-1)

    if np.issubdtype(seg.dtype, np.integer):
        return seg.astype(np.float32) / 32768.0
    return seg.astype(np.float32)


def evaluate_audio_pause(
    audio: Optional[np.ndarray],
    timestamp: float,
    sample_rate: int = AUDIO_SR,
    audio_offset: float = 0.0,
    window_sec: float = 0.4,
) -> Tuple[float, str, str]:
    """Measure acoustic energy immediately following timestamp.
    Returns (score_modifier, audio_status, reason).

    audio_status is one of:
      - 'unavailable': no audio provided
      - 'insufficient_context': timestamp outside buffer or segment too short
      - 'inconclusive': audio was analyzed but energy transition was flat/neutral
      - 'validated': audio confirmed clear silence drop or loud ongoing speech
    """
    if audio is None or len(audio) == 0:
        return 0.0, "unavailable", "no_audio"

    # Calculate index in audio array taking source offset into account
    audio_time = timestamp - audio_offset
    center_sample = int(audio_time * sample_rate)

    total_samples = audio.shape[0] if (audio.ndim == 1 or audio.shape[0] > audio.shape[1]) else audio.shape[1]

    if center_sample < 0 or center_sample >= total_samples:
        return 0.0, "insufficient_context", "timestamp_outside_audio_bounds"

    # Pre-window (250 ms before cut) vs Post-window (window_sec ms after cut)
    pre_samples = int(0.25 * sample_rate)
    post_samples = int(window_sec * sample_rate)

    pre_start = max(0, center_sample - pre_samples)
    post_end = min(total_samples, center_sample + post_samples)

    if (center_sample - pre_start) < 100 or (post_end - center_sample) < 100:
        return 0.0, "insufficient_context", "insufficient_audio_samples"

    pre_seg = _to_mono_slice(audio, pre_start, center_sample)
    post_seg = _to_mono_slice(audio, center_sample, post_end)

    pre_rms = float(np.sqrt(np.mean(pre_seg ** 2) + 1e-9))
    post_rms = float(np.sqrt(np.mean(post_seg ** 2) + 1e-9))

    # Evaluate energy transition
    if post_rms <= SILENCE_RMS_FLOOR:
        return 20.0, "validated", "clean_silence"

    ratio = post_rms / (pre_rms + 1e-6)
    if ratio < 0.4:
        return 15.0, "validated", f"energy_drop_ratio_{ratio:.2f}"
    elif ratio < 0.7:
        return 8.0, "validated", f"moderate_energy_drop_{ratio:.2f}"
    elif ratio > 1.3 and post_rms > 0.03:
        return -25.0, "validated", f"loud_sound_continues_{ratio:.2f}"

    return 0.0, "inconclusive", f"neutral_audio_ratio_{ratio:.2f}"


def score_candidate_boundary(
    word_idx: int,
    words: List[Dict[str, Any]],
    original_end: float,
    start: float,
    min_len: float,
    max_len: float,
    audio: Optional[np.ndarray] = None,
    audio_offset: float = 0.0,
) -> Tuple[float, Dict[str, Any]]:
    """Score a potential word endpoint based on semantic completeness,
    punctuation, pauses, acoustic verification, and distance from original_end.
    """
    w = words[word_idx]
    cand_end = float(w["e"])
    clip_dur = cand_end - start

    # Hard boundary checks
    if clip_dur < min_len or clip_dur > max_len:
        return -1000.0, {"reason": "violates_duration_limits"}

    clean_w, is_term, is_clause = _extract_punctuation(w.get("w", ""))

    # 1. Punctuation score
    punct_score = 0.0
    if is_term:
        punct_score = 45.0
    elif is_clause:
        punct_score = 15.0

    # 2. Hanging word penalty (only if not ending with terminal punctuation)
    hanging_penalty = 0.0
    if clean_w in HANGING_WORDS and not is_term:
        hanging_penalty = -50.0

    # 3. Inter-word pause score
    pause_score = 0.0
    gap = 0.0
    next_word = words[word_idx + 1] if word_idx + 1 < len(words) else None
    if next_word:
        gap = float(next_word["s"]) - cand_end
        if gap >= 0.7:
            pause_score = 35.0
        elif gap >= 0.4:
            pause_score = 25.0
        elif gap >= 0.2:
            pause_score = 10.0
        elif gap < 0.08:
            pause_score = -20.0
    else:
        pause_score = 15.0

    # 4. Sentence completion bonus
    completion_bonus = 0.0
    if is_term and (gap >= 0.25 or next_word is None):
        completion_bonus = 30.0

    # 5. Distance penalty from actual original endpoint
    dist = cand_end - original_end
    if dist > 0:
        dist_penalty = dist * 3.5
        if dist > 3.0:
            dist_penalty += (dist - 3.0) * 2.0  # Progressive penalty for large overrun
    else:
        dist_penalty = abs(dist) * 4.0  # Truncation penalty

    # Stability bonus if candidate is essentially at the original endpoint
    stability_bonus = 15.0 if abs(dist) <= 0.25 else 0.0

    # 6. Audio pause validation
    audio_score, audio_status, audio_reason = evaluate_audio_pause(
        audio, cand_end, audio_offset=audio_offset
    )

    total_score = (
        punct_score
        + hanging_penalty
        + pause_score
        + completion_bonus
        + audio_score
        + stability_bonus
        - dist_penalty
    )

    details = {
        "word": w.get("w"),
        "cand_end": round(cand_end, 2),
        "dist": round(dist, 2),
        "punct_score": punct_score,
        "hanging_penalty": hanging_penalty,
        "pause_score": pause_score,
        "gap": round(gap, 3),
        "completion_bonus": completion_bonus,
        "stability_bonus": stability_bonus,
        "dist_penalty": round(dist_penalty, 2),
        "audio_score": audio_score,
        "audio_status": audio_status,
        "audio_reason": audio_reason,
        "total_score": round(total_score, 2),
    }

    return total_score, details


def evaluate_baseline_endpoint(
    original_end: float,
    start: float,
    words: List[Dict[str, Any]],
    min_len: float,
    max_len: float,
    audio: Optional[np.ndarray] = None,
    audio_offset: float = 0.0,
) -> Tuple[float, Dict[str, Any]]:
    """Directly evaluate the quality and naturalness of the actual original endpoint.
    Avoids assuming the nearest word ending is the original endpoint."""
    clip_dur = original_end - start
    if clip_dur < min_len or clip_dur > max_len:
        return -1000.0, {"cut_midword": False, "hanging": False, "score": -1000.0}

    # Check if original_end lands inside an ongoing word
    cut_word = next((w for w in words if float(w["s"]) < original_end < float(w["e"])), None)
    if cut_word:
        return -50.0, {
            "cut_midword": True,
            "hanging": False,
            "word": cut_word.get("w"),
            "score": -50.0,
            "reason": f"cut_midword_in_'{cut_word.get('w')}'",
        }

    # It lands in an inter-word gap or after words
    prev_word = None
    next_word = None
    for w in words:
        if float(w["e"]) <= original_end + 0.02:
            prev_word = w
        elif float(w["s"]) >= original_end - 0.02 and next_word is None:
            next_word = w

    if not prev_word:
        return 0.0, {"cut_midword": False, "hanging": False, "score": 0.0}

    prev_clean, is_term, is_clause = _extract_punctuation(prev_word.get("w", ""))
    is_hanging = prev_clean in HANGING_WORDS and not is_term

    if is_hanging:
        return -40.0, {
            "cut_midword": False,
            "hanging": True,
            "word": prev_word.get("w"),
            "score": -40.0,
            "reason": f"ends_on_hanging_word_'{prev_word.get('w')}'",
        }

    # Baseline has punctuation and/or natural pause
    score = 20.0  # Baseline stability bonus
    if is_term:
        score += 45.0
    elif is_clause:
        score += 15.0

    gap = (float(next_word["s"]) - float(prev_word["e"])) if next_word else 1.0
    if gap >= 0.7:
        score += 35.0
    elif gap >= 0.4:
        score += 25.0
    elif gap >= 0.2:
        score += 10.0

    if original_end - float(prev_word["e"]) >= 0.12:
        score += 10.0  # Clean post-word vocal clearance

    audio_score, audio_status, audio_reason = evaluate_audio_pause(
        audio, original_end, audio_offset=audio_offset
    )
    score += audio_score

    details = {
        "cut_midword": False,
        "hanging": False,
        "word": prev_word.get("w"),
        "score": round(score, 2),
        "audio_status": audio_status,
        "audio_reason": audio_reason,
    }
    return score, details


def refine_clip_endpoint(
    start: float,
    end: float,
    words: Optional[List[Dict[str, Any]]] = None,
    audio: Optional[np.ndarray] = None,
    audio_offset: float = 0.0,
    min_len: float = 15.0,
    max_len: float = 60.0,
    video_duration: Optional[float] = None,
    pre_window: float = DEFAULT_PRE_WINDOW,
    post_window: float = DEFAULT_POST_WINDOW,
    pad_silence: float = 0.15,
) -> RefinementResult:
    """Smart Ending Refinement Engine.

    Evaluates transcript and audio around `end` to find the most natural,
    context-aware boundary.
    
    Parameters:
    - start: clip start in seconds
    - end: candidate clip end in seconds
    - words: list of word tokens [{'w': str, 's': float, 'e': float}]
    - audio: optional 16kHz audio waveform (mono or multi-channel numpy array)
    - audio_offset: timestamp corresponding to index 0 of audio array (default 0.0)
    - min_len: minimum clip duration allowed
    - max_len: maximum clip duration allowed
    - video_duration: total duration of source video (upper bound)
    - pre_window: maximum seconds to search backwards from `end`
    - post_window: maximum seconds to search forwards from `end`
    - pad_silence: small acoustic breathing room added to word end (default 0.15s)
    
    Returns:
    - RefinementResult dataclass containing refined `end`, `score`, and diagnosis.
    """
    orig_end = float(end)
    hard_max_end = start + max_len
    if video_duration is not None and video_duration > 0:
        hard_max_end = min(hard_max_end, float(video_duration))

    hard_min_end = start + min_len

    clean_words = _sanitize_words(words)

    # Edge Case 1: Missing or empty transcript
    if not clean_words:
        safe_end = max(hard_min_end, min(orig_end, hard_max_end))
        return RefinementResult(
            end=round(safe_end, 2),
            original_end=orig_end,
            changed=abs(safe_end - orig_end) > 0.02,
            score=0.0,
            reason="missing_transcript_preserved",
            audio_status="unavailable" if audio is None else "insufficient_context",
            audio_validated=False,
            details={"fallback": True},
        )

    # Evaluate the quality of the actual original endpoint directly
    baseline_score, baseline_details = evaluate_baseline_endpoint(
        original_end=orig_end,
        start=start,
        words=clean_words,
        min_len=min_len,
        max_len=max_len,
        audio=audio,
        audio_offset=audio_offset,
    )

    # Calculate search interval around original end
    search_min = max(hard_min_end, orig_end - pre_window)
    search_max = min(hard_max_end, orig_end + post_window)

    if search_min >= search_max:
        safe_end = min(hard_max_end, max(hard_min_end, orig_end))
        return RefinementResult(
            end=round(safe_end, 2),
            original_end=orig_end,
            changed=abs(safe_end - orig_end) > 0.02,
            score=baseline_score,
            reason="search_window_collapsed_by_constraints",
            audio_status="unavailable" if audio is None else "insufficient_context",
            audio_validated=False,
            details={"hard_min_end": hard_min_end, "hard_max_end": hard_max_end},
        )

    # Find candidate word indices within search range
    candidates: List[Tuple[int, float, Dict[str, Any]]] = []
    for idx, w in enumerate(clean_words):
        we = float(w["e"])
        if search_min - 0.2 <= we <= search_max + 0.2:
            score, details = score_candidate_boundary(
                word_idx=idx,
                words=clean_words,
                original_end=orig_end,
                start=start,
                min_len=min_len,
                max_len=max_len,
                audio=audio,
                audio_offset=audio_offset,
            )
            if score > -500.0:  # Valid within duration limits
                candidates.append((idx, score, details))

    # Edge Case 2: No valid word boundaries in range
    if not candidates:
        safe_end = min(hard_max_end, max(hard_min_end, orig_end))
        return RefinementResult(
            end=round(safe_end, 2),
            original_end=orig_end,
            changed=abs(safe_end - orig_end) > 0.02,
            score=baseline_score,
            reason="no_candidates_in_range_preserved",
            audio_status="unavailable" if audio is None else "insufficient_context",
            audio_validated=False,
            details={},
        )

    # Sort candidates by score descending, then by distance to original_end ascending
    candidates.sort(key=lambda c: (-c[1], abs(c[2]["dist"])))
    best_idx, best_score, best_details = candidates[0]

    # Adoption Logic:
    # We adopt a new boundary ONLY if:
    # 1. Baseline was cut mid-word, OR
    # 2. Baseline ended on an incomplete hanging word, OR
    # 3. Candidate provides a substantial improvement over baseline (+18 score)
    must_fix_baseline = baseline_details.get("cut_midword") or baseline_details.get("hanging")
    significant_improvement = (best_score - baseline_score) >= 18.0

    should_adopt = must_fix_baseline or significant_improvement

    if not should_adopt:
        # Keep actual original endpoint (clamped cleanly to hard limits)
        safe_orig = max(hard_min_end, min(orig_end, hard_max_end))
        audio_stat = baseline_details.get("audio_status", "unavailable")
        return RefinementResult(
            end=round(safe_orig, 2),
            original_end=orig_end,
            changed=abs(safe_orig - orig_end) > 0.02,
            score=baseline_score,
            reason="preserved_original_sufficiently_natural",
            audio_status=audio_stat,
            audio_validated=audio_stat == "validated",
            details=baseline_details,
        )

    # Candidate boundary adopted
    chosen_end = float(clean_words[best_idx]["e"])
    next_w = clean_words[best_idx + 1] if best_idx + 1 < len(clean_words) else None
    if next_w:
        max_pad = max(0.0, float(next_w["s"]) - chosen_end - 0.05)
        added_pad = min(pad_silence, max_pad)
    else:
        added_pad = pad_silence

    final_end = chosen_end + added_pad
    final_end = max(hard_min_end, min(final_end, hard_max_end))
    final_end = round(final_end, 2)

    if final_end > orig_end + 0.1:
        reason = f"extended_to_complete_sentence_at_'{best_details['word']}'"
    elif final_end < orig_end - 0.1:
        reason = f"pulled_back_to_clean_boundary_at_'{best_details['word']}'"
    else:
        reason = "aligned_to_clean_word_boundary"

    audio_stat = best_details.get("audio_status", "unavailable")

    return RefinementResult(
        end=final_end,
        original_end=orig_end,
        changed=abs(final_end - orig_end) > 0.02,
        score=best_score,
        reason=reason,
        audio_status=audio_stat,
        audio_validated=audio_stat == "validated",
        details=best_details,
    )
