"""audio_video_fusion.py — Audio Impact Detection + Hysteresis State Machine

Implements the audio side of the audio-video fusion design (see the project doc), plus the
WAITING / POINT_ACTIVE state machine that turns a fused active_score time series into rally
boundaries. Player-motion and ball-activity scores are NOT computed here — they're produced by
existing modules (RobustKinematicDetector in main.py, and the ball tracker) and fused with the
audio score in BatchTennisPipeline.process_fusion_clip().

Deviations from the doc's original v1 design, based on earlier discussion:
- Impact detection uses band-pass filtered (~1-4kHz) onset detection (rising-edge energy, not
  raw broadband energy level) to cut down on false positives from crowd noise, commentary, and
  low-frequency rumble — a sustained loud stretch won't score high, only sharp transients will.
- No new heavy dependency (no librosa) — built on scipy, which this project already depends on.
- Normalizes onset strength against a robust ceiling (95th percentile) rather than the raw max,
  so a single outlier transient doesn't compress the rest of the scale.
"""
import os
import subprocess
import tempfile

import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, sosfiltfilt


# ---------------------------------------------------------------------------
# Audio impact detection
# ---------------------------------------------------------------------------

def extract_audio_wav(video_path, sr=16000):
    """Extracts mono WAV audio from a video file via ffmpeg. Returns (sample_rate, samples)."""
    tmp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp_wav.close()
    command = ["ffmpeg", "-y", "-i", video_path, "-vn", "-ac", "1", "-ar", str(sr), "-f", "wav", tmp_wav.name]
    try:
        subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        rate, samples = wavfile.read(tmp_wav.name)
        samples = samples.astype(np.float32)
        if samples.ndim > 1:  # collapse stereo to mono, just in case -ac 1 didn't apply
            samples = samples.mean(axis=1)
        return rate, samples
    finally:
        if os.path.exists(tmp_wav.name):
            os.remove(tmp_wav.name)


def _bandpass_filter(samples, sr, low_hz=1000, high_hz=4000):
    """Bandpass filters audio to the frequency range typical of racket/ball impact transients."""
    nyq = sr / 2.0
    low = max(low_hz / nyq, 1e-4)
    high = min(high_hz / nyq, 0.999)
    sos = butter(4, [low, high], btype="band", output="sos")
    return sosfiltfilt(sos, samples)


def compute_impact_score_series(video_path, hop_sec=0.5, sr=16000):
    """Returns (times, impact_scores): impact_scores in [0,1] at `hop_sec` resolution.

    Derived from bandpass-filtered short-time RMS energy, then onset-ified by taking only the
    positive frame-to-frame *increase* in energy — a rising edge indicates an impact; a flat
    (even loud) stretch does not, unlike scoring on raw energy level directly.
    """
    rate, samples = extract_audio_wav(video_path, sr=sr)
    filtered = _bandpass_filter(samples, rate)

    hop_len = max(1, int(hop_sec * rate))
    n_hops = max(1, len(filtered) // hop_len)

    energy = np.zeros(n_hops)
    for i in range(n_hops):
        window = filtered[i * hop_len:(i + 1) * hop_len]
        energy[i] = np.sqrt(np.mean(window.astype(np.float64) ** 2) + 1e-9)

    onset = np.diff(energy, prepend=energy[0])
    onset = np.clip(onset, 0, None)

    ceiling = np.percentile(onset, 95) if np.any(onset) else 1.0
    ceiling = max(ceiling, 1e-6)
    impact_scores = np.clip(onset / ceiling, 0, 1)

    times = np.arange(n_hops) * hop_sec
    return times, impact_scores


def get_score_at(times, scores, t_sec):
    """Nearest-neighbor lookup of a precomputed score series at time t_sec."""
    if len(times) == 0:
        return 0.0
    idx = int(np.searchsorted(times, t_sec))
    idx = min(idx, len(times) - 1)
    return float(scores[idx])


# ---------------------------------------------------------------------------
# Fusion state machine
# ---------------------------------------------------------------------------

class RallyStateMachine:
    """WAITING / POINT_ACTIVE hysteresis state machine (per the fusion design doc's section 5.4).

    Requires active_score to stay above `enter_threshold` for `enter_dwell_sec` before switching
    WAITING -> POINT_ACTIVE, and below `exit_threshold` for `exit_dwell_sec` before switching back
    -- prevents single-sample noise from creating spurious rally boundaries.
    """

    def __init__(self, enter_threshold=0.55, exit_threshold=0.30,
                 enter_dwell_sec=1.5, exit_dwell_sec=2.5, sample_interval_sec=0.5):
        self.enter_threshold = enter_threshold
        self.exit_threshold = exit_threshold
        self.enter_dwell_samples = max(1, int(round(enter_dwell_sec / sample_interval_sec)))
        self.exit_dwell_samples = max(1, int(round(exit_dwell_sec / sample_interval_sec)))
        self.state = "WAITING"
        self._above_count = 0
        self._below_count = 0

    def update(self, active_score):
        """Feeds one new sample; returns the status string for this sample ("PLAYING (Rally)" or
        "NON-PLAYING (Break)"), matching the convention used elsewhere in main.py's timeline."""
        if self.state == "WAITING":
            self._above_count = self._above_count + 1 if active_score > self.enter_threshold else 0
            if self._above_count >= self.enter_dwell_samples:
                self.state = "POINT_ACTIVE"
                self._below_count = 0
        else:  # POINT_ACTIVE
            self._below_count = self._below_count + 1 if active_score < self.exit_threshold else 0
            if self._below_count >= self.exit_dwell_samples:
                self.state = "WAITING"
                self._above_count = 0

        return "PLAYING (Rally)" if self.state == "POINT_ACTIVE" else "NON-PLAYING (Break)"