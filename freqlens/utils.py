from typing import Tuple

import librosa
import numpy as np
from pydub import AudioSegment


def normalize_loudness(track_1_path: str, track_2_path: str) -> Tuple[AudioSegment, AudioSegment]:
    """
    Normalize the loudness of two tracks so that they can be compared fairly.
    :param track_1_path: Path to your audio file.
    :param track_2_path: Path to the reference audio file.
    """
    # Load the tracks using pydub
    track_1 = AudioSegment.from_file(track_1_path)
    track_2 = AudioSegment.from_file(track_2_path)

    # Calculate RMS values
    rms_1 = track_1.rms
    rms_2 = track_2.rms

    # Normalize the reference track to match the RMS of your track
    gain = 10 * ((rms_1 / rms_2) - 1)
    normalized_track_2 = track_2.apply_gain(gain)

    return track_1, normalized_track_2


def resample_audio(signal: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resample an audio signal to a target sample rate."""
    return librosa.resample(signal, orig_sr=orig_sr, target_sr=target_sr)
