from typing import Tuple, Dict

import librosa
import numpy as np
from scipy.signal import butter, lfilter


def load_audio(filepath: str, sr: int = 44100) -> Tuple[np.ndarray, int]:
    """Load an audio file."""
    audio, sample_rate = librosa.load(filepath, sr=sr)
    return audio, sample_rate


def calculate_rms(signal: np.ndarray) -> float:
    """Calculate RMS (Root Mean Square) of the signal."""
    return float(np.sqrt(np.mean(np.square(signal))))


def calculate_peak(signal: np.ndarray) -> float:
    """Calculate peak value of the signal."""
    return float(np.max(np.abs(signal)))


def bandpass_filter(signal: np.ndarray, lowcut: float, highcut: float, sr: int, order: int = 5) -> np.ndarray:
    """Apply a bandpass filter to isolate frequency ranges."""
    nyquist = 0.5 * sr
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype="band")
    return lfilter(b, a, signal)


def dominant_frequencies(
    signal: np.ndarray, sr: int, lowcut: float, highcut: float, n_fft: int = 2048
) -> Tuple[float, float]:
    """Identify the dominant and least dominant frequencies within a band."""
    fft = np.abs(np.fft.rfft(signal, n=n_fft))
    freqs = np.fft.rfftfreq(n_fft, 1 / sr)

    band_mask = (freqs >= lowcut) & (freqs <= highcut)
    dominant_freq = freqs[band_mask][np.argmax(fft[band_mask])]
    least_dominant_freq = freqs[band_mask][np.argmin(fft[band_mask])]

    return float(dominant_freq), float(least_dominant_freq)


def analyze_band(signal: np.ndarray, sr: int, lowcut: float, highcut: float) -> Dict[str, float]:
    """Analyze an audio signal for a specific frequency band."""
    filtered_signal = bandpass_filter(signal, lowcut, highcut, sr)
    rms = calculate_rms(filtered_signal)
    peak = calculate_peak(filtered_signal)
    dominant_freq, least_dominant_freq = dominant_frequencies(filtered_signal, sr, lowcut, highcut)

    return {"rms": rms, "peak": peak, "dominant_freq": dominant_freq, "least_dominant_freq": least_dominant_freq}
