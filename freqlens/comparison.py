from typing import List, Tuple, Dict

from analysis import load_audio, analyze_band


def compare_tracks(
    track_1_path: str, track_2_path: str, bands: List[Tuple[float, float]]
) -> List[Dict[str, Dict[str, float]]]:
    """
    Compare two audio tracks across the specified frequency bands.
    :param track_1_path: Path to the first audio file (your track).
    :param track_2_path: Path to the second audio file (reference track).
    :param bands: List of frequency bands to compare [(lowcut, highcut)].
    """
    # Load the audio files
    signal_1, sr_1 = load_audio(track_1_path)
    signal_2, sr_2 = load_audio(track_2_path)

    # Ensure sample rates match
    if sr_1 != sr_2:
        raise ValueError("Sample rates of the two tracks must match.")

    comparison_results: List[Dict[str, Dict[str, float]]] = []

    # Compare across frequency bands
    for lowcut, highcut in bands:
        analysis_1 = analyze_band(signal_1, sr_1, lowcut, highcut)
        analysis_2 = analyze_band(signal_2, sr_1, lowcut, highcut)

        result: Dict[str, Dict[str, float]] = {
            "band": {"lowcut": lowcut, "highcut": highcut},
            "track_1": analysis_1,
            "track_2": analysis_2,
        }

        comparison_results.append(result)

    return comparison_results


def display_comparison(results: List[Dict[str, Dict[str, float]]]) -> None:
    """Display comparison results between two tracks."""
    for result in results:
        band = result["band"]
        print(f"Band {band['lowcut']}Hz - {band['highcut']}Hz:")
        print(
            f"  Track 1 - RMS: {result['track_1']['rms']:.2f}, Peak: {result['track_1']['peak']:.2f}, "
            f"Dominant Freq: {result['track_1']['dominant_freq']:.2f}Hz"
        )
        print(
            f"  Track 2 - RMS: {result['track_2']['rms']:.2f}, Peak: {result['track_2']['peak']:.2f}, "
            f"Dominant Freq: {result['track_2']['dominant_freq']:.2f}Hz"
        )
        print()
