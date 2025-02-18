import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from scipy.signal import correlate
import soundfile as sf


def load_audio(file_path, sr=8000):
    signal, sr = librosa.load(file_path, sr=sr, mono=True)
    return signal, sr


def normalize_audio(signal):
    return (signal - np.mean(signal)) / np.std(signal)


def compute_cross_correlation(watermark, voice):
    """
    Compute the cross-correlation between the watermark and the voice.
    """
    cross_corr = correlate(voice, watermark, mode="full", method="fft")
    lags = np.arange(-len(voice) + 1, len(watermark))
    return cross_corr, lags


def find_possible_watermark_start(watermark_signal, voice_signal, top_k=2, sr=8000):
    # Normalize the signals
    watermark_signal = normalize_audio(watermark_signal)
    voice_signal = normalize_audio(voice_signal)

    # Compute cross-correlation
    cross_corr, lags = compute_cross_correlation(
        watermark_signal, voice_signal)

    possible_start_times = []

    for iter_idx in range(top_k):
        # Find the best match (maximum cross-correlation value and corresponding lag)
        max_corr_idx = np.argmax(cross_corr)
        max_corr_lag = lags[max_corr_idx]

        start_watermark_time = (
            max_corr_lag + len(voice_signal) - len(watermark_signal)) / sr

        possible_start_times.append(start_watermark_time)

        # clean max area
        cross_corr[max_corr_idx - len(watermark_signal)                   : max_corr_idx + len(watermark_signal)] = 0
    print('start', possible_start_times)
    return possible_start_times


def main(watermark_path, voice_path, iter_count=2, sr=8000):
    # Load and preprocess the audio
    watermark_signal, _ = load_audio(watermark_path, sr)
    voice_signal_original, _ = load_audio(voice_path, sr=sr)

    # Normalize the signals
    watermark_signal = normalize_audio(watermark_signal)
    voice_signal = normalize_audio(voice_signal_original)

    # Compute cross-correlation
    cross_corr, lags = compute_cross_correlation(
        watermark_signal, voice_signal)

    possible_start_end_pairs = []

    for iter_idx in range(iter_count):
        # Find the best match (maximum cross-correlation value and corresponding lag)
        max_corr_idx = np.argmax(cross_corr)
        max_corr_lag = lags[max_corr_idx]

        end_watermark_time = (max_corr_lag + len(voice_signal)) / sr
        start_watermark_time = end_watermark_time - len(watermark_signal) / sr

        possible_start_end_pairs.append(
            (start_watermark_time, end_watermark_time))
        # Visualize the cross-correlation
        cross_corr[max_corr_idx - len(watermark_signal)                   : max_corr_idx + len(watermark_signal)] = 0

    return possible_start_end_pairs


if __name__ == '__main__':
    watermarks_paths = ['watermark_1.mp3', 'watermark_2.mp3']
    for watermark in watermarks_paths:
        print(main(watermark, './1/Goblin.mp4', iter_count=2))
