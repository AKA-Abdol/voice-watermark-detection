import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from scipy.signal import correlate
import soundfile as sf

def load_audio(file_path, sr=16000):
    """
    Load an audio file and return the signal and sampling rate.
    """
    signal, sr = librosa.load(file_path, sr=sr, mono=True)
    return signal, sr

def normalize_audio(signal):
    """
    Normalize the audio signal to have zero mean and unit variance.
    """
    return (signal - np.mean(signal)) / np.std(signal)

def compute_cross_correlation(watermark, voice):
    """
    Compute the cross-correlation between the watermark and the voice.
    """
    cross_corr = correlate(voice, watermark, mode="full", method="fft")
    lags = np.arange(-len(voice) + 1, len(watermark))
    return cross_corr, lags

def plot_cross_correlation(cross_corr, lags, sr):
    """
    Visualize the cross-correlation.
    """
    # Convert lags to time
    lag_times = lags / sr
    plt.figure(figsize=(10, 6))
    plt.plot(lag_times, cross_corr)
    plt.title("Cross-Correlation")
    plt.xlabel("Lag (seconds)")
    plt.ylabel("Cross-Correlation Coefficient")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
def estimate_gain(main_audio_segment, watermark):
    """
    Estimate the gain factor to match the watermark's amplitude in the main audio.
    """
    watermark_energy = np.sum(watermark ** 2)
    segment_energy = np.sum(main_audio_segment ** 2)
    gain = np.sqrt(segment_energy / watermark_energy)
    return gain

def remove_watermark(voice_signal, watermark_signal, max_corr_idx, sr):
    """
    Replace the watermark in the voice signal with silence.
    """
    
    # Calculate start and end points of the watermark in the voice signal
    start_sample = max(max_corr_idx - len(watermark_signal), 0)
    end_sample = max_corr_idx
    
    # Perform STFT on the audio signal
    D = librosa.stft(voice_signal)
    print(D.shape, start_sample, end_sample)
    print(len(voice_signal) / 512)
    
    start_time_D = int(start_sample / 512)
    end_time_D = int(end_sample / 512)
    
    # Zero out the watermark in the frequency domain
    D[:, start_time_D:end_time_D] = 0
    
    # Perform inverse STFT to convert back to the time domain
    y_clean = librosa.istft(D)
    print(y_clean.shape)
    
    return y_clean

# Paths to the audio files
watermark_idx = 1
file_idx = 1
watermark_path = f"watermark_{watermark_idx}.mp3"  # Replace with the actual file path
voice_path = f"{watermark_idx}/{file_idx}.mp3"         # Replace with the actual file path

# Load and preprocess the audio
watermark_signal, sr = load_audio(watermark_path)
print(sr)
voice_signal_original, _ = load_audio(voice_path, sr=sr)  # Ensure both signals have the same sampling rate

# Normalize the signals
watermark_signal = normalize_audio(watermark_signal)
voice_signal = normalize_audio(voice_signal_original)

# Compute cross-correlation
cross_corr, lags = compute_cross_correlation(watermark_signal, voice_signal)
print(lags)

# Visualize the cross-correlation
plot_cross_correlation(cross_corr, lags, sr)

# Find the best match (maximum cross-correlation value and corresponding lag)
max_corr_idx = np.argmax(cross_corr)
max_corr_lag = lags[max_corr_idx] / sr  # Convert lag to seconds
print(f"Maximum Cross-Correlation: {cross_corr[max_corr_idx]:.4f}")
print(f"Time of Best Match (seconds): {max_corr_lag:.2f}")

# Remove the watermark and replace it with silence
voice_signal_no_watermark = remove_watermark(voice_signal_original, watermark_signal, max_corr_idx, sr)

output_path = 'out.mp3'

# fig, (ax1, ax2) = plt.subplots(2, 1)
# ax1.plot(voice_signal)
# ax1.set_title('Source Voice')
# ax2.plot(voice_signal_no_watermark)
# ax2.set_title('No Watermark')
# plt.show()

# Save the modified audio
sf.write(output_path, voice_signal_no_watermark, sr)
print(f"Watermark removed. Output saved to {output_path}")