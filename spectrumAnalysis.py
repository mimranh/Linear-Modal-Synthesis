import numpy as np
import librosa
import matplotlib.pyplot as plt

def compare_spectra(original_path, synthesis_path, label="Wood Door"):
    # 1. Load both audios
    y_org, sr_org = librosa.load(original_path)
    y_syn, sr_syn = librosa.load(synthesis_path)
    
    # 2. Compute FFT (Magnitude Spectrum)
    # We use a high n_fft for fine frequency resolution
    n_fft = 8192
    # Use 'n' instead of 'n_fft' for numpy.fft.rfft
    fft_org = np.abs(np.fft.rfft(y_org, n=n_fft))
    fft_syn = np.abs(np.fft.rfft(y_syn, n=n_fft))
    freqs = np.fft.rfftfreq(n_fft, 1/sr_org)
    
    # Convert to Decibels for better visualization of overtones
    db_org = librosa.amplitude_to_db(fft_org, ref=np.max)
    db_syn = librosa.amplitude_to_db(fft_syn, ref=np.max)
    
    # 3. Plotting
    plt.figure(figsize=(12, 6))
    plt.plot(freqs, db_org, label='Original Recording (Target)', alpha=0.7, color='blue')
    plt.plot(freqs, db_syn, label='Synthesized (FEM + Rayleigh)', alpha=0.7, color='red', linestyle='--')
    
    # Focus on the audible range where your peaks were found (0 - 12kHz)
    plt.xlim(0, 12000)
    plt.ylim(-60, 0) # Focus on top 60dB
    
    plt.title(f"Frequency Spectrum Comparison: {label}")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (dB)")
    plt.legend()
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(f"spectrum_comparison_{label}.png")
    plt.show()

# Run it:
if __name__ == "__main__":
    compare_spectra("wood_door.wav", "synthesized_Real_Door_Knock.wav")