import numpy as np
import librosa
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from sklearn.linear_model import LinearRegression

def extract_real_world_features(wav_path, n_modes=6, label="", pltshow=False):
    """
    Extracts modal frequencies, damping (decay rates), and initial amplitudes 
    from a recording. Optimized for short-duration wood impacts.
    """
    # 1. Load and Clean Audio
    y, sr = librosa.load(wav_path)
    # Pre-emphasis helps separate peaks in the high-frequency range
    y = librosa.effects.preemphasis(y) 
    
    # 2. STFT for Time-Frequency Analysis
    n_fft = 4096
    S = np.abs(librosa.stft(y, n_fft=n_fft))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    times = librosa.frames_to_time(np.arange(S.shape[1]), sr=sr)
    
    # 3. Peak Picking (Identifying the Modes)
    # Look at the first 50-80ms where the impact energy is highest
    mean_spec = np.mean(S[:, :8], axis=1) 
    peak_indices, _ = find_peaks(mean_spec, distance=20, prominence=0.01)
    
    # Take the top N peaks based on initial energy
    top_peak_indices = peak_indices[np.argsort(mean_spec[peak_indices])][-n_modes:]
    
    # CRITICAL: Sort by frequency so target_f[0] matches target_d[0] and target_a[0]
    sorted_indices = top_peak_indices[np.argsort(freqs[top_peak_indices])]
    
    target_f = freqs[sorted_indices]
    target_a = mean_spec[sorted_indices] 
    target_d = []
    
    # 4. Decay Estimation (Log-Slope Analysis)
    # Based on your waveform, the sound dies out in ~0.08s.
    # We use a very tight window (indices 2 to 6) to avoid the noise floor.
    for p_idx in sorted_indices:
        energy = np.log(S[p_idx, :] + 1e-8)
        
        # Fit a line to the immediate decay following the impact
        # times[2:7] typically covers roughly the 40ms to 120ms range
        X_decay = times[2:7].reshape(-1, 1)
        y_decay = energy[2:7]
        
        model = LinearRegression()
        model.fit(X_decay, y_decay)
        
        # d = -slope of the energy decay
        decay_val = -model.coef_[0]
        
        # Physical floor: Even resonant wood doesn't ring forever.
        # If the recording is noisy, we default to a realistic wood value.
        target_d.append(max(decay_val, 20.0))
        
    # --- 3D SPECTROGRAM VISUALIZATION ---
    if pltshow:
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Downsample for visualization performance
        S_db = librosa.amplitude_to_db(S, ref=np.max)
        X, Y = np.meshgrid(times[:50], freqs[:500]) # Zoom into low-mid range
        Z = S_db[:500, :50]
        
        surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=0.8)
        
        # Highlight extracted peaks with red dots in 3D
        for i in range(len(target_f)):
            ax.scatter(times[0], target_f[i], 0, color='red', s=50, label='Extracted Mode' if i==0 else "")

        ax.set_title(f"Modal Decay Spectrogram: {label}")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Frequency (Hz)")
        ax.set_zlabel("Magnitude (dB)")
        plt.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
        plt.show()

    return target_f, np.array(target_d), np.array(target_a)