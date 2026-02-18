import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt

def get_material_frequencies(E, rho, L=1.0, num_modes=5):
    """
    Analytic modal frequencies for a clamped-free bar (Simplified Physics).
    f = (n * sqrt(E/rho)) / (2 * L)
    """
    gamma = np.sqrt(E / rho)
    indices = np.arange(1, num_modes + 1)
    frequencies = (indices * gamma) / (2 * L)
    return frequencies

# If you increase Young's Modulus (E), the frequency (pitch) goes up.
# If you increase Density (rho), the frequency goes down.

def generate_modal_sound(frequencies, dampings, amplitudes, duration=2.0, sample_rate=44100):
    """
    Synthesizes sound from modal parameters.
    """
    t = np.linspace(0, duration, int(sample_rate * duration))
    sound = np.zeros_like(t)
    
    for f, d, a in zip(frequencies, dampings, amplitudes):
        # The core modal synthesis formula: a * e^(-d*t) * sin(2*pi*f*t)
        mode = a * np.exp(-d * t) * np.sin(2 * np.pi * f * t)
        sound += mode
        
    # Normalize sound to avoid clipping
    sound = sound / np.max(np.abs(sound))
    return (sound * 32767).astype(np.int16)

# --- Define some "Material" properties (Example: Steel-like) ---
# In Step 2, these will come from your FEM/Eigenvalue solver
freqs = [440, 885, 1320, 2100]  # Hz
damps = [2.0, 4.5, 7.0, 12.0]   # Decay rates (higher = faster decay)
amps = [1.0, 0.6, 0.4, 0.2]     # Initial strike energy

# Generate and save
audio_data = generate_modal_sound(freqs, damps, amps)
plt.plot(audio_data)
plt.show()
wavfile.write("modal_test.wav", 44100, audio_data)
print("Sound generated: modal_test.wav")