import numpy as np

def synthesize_represented_sound(modes, fs, duration):
    """
    Synthesizes the time-domain signal from modal parameters (Eq. 2 in paper).
    """
    t = np.arange(0, duration, 1/fs)
    signal = np.zeros_like(t)
    
    for m in modes:
        f = m["f"]
        d = m["d"]
        A = m["A"]
        
        # Angular frequency
        omega = 2 * np.pi * f
        
        # Generate decaying sinusoid
        # Note: A is the amplitude from the power spectrogram, 
        # so we take sqrt(A) for the pressure signal.
        component = A * np.exp(-d * t) * np.sin(omega * t)
        signal += component
        
    # Normalize to prevent clipping
    if np.max(np.abs(signal)) > 0:
        signal = signal / np.max(np.abs(signal))
        
    return signal, t