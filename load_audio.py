import numpy as np
import librosa


def load_audio(filepath,
               target_fs=None,
               normalize=True,
               remove_dc=True,
               trim_silence=True):

    # ------------------------------------------------------
    # Load audio
    # ------------------------------------------------------
    signal, fs = librosa.load(
        filepath,
        sr=target_fs,
        mono=True
    )

    # ------------------------------------------------------
    # Remove DC offset
    # ------------------------------------------------------
    if remove_dc:
        signal = signal - np.mean(signal)

    # ------------------------------------------------------
    # Trim silence (important for impact sounds)
    # ------------------------------------------------------
    if trim_silence:
        signal, _ = librosa.effects.trim(
            signal,
            top_db=40
        )

    # ------------------------------------------------------
    # Normalize
    # ------------------------------------------------------
    if normalize:
        max_val = np.max(np.abs(signal))
        if max_val > 0:
            signal = signal / max_val

    return signal, fs