import numpy as np
from load_audio import load_audio
from modal_analysis import ModalFeatureExtractor
from diagnostics import ModalDiagnostics
from material_estimation import estimate_material_parameters # New file suggested above
from material_estimation import estimate_material_parameters_robust
from material_estimation import estimate_material_linear_regression
from synthesis import synthesize_represented_sound
import soundfile as sf # or your preferred wav saver
from utility import log_data_to_json


# ------------------------------------------------------
# 1. Load audio
# ------------------------------------------------------
filename ="wood_paper"
signal, fs = load_audio(f"{filename}.wav")

# ------------------------------------------------------
# 2. Extract modal features (Greedy Multi-Resolution)
# ------------------------------------------------------
extractor = ModalFeatureExtractor(
    fs,
    n_fft_list=[512, 1024, 2048, 4096, 8192],
    max_modes=40,
    energy_threshold=0.1 
)

validated_modes = extractor.extract_features(signal)

# ------------------------------------------------------
# 3. Print Extraction Summary
# ------------------------------------------------------
res_counts = {n: 0 for n in extractor.n_fft_list}
for mode in validated_modes:
    res_counts[mode['res_n_fft']] += 1

print("\n--- Extraction Summary (Greedy Multi-Res) ---")
for res, count in res_counts.items():
    print(f"FFT Size {res:5}: {count} modes extracted")
print(f"Total validated modes: {len(validated_modes)}")


validated_modes = [m for m in validated_modes if m['f'] < 2000]
# ------------------------------------------------------
# 4. Material Parameter Estimation (Section 4.2 of Paper)
# ------------------------------------------------------
# This finds the Alpha and Beta that define "Wood" damping

alpha, beta = estimate_material_linear_regression(validated_modes)
print(f"\n--- Estimated Material Parameters ---")
print(f"Alpha (Mass Damping): {alpha:.4f}")
print(f"Beta (Stiffness Damping): {beta:.8f}")

# ------------------------------------------------------
# 5. Diagnostics and Visual Verification
# ------------------------------------------------------
diag = ModalDiagnostics()

# A: Check the distribution of frequency and damping
diag.plot_mode_statistics(validated_modes)

# B: Visual check: Do the extracted modes align with the recording?
# We use the highest resolution (8192) for the background spectrogram
spectrograms = extractor.compute_multires_stft(signal)
diag.plot_spectrogram_with_modes(spectrograms[8192], fs, 8192, validated_modes)

# C: Physical check: Do the modes follow the Rayleigh curve?
diag.plot_rayleigh_fit(validated_modes, alpha, beta)

# D: Energy Decay
diag.plot_energy_decay(signal, fs)


# ------------------------------------------------------
# 6. Generate Represented Sound
# ------------------------------------------------------
duration = len(signal) / fs # Keep same duration as original
rep_signal, t = synthesize_represented_sound(validated_modes, fs, duration)

# Save for comparison
sf.write(f"synthesis_{filename}.wav", rep_signal, fs)
print("\n--- Represented Sound Generated ---")
print(f"Saved to: synthesis_{filename}.wav")

# ------------------------------------------------------
# 7. Diagnostics: Compare Original vs. Represented
# ------------------------------------------------------
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(t[:len(signal)], signal, label="Original Recording", alpha=0.7)
plt.legend()
plt.subplot(2, 1, 2)
plt.plot(t, rep_signal, label="Represented Sound (Modal)", color='orange', alpha=0.7)
plt.legend()
plt.title("Waveform Comparison: Real vs. Modal")
plt.savefig(f"synthesis_{filename}.png")
plt.show()

log_data_to_json("wood_modal_data.json", validated_modes, alpha, beta, fs)

print("DONE")
