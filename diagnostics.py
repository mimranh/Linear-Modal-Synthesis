import numpy as np
import matplotlib.pyplot as plt


class ModalDiagnostics:

    def plot_spectrogram_with_modes(self, P, fs, n_fft, modes):

        plt.figure(figsize=(10,6))
        plt.imshow(10*np.log10(P+1e-10),
                   aspect='auto',
                   origin='lower',
                   cmap='magma')

        freqs = np.fft.rfftfreq(n_fft, 1/fs)

        for m in modes:
            f_idx = np.argmin(np.abs(freqs - m["f"]))
            plt.axhline(f_idx, color='cyan', linestyle='--')

        plt.title("Spectrogram with Extracted Modes")
        plt.colorbar(label="dB")
        plt.show()


    def plot_hill_fit(self, patch, fitted_model):

        fig = plt.figure(figsize=(12,5))

        plt.subplot(1,2,1)
        plt.title("Observed Patch")
        plt.imshow(np.log(patch+1e-10),
                   aspect='auto',
                   origin='lower')

        plt.subplot(1,2,2)
        plt.title("Fitted Hill")
        plt.imshow(np.log(fitted_model+1e-10),
                   aspect='auto',
                   origin='lower')

        plt.show()


    def plot_mode_statistics(self, validated_modes):

        freqs = [m["f"] for m in validated_modes]
        dampings = [m["d"] for m in validated_modes]

        plt.figure(figsize=(10,4))

        plt.subplot(1,2,1)
        plt.title("Extracted Frequencies")
        plt.hist(freqs, bins=20)

        plt.subplot(1,2,2)
        plt.title("Extracted Dampings")
        plt.hist(dampings, bins=20)

        plt.show()


    def plot_energy_decay(self, signal, fs):

        energy = signal**2
        t = np.arange(len(signal))/fs

        plt.figure()
        plt.plot(t, 10*np.log10(energy+1e-10))
        plt.title("Signal Energy Decay")
        plt.xlabel("Time (s)")
        plt.ylabel("Energy (dB)")
        plt.show()

    def plot_rayleigh_fit(self, modes, alpha, beta):
        """
        Plots extracted modes vs the fitted Rayleigh damping curve.
        """
        f_extracted = np.array([m["f"] for m in modes])
        d_extracted = np.array([m["d"] for m in modes])
        
        # Sort for plotting the curve
        f_range = np.linspace(min(f_extracted), max(f_extracted), 500)
        w_range = 2 * np.pi * f_range
        # Rayleigh formula: d = 0.5 * (alpha/w + beta*w)
        d_curve = 0.5 * (alpha / w_range + beta * w_range)
        
        plt.figure(figsize=(8, 5))
        plt.scatter(f_extracted, d_extracted, color='red', label='Extracted Modes')
        plt.plot(f_range, d_curve, label='Rayleigh Fit (Material Property)')
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Damping (d)")
        plt.title("Modal Damping vs. Rayleigh Model")
        plt.legend()
        plt.show()