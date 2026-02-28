import numpy as np
import librosa
from scipy.optimize import least_squares

class ModalFeatureExtractor:
    def __init__(self, fs, n_fft_list=[512, 1024, 2048, 4096], 
                 max_modes=20, energy_threshold=0.15):
        self.fs = fs
        self.n_fft_list = n_fft_list
        self.max_modes = max_modes
        self.energy_threshold = energy_threshold

    def compute_multires_stft(self, signal):
        """Computes multiple power spectrograms as per Section 4."""
        spectrograms = {}
        for n_fft in self.n_fft_list:
            S = librosa.stft(signal, n_fft=n_fft, hop_length=n_fft // 4)
            P = np.abs(S) ** 2
            spectrograms[n_fft] = P
        return spectrograms

    def find_global_peak(self, spectrograms):
        """Searches across all resolution levels for the highest energy peak."""
        max_val = -1
        best_res = None
        best_idx = None
        
        for n_fft, P in spectrograms.items():
            # Use average spectral density as suggested in Section 4
            idx = np.unravel_index(np.argmax(P), P.shape)
            if P[idx] > max_val:
                max_val = P[idx]
                best_res = n_fft
                best_idx = idx
        return best_res, best_idx

    def hill_model(self, T, F, f0, d, A, sigma_f):
        """2D Hill model equation for damped sinusoids (Eqn 1 & Fig 3)."""
        return A * np.exp(-2 * d * T) * np.exp(-(F - f0)**2 / (2 * sigma_f**2))

    def fit_hill(self, P, k0, m0, n_fft):
        """Local shape fitting around a peak to refine (f, d, a)."""
        # Extract local 2D patch
        f_win, t_win = 10, 15
        k_min, k_max = max(0, k0-f_win), min(P.shape[0], k0+f_win)
        m_min, m_max = max(0, m0-t_win), min(P.shape[1], m0+t_win)
        patch = P[k_min:k_max, m_min:m_max]
        
        # Setup coordinates for fitting
        hop = n_fft // 4
        t_vals = (np.arange(m_min, m_max) * hop) / self.fs
        freqs = np.fft.rfftfreq(n_fft, 1/self.fs)
        f_vals = freqs[k_min:k_max]
        T, F = np.meshgrid(t_vals, f_vals)

        # Initial guesses
        f0_init = freqs[k0]
        d_init = 50.0 # Standard start for wood/plastic
        A_init = np.max(patch)
        sig_init = (freqs[1] - freqs[0]) * 2

        def residual(params):
            f0, d, A, sigma_f = params
            model = self.hill_model(T, F, f0, d, A, sigma_f)
            return (model - patch).ravel()

        res = least_squares(residual, [f0_init, d_init, A_init, sig_init],
                            bounds=([0, 0, 0, 1e-6], [self.fs/2, 2000, np.inf, 1000]))
        
        f_fit, d_fit, a_fit, sig_fit = res.x
        return {"f": f_fit, "d": d_fit, "A": a_fit, "sigma": sig_fit, "res_n_fft": n_fft}

    def subtract_from_all(self, spectrograms, feature):
        """Subtracts the fitted mode from EVERY resolution level."""
        for n_fft, P in spectrograms.items():
            hop = n_fft // 4
            freqs = np.fft.rfftfreq(n_fft, 1/self.fs)
            t_vals = (np.arange(P.shape[1]) * hop) / self.fs
            T, F = np.meshgrid(t_vals, freqs)
            
            model = self.hill_model(T, F, feature['f'], feature['d'], 
                                   feature['A'], feature['sigma'])
            P -= model
            np.clip(P, 0, None, out=P) # Ensure no negative energy

    def extract_features(self, signal):
        """The core Greedy Multi-Resolution Extraction Loop."""
        spectrograms = self.compute_multires_stft(signal)
        extracted_modes = []
        
        # Track initial total energy for thresholding
        initial_energy = sum(np.sum(P) for P in spectrograms.values())

        for i in range(self.max_modes):
            current_energy = sum(np.sum(P) for P in spectrograms.values())
            if current_energy / initial_energy < self.energy_threshold:
                break

            # 1. Global peak search across resolutions
            n_fft, (k0, m0) = self.find_global_peak(spectrograms)
            
            # 2. Local Hill Fitting
            feature = self.fit_hill(spectrograms[n_fft], k0, m0, n_fft)
            extracted_modes.append(feature)

            # 3. Subtract from ALL resolutions
            self.subtract_from_all(spectrograms, feature)
            
        return extracted_modes