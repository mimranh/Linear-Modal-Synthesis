import numpy as np
import librosa
from scipy.signal import find_peaks
from scipy.optimize import least_squares


class ModalFeatureExtractor:

    def __init__(self, fs,
                 n_fft_list=[512, 1024, 2048, 4096],
                 max_modes=20,
                 energy_threshold=1e-2):

        self.fs = fs
        self.n_fft_list = n_fft_list
        self.max_modes = max_modes
        self.energy_threshold = energy_threshold

    # ----------------------------------------------------------
    # MULTI-RESOLUTION STFT
    # ----------------------------------------------------------

    def compute_multires_stft(self, signal):
        spectrograms = []
        for n_fft in self.n_fft_list:
            S = librosa.stft(signal, n_fft=n_fft,
                             hop_length=n_fft // 4)
            P = np.abs(S) ** 2
            spectrograms.append((n_fft, P))
        return spectrograms

    # ----------------------------------------------------------
    # PEAK DETECTION
    # ----------------------------------------------------------

    def detect_global_peak(self, P):
        return np.unravel_index(np.argmax(P), P.shape)

    # ----------------------------------------------------------
    # PATCH EXTRACTION
    # ----------------------------------------------------------

    def extract_local_patch(self, P, k0, m0,
                            t_win=25, f_win=15):

        m_min = max(0, m0 - t_win)
        m_max = min(P.shape[1], m0 + t_win)

        k_min = max(0, k0 - f_win)
        k_max = min(P.shape[0], k0 + f_win)

        patch = P[k_min:k_max, m_min:m_max]

        return patch, m_min, k_min

    # ----------------------------------------------------------
    # 2D HILL MODEL (LOG DOMAIN)
    # ----------------------------------------------------------

    def hill_residual(self, params, T, F, Y_obs):

        f0, d, logA, sigma_f = params

        model = (
            logA
            - 2 * d * T
            - (F - f0) ** 2 / (2 * sigma_f ** 2)
        )

        return (model - Y_obs).ravel()

    # ----------------------------------------------------------
    # HILL FITTING
    # ----------------------------------------------------------

    def fit_hill(self, patch, m_offset, k_offset, n_fft):

        epsilon = 1e-10
        Y_obs = np.log(patch + epsilon)

        n_freq, n_time = patch.shape

        hop = n_fft // 4
        t_vals = np.arange(n_time) * hop / self.fs
        freqs = np.fft.rfftfreq(n_fft, 1 / self.fs)

        f_vals = freqs[k_offset:k_offset + n_freq]

        T, F = np.meshgrid(t_vals, f_vals)

        # Initial guesses
        f0_init = f_vals[np.argmax(np.sum(patch, axis=1))]
        d_init = 50
        logA_init = np.max(Y_obs)
        sigma_init = 50

        x0 = [f0_init, d_init, logA_init, sigma_init]

        result = least_squares(
            self.hill_residual,
            x0,
            args=(T, F, Y_obs),
            bounds=(
                [0, 0, -np.inf, 1],
                [self.fs / 2, 5000, np.inf, 2000]
            )
        )

        f0, d, logA, sigma_f = result.x

        return {
            "f": f0,
            "d": d,
            "A": np.exp(logA),
            "sigma_f": sigma_f,
            "cost": result.cost
        }

    # ----------------------------------------------------------
    # MODEL SUBTRACTION
    # ----------------------------------------------------------

    def subtract_model(self, P, feature, n_fft):

        hop = n_fft // 4
        n_freq, n_time = P.shape

        t_vals = np.arange(n_time) * hop / self.fs
        freqs = np.fft.rfftfreq(n_fft, 1 / self.fs)

        T, F = np.meshgrid(t_vals, freqs)

        model = (
            feature["A"]
            * np.exp(-2 * feature["d"] * T)
            * np.exp(-(F - feature["f"]) ** 2
                     / (2 * feature["sigma_f"] ** 2))
        )

        P_new = P - model
        P_new[P_new < 0] = 0

        return P_new

    # ----------------------------------------------------------
    # CONFIDENCE METRIC
    # ----------------------------------------------------------

    def compute_confidence(self, patch, feature):

        total_energy = np.sum(patch)

        # Reconstruct fitted hill
        n_freq, n_time = patch.shape
        t_vals = np.arange(n_time)
        f_vals = np.arange(n_freq)

        # crude approximation
        reconstructed = feature["A"]

        confidence = 1 - feature["cost"] / (total_energy + 1e-10)

        return confidence

    # ----------------------------------------------------------
    # GREEDY MULTI-MODE EXTRACTION
    # ----------------------------------------------------------
    def extract_features(self, signal):

        features_per_resolution = []

        spectrograms = self.compute_multires_stft(signal)
        

        for n_fft, P in spectrograms:

            P_work = P.copy()
            resolution_features = []

            initial_energy = np.sum(P_work)

            for _ in range(self.max_modes):

                if np.sum(P_work) / initial_energy < 0.15:
                    break

                current_ratio = np.sum(P_work) / initial_energy
                print("Energy ratio:", current_ratio)
            

                k0, m0 = self.detect_global_peak(P_work)

                patch, m_offset, k_offset = \
                    self.extract_local_patch(P_work, k0, m0)

                feature = self.fit_hill(
                    patch,
                    m_offset,
                    k_offset,
                    n_fft
                )

                confidence = self.compute_confidence(
                    patch,
                    feature
                )

                if confidence < 0.5:
                    break

                resolution_features.append(feature)

                P_work = self.subtract_model(
                    P_work,
                    feature,
                    n_fft
                )

            features_per_resolution.append(resolution_features)

        return features_per_resolution