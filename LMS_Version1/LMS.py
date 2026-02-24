import numpy as np
import librosa
import matplotlib.pyplot as plt
from scipy.linalg import eigh, pinvh
from scipy.signal import find_peaks
from sklearn.linear_model import LinearRegression
from mpl_toolkits.mplot3d import Axes3D
import scipy.io.wavfile as wav
from scipy.optimize import curve_fit
from scipy.optimize import minimize


class ModalMaterialStudio:
    def __init__(self, sr=44100):
        self.sr = sr
        self.calibrated_materials = {}

    def _trim_silence(self, y, threshold=0.01):
        """Removes leading silence so synthesis and original align."""
        mask = np.abs(y) > threshold
        return y[np.argmax(mask):]

    # --- PART 1: FEM ENGINE ---
    def solve_geometry(self, E, rho, L, width, thickness, num_elements=100):
        """Generates the 'Digital Twin' physics."""
        n_nodes = num_elements + 1
        total_dof = n_nodes * 2
        dx = L / num_elements
        I = (width * thickness**3) / 12
        A = width * thickness
        
        K = np.zeros((total_dof, total_dof))
        M = np.zeros((total_dof, total_dof))
        
        k_e = (E * I / dx**3) * np.array([
            [12, 6*dx, -12, 6*dx], [6*dx, 4*dx**2, -6*dx, 2*dx**2],
            [-12, -6*dx, 12, -6*dx], [6*dx, 2*dx**2, -6*dx, 4*dx**2]
        ])
        m_e = (rho * A * dx / 420) * np.array([
            [156, 22*dx, 54, -13*dx], [22*dx, 4*dx**2, 13*dx, -3*dx**2],
            [54, 13*dx, 156, -22*dx], [-13*dx, -3*dx**2, -22*dx, 4*dx**2]
        ])

        for i in range(num_elements):
            idx = i * 2
            K[idx:idx+4, idx:idx+4] += k_e
            M[idx:idx+4, idx:idx+4] += m_e

        eigvals, eigvecs = eigh(K, M)
        freqs_hz = np.sqrt(np.maximum(eigvals, 0)) / (2 * np.pi)
        valid = freqs_hz > 1.0 
        return freqs_hz[valid], eigvecs[:, valid], K, eigvals

    # --- PART 2: CALIBRATION ---

    def calibrate_from_audio(self, wav_path, material_name, n_modes=6, show_visuals=True):
        # --- 1. Data Loading and Silence Trimming ---
        # Ensures waveform alignment for the 'Initial Attack' comparison
        y_raw, _ = librosa.load(wav_path, sr=self.sr)
        y, _ = librosa.effects.trim(y_raw, top_db=30)
        
        S = np.abs(librosa.stft(y, n_fft=4096))
        freqs = librosa.fft_frequencies(sr=self.sr, n_fft=4096)
        times = librosa.frames_to_time(np.arange(S.shape[1]), sr=self.sr)
        
        # --- 2. IMPROVED MODE DETECTION (Diversity Filter) ---
        # Focus on the initial impact frames to catch fast-decaying high modes
        impact_spec = np.mean(S[:, :4], axis=1) 
        all_peaks, _ = find_peaks(impact_spec, prominence=0.005, distance=10)
        
        # Sort peaks by magnitude but filter for spectral spread
        sorted_indices = np.argsort(impact_spec[all_peaks])[::-1]
        selected_peaks = []
        
        for idx in sorted_indices:
            p = all_peaks[idx]
            if not selected_peaks:
                selected_peaks.append(p)
            else:
                # Anchor modes at least 400Hz apart to stabilize the Rayleigh fit
                if np.all(np.abs(freqs[p] - freqs[selected_peaks]) > 400):
                    selected_peaks.append(p)
            if len(selected_peaks) >= n_modes:
                break
                
        top_peaks = np.sort(selected_peaks)
        f_ref = freqs[top_peaks]
        
        # --- 3. Damping Extraction ---
        d_ref = []
        for p in top_peaks:
            energy = np.log(S[p, :] + 1e-8)
            # Calculate decay rate d from the log-magnitude slope
            model = np.polyfit(times[1:6], energy[1:6], 1)
            d_ref.append(max(-model[0], 10.0))
        d_ref = np.array(d_ref)

        # --- 4. PERCEPTUAL OPTIMIZATION (Eq 20 Logic) ---
        # Minimize Euclidean distance in the transformed log-log space
        def perceptual_loss(params, f_targets, d_targets):
            alpha, beta = params
            omega = 2 * np.pi * f_targets
            d_pred = (alpha / 2) + (beta * omega**2 / 2)
            
            # X(f) = log(f), Y(d) = log(d)
            x_target, y_target = np.log10(f_targets), np.log10(d_targets)
            x_pred, y_pred = np.log10(f_targets), np.log10(np.maximum(d_pred, 1e-3))
            
            # Euclidean distance in transformed space
            distance = np.sqrt((x_target - x_pred)**2 + (y_target - y_pred)**2)
            return np.sum(distance)

        res = minimize(perceptual_loss, x0=[50.0, 1e-5], args=(f_ref, d_ref),
                    bounds=[(1.0, 500.0), (1e-6, 1e-3)])
        
        alpha, beta = res.x
        
        # --- 5. Storage and Visualization ---
        self.calibrated_materials[material_name] = {
            "alpha": alpha, 
            "beta": beta, 
            "orig_y": y,
            "orig_spec": np.mean(S[:, :8], axis=1), 
            "orig_freqs": freqs,
            "f_ref": f_ref, 
            "d_ref": d_ref
        }

        if show_visuals:
            self._plot_calibration_results_with_transformation(
                times, freqs, S, f_ref, d_ref, alpha, beta, material_name
            )
        
        return f_ref, d_ref



    def calibrate_from_audio1(self, wav_path, material_name, n_modes=6, show_visuals=True):
        # --- 1. Load and Trim ---
        y_raw, _ = librosa.load(wav_path, sr=self.sr)
        y, _ = librosa.effects.trim(y_raw, top_db=30)
        
        S = np.abs(librosa.stft(y, n_fft=4096))
        freqs = librosa.fft_frequencies(sr=self.sr, n_fft=4096)
        times = librosa.frames_to_time(np.arange(S.shape[1]), sr=self.sr)
        
        # --- 2. Feature Extraction ---
        mean_spec = np.mean(S[:, :8], axis=1)
        peaks, _ = find_peaks(mean_spec, distance=20, prominence=0.01)
        top_peaks = peaks[np.argsort(mean_spec[peaks])][-n_modes:]
        top_peaks = top_peaks[np.argsort(freqs[top_peaks])]
        
        f_ref = freqs[top_peaks]
        d_ref = []
        
        for p in top_peaks:
            energy = np.log(S[p, :] + 1e-8)
            model = np.polyfit(times[1:6], energy[1:6], 1)
            d_ref.append(max(-model[0], 10.0))
        d_ref = np.array(d_ref)

        # --- 3. PERCEPTUAL OPTIMIZATION (Eq 20 Logic) ---
        def perceptual_loss(params, f_targets, d_targets):
            alpha, beta = params
            w = 2 * np.pi * f_targets
            d_pred = (alpha / 2) + (beta * w**2 / 2)
            
            # Transformation to 2D Space: x = log(f), y = log(d)
            x_target, y_target = np.log10(f_targets), np.log10(d_targets)
            x_pred, y_pred = np.log10(f_targets), np.log10(np.maximum(d_pred, 1e-3))
            
            # Euclidean distance D in the transformed space
            distance = np.sqrt((x_target - x_pred)**2 + (y_target - y_pred)**2)
            return np.sum(distance)

        # Minimize distance in the log-log space
        res = minimize(perceptual_loss, x0=[50.0, 1e-5], args=(f_ref, d_ref),
                    bounds=[(1.0, 500.0), (1e-9, 1e-3)])
        
        alpha, beta = res.x
        
        self.calibrated_materials[material_name] = {
            "alpha": alpha, "beta": beta, "orig_y": y,
            "orig_spec": mean_spec, "orig_freqs": freqs,
            "f_ref": f_ref, "d_ref": d_ref
        }

        if show_visuals:
            self._plot_calibration_results_with_transformation(times, freqs, S, f_ref, d_ref, alpha, beta, material_name)
        
        return f_ref, d_ref


    def linear_calibrate_from_audio(self, wav_path, material_name, n_modes=6, show_visuals=True):
        # --- Data Loading and Trimming ---
        y_raw, _ = librosa.load(wav_path, sr=self.sr)
        y, _ = librosa.effects.trim(y_raw, top_db=30)
        
        S = np.abs(librosa.stft(y, n_fft=4096))
        freqs = librosa.fft_frequencies(sr=self.sr, n_fft=4096)
        times = librosa.frames_to_time(np.arange(S.shape[1]), sr=self.sr)
        
        mean_spec = np.mean(S[:, :8], axis=1)
        peaks, _ = find_peaks(mean_spec, distance=20, prominence=0.01)
        top_peaks = peaks[np.argsort(mean_spec[peaks])][-n_modes:]
        top_peaks = top_peaks[np.argsort(freqs[top_peaks])]
        
        f_ref = freqs[top_peaks]
        omega_ref = 2 * np.pi * f_ref
        d_ref = []
        
        for p in top_peaks:
            energy = np.log(S[p, :] + 1e-8)
            model = np.polyfit(times[1:6], energy[1:6], 1)
            d_ref.append(max(-model[0], 10.0))
        d_ref = np.array(d_ref)

        # --- THE TRANSFORMATION (Paper Fig 6-b Logic) ---
        # We transform the data to a space where Rayleigh is a linear function:
        # y_trans = 2d / omega^2, x_trans = 1 / omega^2
        # This allows us to find alpha (intercept) and beta (slope) more stably.
        def rayleigh_linear(f, alpha, beta):
            w = 2 * np.pi * f
            return (alpha / 2) + (beta * w**2 / 2)

        popt, _ = curve_fit(rayleigh_linear, f_ref, d_ref, 
                            p0=[100, 1e-5], 
                            bounds=([1.0, 1e-6], [100.0, 1e-1]))
        alpha, beta = popt
        
        self.calibrated_materials[material_name] = {
            "alpha": alpha, "beta": beta, "orig_y": y,
            "orig_spec": mean_spec, "orig_freqs": freqs,
            "f_ref": f_ref, "d_ref": d_ref
        }

        if show_visuals:
            self._plot_calibration_results_with_transformation(times, freqs, S, f_ref, d_ref, alpha, beta, material_name)
            #self._plot_calibration_results(times, freqs, S, f_ref, d_ref, alpha, beta, material_name)
        
        return f_ref, d_ref


    def _plot_calibration_results_with_transformation(self, times, freqs, S, f_ref, d_ref, alpha, beta, name):
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # Panel 1: Quadratic Space (f vs d)
        f_line = np.linspace(min(f_ref)*0.8, max(f_ref)*1.5, 100)
        d_line = (alpha/2) + (beta * (2*np.pi*f_line)**2 / 2)
        axes[0].scatter(f_ref, d_ref, color='blue', label='Real')
        axes[0].plot(f_line, d_line, 'r--', label='Rayleigh Fit')
        axes[0].set_title("Standard Space (Quadratic)")

        # Panel 2: Perceptual Space (Log-f vs Log-d) - This is your Fig 6-b logic!
        axes[1].scatter(f_ref, d_ref, color='green', label='Transformed Data')
        axes[1].plot(f_line, d_line, 'k-', label='Perceptual Match')
        axes[1].set_xscale('log')
        axes[1].set_yscale('log')
        axes[1].set_title("Perceptual Space (Log-Log / Fig 6-b)")
        axes[1].set_xlabel("Frequency (log Hz)")
        axes[1].set_ylabel("Damping (log 1/s)")

        # Panel 3: Spectrum Match
        axes[2].plot(freqs[:1000], librosa.amplitude_to_db(np.mean(S[:1000, :8], axis=1), ref=np.max))
        axes[2].scatter(f_ref, [0]*len(f_ref), color='red', marker='v')
        axes[2].set_title("Mode Selection")
        
        plt.tight_layout()
        plt.show()
        
    # --- PART 3: ANALYSIS UTILITY ---
    def _plot_calibration_results(self, times, freqs, S, f_ref, d_ref, alpha, beta, name):
        fig = plt.figure(figsize=(15, 5))
        
        ax1 = fig.add_subplot(131, projection='3d')
        X, Y = np.meshgrid(times[:40], freqs[:400])
        Z = librosa.amplitude_to_db(S[:400, :40], ref=np.max)
        ax1.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')
        ax1.set_title(f"3D Decay Waterfall: {name}")

        ax2 = fig.add_subplot(132)
        f_sim = np.linspace(min(f_ref)*0.5, max(f_ref)*1.5, 100)
        omega_sim = 2 * np.pi * f_sim
        d_sim = (alpha/2) + (beta * omega_sim**2 / 2)
        ax2.scatter(f_ref/1000, d_ref, color='blue', marker='x', label='Extracted (Real)')
        ax2.plot(f_sim/1000, d_sim, 'r--', label='Rayleigh Fit')
        ax2.set_xlabel("Frequency (kHz)")
        ax2.set_ylabel("Damping Rate (1/s)")
        ax2.set_title("Damping Calibration")
        ax2.legend()

        ax3 = fig.add_subplot(133)
        ax3.plot(freqs[:1000], librosa.amplitude_to_db(np.mean(S[:1000, :8], axis=1), ref=np.max))
        ax3.scatter(f_ref, librosa.amplitude_to_db(np.mean(S[freqs.searchsorted(f_ref), :8], axis=1), ref=np.max), color='red')
        ax3.set_title("Mode Selection")
        plt.tight_layout()
        plt.show()

    def plot_comparison(self, material_name, synth_y):
        """Plots Time and Frequency comparison between Original and Synthesized."""
        mat = self.calibrated_materials[material_name]
        orig_y = mat['orig_y']
        
        plt.figure(figsize=(14, 5))
        
        # Time Domain
        plt.subplot(1, 2, 1)
        t_orig = np.linspace(0, len(orig_y)/self.sr, len(orig_y))
        t_synth = np.linspace(0, len(synth_y)/self.sr, len(synth_y))
        plt.plot(t_orig[:5000], orig_y[:5000], label='Original', alpha=0.7)
        plt.plot(t_synth[:5000], synth_y[:5000], label='Synthesized', alpha=0.7, linestyle='--')
        plt.title("Waveform Comparison (Initial Attack)")
        plt.legend()

        # Frequency Domain
        plt.subplot(1, 2, 2)
        synth_spec = np.abs(librosa.stft(synth_y, n_fft=4096))
        mean_synth = np.mean(synth_spec[:, :8], axis=1)
        
        plt.plot(mat['orig_freqs'][:800], librosa.amplitude_to_db(mat['orig_spec'][:800], ref=np.max), label='Original')
        plt.plot(mat['orig_freqs'][:800], librosa.amplitude_to_db(mean_synth[:800], ref=np.max), label='Synthesized', linestyle='--')
        plt.title("Spectrum Comparison")
        plt.legend()
        plt.show()

    # --- PART 4: SYNTHESIS ---
    def synthesize(self, material_name, freqs_hz, eigvecs, K, eigvals, impact_dof, duration=1.0):
        """Combines Modal Tone + Residual Crunch."""
        mat = self.calibrated_materials[material_name]
        t = np.linspace(0, duration, int(self.sr * duration))
        omega = 2 * np.pi * freqs_hz[:6]
        d_vals = (mat['alpha'] / 2) + (mat['beta'] * omega**2 / 2)
        
        audio = np.zeros_like(t)
        for i in range(len(omega)):
            audio += np.exp(-d_vals[i] * t) * np.sin(omega[i] * t)
            
        # Residual Compensation
        u_static = pinvh(K) @ np.eye(K.shape[0])[impact_dof]
        u_modal = np.zeros_like(u_static)
        for i in range(2, 8):
            u_modal += (eigvecs[impact_dof, i] * eigvecs[:, i]) / eigvals[i]
        
        res_amp = (u_static - u_modal)[impact_dof]
        residual = (res_amp * 0.05) * np.exp(-1200 * t) * np.random.normal(0, 1, len(t))
        
        final = audio + residual
        return final / np.max(np.abs(final))
        
def main():
    studio = ModalMaterialStudio()

    # 1. Calibrate from Door
    print("Calibrating material from recording...")
    studio.calibrate_from_audio("wood_door.wav", "wood_pine")

    # 2. Physics for a New Shape (Ruler)
    print("Solving FEM for new geometry...")
    f, v, K, ev = studio.solve_geometry(E=12e9, rho=500, L=0.2, width=0.03, thickness=0.005)

    # 3. Synthesize Transfer
    print("Synthesizing transfer audio...")
    ruler_audio = studio.synthesize("wood_pine", f, v, K, ev, impact_dof=0)

    # 4. Compare and Save
    studio.plot_comparison("wood_pine", ruler_audio)
    wav.write("transfer_ruler_wood.wav", 44100, (ruler_audio * 32767).astype(np.int16))
    print("Done. Saved as transfer_ruler_wood.wav")

if __name__ == "__main__":
    main()