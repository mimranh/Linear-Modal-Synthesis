import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt
import generate_model_sound as gms
import rayleigh_objective as ro
import fem_solver as femproper
import fem_proxy as fem
import material_library as mat
import plotUtility as pu
import featureExtractor as fext # Assuming you saved the extractor here
import spectrumAnalysis as cmp

def main():
    # 1. Define Real-World Scenarios
    # Make sure 'L', 'thick', and 'width' match the real objects you recorded!
    scenarios = [
            {
            "wav_path": "wood_door.wav", 
            "mat_name": "wood_pine",
            "L": 4,       # 80 cm
            "thick": 0.3,  # 2 cm (thinner vibrates faster/higher)
            "width": 5, 
            "label": "Real_Door_Knock"
             }
        ]
    
    print("="*60)
    print("REAL-WORLD EXAMPLE-GUIDED MODAL SYNTHESIS")
    print("="*60)

    for sc in scenarios:
        label = sc['label']
        print(f"\n>>> ANALYZING RECORDING: {sc['wav_path']}")
        
        # --- NEW: ADVANCED FEATURE EXTRACTION (BLUE CROSSES) ---
        # This replaces the synthetic noise generation.
        # It also triggers the 3D Waterfall Spectrogram plot.
        
        f_ref, d_ref, target_a = fext.extract_real_world_features(sc['wav_path'], n_modes=6, label=label, pltshow=False)
        omega_ref = 2 * np.pi * f_ref
        
        # 2. Load Material Properties for FEM
        props = mat.get_material_properties(sc["mat_name"])
        
        # 3. Call Proper FEM (The Digital Twin)
        
        num_el = 100
        phys_freqs_all, eigenvectors_all = femproper.solve_proper_fem(
            E=props["E"], rho=props["rho"], 
            L=sc["L"], width=sc["width"], thickness=sc["thick"],
            num_elements=num_el 
        )
        
        valid_indices = np.where(phys_freqs_all > 1.0)[0]
        phys_freqs = phys_freqs_all[valid_indices]
        eigenvectors = eigenvectors_all[:, valid_indices]

        # Our model's frequencies (The Red Circles)
        target_freqs_hz = phys_freqs[:6]
        target_omega = target_freqs_hz * 2 * np.pi
        
        # 4. Rayleigh Optimization (The Matching Stage)
        # We find Alpha and Beta that make the RED circles match the BLUE crosses
        alpha, beta = ro.run_point_set_matching(target_omega, d_ref, material_type=sc["mat_name"])
        loss_val, rel_err = ro.calculate_matching_loss(target_omega, d_ref, alpha, beta)
        
        print(f"Extraction Results -> Ref Freqs: {f_ref}")
        print(f"Matching Quality   -> RMSE: {loss_val:.2f} | Relative Error: {rel_err:.2f}%")

        # 5. Visualization (Fig 6 a & b) - Now comparing Real vs Sim
        pu.plot_figure_6_comparison(target_freqs_hz, d_ref, alpha, beta, label)
        
        # 6. Visualization: Physical Mode Shapes
        #pu.plot_modal_shapes(eigenvectors, num_el, label)
        
        # 7. Synthesis (Auralization of the Digital Twin)
        d_est = (alpha / 2) + (beta * target_omega**2 / 2)
        # Use energy-weighted amplitudes for a more natural response
        amps = target_a / np.max(target_a)
        audio_data = gms.generate_modal_sound(target_freqs_hz, d_est, amps, duration=2)
        
        # --- Post-Processing ---
        if np.max(np.abs(audio_data)) > 0:
            audio_data = audio_data / np.max(np.abs(audio_data))
        
        # 8. Waveform Plot
        plt.figure(figsize=(10, 3))
        plt.plot(np.linspace(0, 2.0, len(audio_data)), audio_data)
        plt.title(f"Synthesized Waveform: {label}")
        plt.grid(True, alpha=0.3)
        plt.show()
        
        # 9. Save Synthesized Audio
        audio_int16 = (audio_data * 32767).astype(np.int16)
        wav_name = f"synthesized_{label}.wav"
        wavfile.write(wav_name, 44100, audio_int16)
        print(f"Successfully synthesized {label} guided by {sc['wav_path']}")



def main_dummy_example():
    # Realistic Real-Life Scenarios with full 3D dimensions
    scenarios = [
        {"name": "steel", "L": 2.0, "thick": 0.1, "width": 0.1, "label": "Steel_Beam_HEB100"},
        {"name": "wood_pine", "L": 1.5, "thick": 0.038, "width": 0.089, "label": "Pine_2x4_Stud"},
        {"name": "concrete", "L": 3.0, "thick": 0.15, "width": 0.30, "label": "Concrete_Lintel"}
    ]
    
    print("="*60)
    print("COMPLETE FEM & MODAL SYNTHESIS PIPELINE")
    print("="*60)

    for sc in scenarios:
        label = sc['label']
        print(f"\n>>> PROCESSING: {label}")
        
        # 1. Load Material Properties
        props = mat.get_material_properties(sc["name"])
        target_zeta = props["target_zeta"]
        
        # 2. Call Proper FEM Function
        # We catch BOTH frequencies and eigenvectors now
        num_el = 50
        phys_freqs, eigenvectors = femproper.solve_proper_fem(
            E=props["E"], 
            rho=props["rho"], 
            L=sc["L"], 
            width=sc["width"], 
            thickness=sc["thick"],
            num_elements=num_el 
        )
        
        target_freqs_hz = phys_freqs[:6]
        target_omega = target_freqs_hz * 2 * np.pi
        
        # 3. Simulate Reference Data (Blue Crosses in Fig 6)
        d_ref = target_zeta * target_omega + np.random.normal(0, 2, len(target_omega))
        
        # 4. Rayleigh Optimization & Loss Calculation
        alpha, beta = ro.run_point_set_matching(target_omega, d_ref)
        loss_val, rel_err = ro.calculate_matching_loss(target_omega, d_ref, alpha, beta)
        
        print(f"FEM Result -> Fundamental: {target_freqs_hz[0]:.2f} Hz")
        print(f"Fit Quality -> RMSE: {loss_val:.2f} | Relative Error: {rel_err:.2f}%")

        # 5. Visualization (A) & (B) from Figure 6
        pu.plot_figure_6_comparison(target_freqs_hz, d_ref, alpha, beta, label)
        
        # 6. Visualization: Physical Mode Shapes
        # This helps verify if the FEM assembly is physically correct
        pu.plot_modal_shapes(eigenvectors, num_el, label)
        
        # 7. Synthesis (Auralization)
        d_est = (alpha / 2) + (beta * target_omega**2 / 2)
        amps = np.array([1.0, 0.7, 0.5, 0.3, 0.2, 0.1])
        audio_data = gms.generate_modal_sound(target_freqs_hz, d_est, amps, duration=2.5)
        
        # --- FIX: Normalize to avoid clipping/white noise ---
        if np.max(np.abs(audio_data)) > 0:
            audio_data = audio_data / np.max(np.abs(audio_data))
        
        # 8. Plot Waveform for visual verification
        plt.figure(figsize=(10, 3))
        plt.plot(np.linspace(0, 2.5, len(audio_data)), audio_data)
        plt.title(f"Audio Waveform (Time Domain): {label}")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.grid(True, alpha=0.3)
        plt.savefig(f"waveform_{label}.png")
        plt.show()
        
        # 9. Save Audio File
        audio_int16 = (audio_data * 32767).astype(np.int16)
        wav_name = f"proper_fem_{label}.wav"
        wavfile.write(wav_name, 44100, audio_int16)
        print(f"Saved audio and plots for {label}")


def main_fem_proxy():
    # Define scenarios with different Materials and Geometries
    scenarios = [
        {"name": "steel",     "L": 1.0, "thick": 0.02, "label": "Steel_Rod"},
        {"name": "concrete",  "L": 1.0, "thick": 0.10, "label": "Concrete_Slab"},
        {"name": "wood_pine", "L": 1.0, "thick": 0.05, "label": "Wood_Plank"}
    ]
    
    print("="*60)
    print("EXAMPLE-GUIDED SYNTHESIS: POINT SET MATCHING (FIG 6)")
    print("="*60)

    for sc in scenarios:
        label = sc['label']
        print(f"\n>>> PROCESSING: {label}")
        
        # 1. Load Material Physics and Reference Profile
        props = mat.get_material_properties(sc["name"])
        target_zeta = props["target_zeta"]
        
        # 2. FEM Physics (Forward Model)
        phys_freqs, _ = fem.get_fem_parameters(
            props["E"], props["rho"], L=sc["L"], thickness=sc["thick"], width=sc["thick"]
        )
        target_freqs_hz = phys_freqs[:6]
        target_omega = target_freqs_hz * 2 * np.pi
        
        # --- CONVERSION TO DECAY RATE (d) ---
        # Figure 6(a) uses decay rate d = zeta * omega
        # We add a tiny bit of noise to simulate real 'Reference' recording data
        d_ref = target_zeta * target_omega + np.random.normal(0, 2, len(target_omega))
        
        # 3. Rayleigh Optimization (Point Set Matching)
        # Matches Red Circles to Blue Crosses in (f, d) space
        alpha, beta = ro.run_point_set_matching(target_omega, d_ref)
        print(f"Optimized -> Alpha: {alpha:.4f}, Beta: {beta:.2e}")

        loss_val, rel_err = ro.calculate_matching_loss(target_omega, d_ref, alpha, beta)
        
        print(f"Optimized -> Alpha: {alpha:.4f}, Beta: {beta:.2e}")
        print(f"Matching Loss (RMSE): {loss_val:.2f} [1/sec]")
        print(f"Relative Fit Error: {rel_err:.2f}%")

        # 4. Visualization (Replicating Fig 6 a and b)
        pu.plot_figure_6_comparison(target_freqs_hz, d_ref, alpha, beta, label)
        
        # 5. Synthesis (Auralization)
        # We use the matched decay rates for the audio engine
        d_est = (alpha / 2) + (beta * target_omega**2 / 2)
        amps = np.array([1.0, 0.7, 0.5, 0.3, 0.2, 0.1])
        
        audio_data = gms.generate_modal_sound(target_freqs_hz, d_est, amps, duration=2.0)
        
        # Export
        wav_name = f"sound_{label}.wav"
        wavfile.write(wav_name, 44100, audio_data)
        print(f"Saved audio: {wav_name}")


def Main_main():
    # Scenarios using our updated library
    scenarios = [
        {"name": "steel",     "L": 1.0, "thick": 0.02, "label": "Steel_Rod"},
        {"name": "concrete",  "L": 1.0, "thick": 0.10, "label": "Concrete_Slab"},
        {"name": "wood_pine", "L": 1.0, "thick": 0.05, "label": "Wood_Plank"}
    ]
    
    for sc in scenarios:
        print(f"\n--- Simulating {sc['label']} ---")
        
        # 1. Load Material Physics and its Unique Recording Profile
        props = mat.get_material_properties(sc["name"])
        target_zeta = props["target_zeta"]
        
        # 2. Calculate Frequencies
        phys_freqs, _ = fem.get_fem_parameters(
            props["E"], props["rho"], L=sc["L"], thickness=sc["thick"], width=sc["thick"]
        )
        target_freqs_hz = phys_freqs[:6]
        target_omega = target_freqs_hz * 2 * np.pi
        
        # 3. Rayleigh Optimization (Using the NEW Log-Residual fix)
        # This will now capture the 'U' shape for each specific material
        alpha, beta = ro.run_rayleigh_optimization(target_omega, target_zeta)
        print(f"Alpha: {alpha:.2f}, Beta: {beta:.2e}")

        # 4. Visualization of the Fit
        pu.plot_rayleigh_fit(target_omega, target_zeta, alpha, beta, filename=f"fit_{sc['label']}.png")
        
        # 5. Synthesis
        decay_rates = target_zeta * target_omega
        amps = np.array([1.0, 0.6, 0.4, 0.3, 0.2, 0.1])
        audio = gms.generate_modal_sound(target_freqs_hz, decay_rates, amps, duration=2.0)
        
        wavfile.write(f"sound_{sc['label']}.wav", 44100, audio)

if __name__ == "__main__":
    main()
    cmp.compare_spectra("wood_door.wav", "synthesized_Real_Door_Knock.wav")