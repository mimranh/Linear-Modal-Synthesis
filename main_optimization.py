import numpy as np
from scipy.io import wavfile
import generate_model_sound as gms
import rayleigh_objective as ro
import fem_proxy as fem
import material_library as mat
import plotUtility as pu


def main():
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