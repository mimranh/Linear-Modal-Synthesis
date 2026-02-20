import numpy as np
from scipy.io import wavfile
import generate_model_sound as gms
import rayleigh_objective as ro
import fem_proxy as fem
import material_library as mat
import plotUtility as pu

def main():
    """
    Main execution flow for Example-Guided Linear Modal Synthesis.
    This script simulates the acoustic response of various building materials
    by solving the underlying physics (FEM) and optimizing damping parameters.
    """
    # 1. Selection of materials to simulate from our library
    test_materials = ["steel", "concrete", "wood_pine", "glass"]
    
    print("="*50)
    print("STARTING PHYSICAL MODAL SYNTHESIS PIPELINE")
    print("="*50)

    for m_name in test_materials:
        print(f"\n>>> PROCESSING MATERIAL: {m_name.upper()}")
        
        # --- STEP 1: PHYSICAL PROPERTIES ---
        # Fetch E (Stiffness) and Rho (Density) from the library
        props = mat.get_material_properties(m_name)
        E = props["E"]
        rho = props["rho"]
        
        # --- STEP 2: FEM PHYSICS (FORWARD MODEL) ---
        # Solve (K - w^2 M)phi = 0 to get natural resonance frequencies
        # Higher 'num_elements' increases accuracy but takes more time
        phys_freqs, _ = fem.get_fem_parameters(E, rho, L=1.5, num_elements=30)
        
        # We synthesize the first 6 audible modes
        target_freqs_hz = phys_freqs[:6]
        target_omega = target_freqs_hz * 2 * np.pi
        print(f"Fundamental Frequency: {target_freqs_hz[0]:.2f} Hz")
        
        # --- STEP 3: DAMPING CALIBRATION (THE 'GUIDED' PART) ---
        # These are the 'Target' damping ratios we want our simulation to match.
        # In a real project, these are extracted from a real audio recording.
        target_zeta = np.array([0.02, 0.015, 0.012, 0.018, 0.025, 0.03])
        
        # Optimize alpha and beta to fit the Rayleigh Damping model
        alpha_opt, beta_opt = ro.run_rayleigh_optimization(target_omega, target_zeta)
        
        # --- STEP 4: AURALIZATION ---
        # Convert the dimensionless damping ratio (zeta) back to decay rates (d)
        # d = zeta * omega
        decay_rates = target_zeta * target_omega
        
        # Define relative loudness for each mode (higher modes usually quieter)
        amps = np.array([1.0, 0.7, 0.5, 0.3, 0.2, 0.1])
        
        # Synthesize the audio signal
        audio_data = gms.generate_modal_sound(target_freqs_hz, decay_rates, amps, duration=1.5)
        
        # --- STEP 5: OUTPUT AND EXPORT ---
        output_filename = f"auralization_{m_name}.wav"
        wavfile.write(output_filename, 44100, audio_data)
        
        print(f"Successfully saved: {output_filename}")
        print(f"Material Info: {props['description']}")

    # Final visualization for the last processed material
    print("\n" + "="*50)
    print("Generating Damping Fit Visualization...")
    pu.plot_rayleigh_fit(target_omega, target_zeta, alpha_opt, beta_opt)
    print("Pipeline Complete. All sounds generated.")

if __name__ == "__main__":
    main()