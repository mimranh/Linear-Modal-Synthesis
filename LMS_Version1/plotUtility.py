import matplotlib.pyplot as plt
import numpy as np

def plot_rayleigh_fit(target_omega, target_zeta, alpha, beta, filename="rayleigh_fit.png"):
    w_range = np.linspace(min(target_omega)*0.5, max(target_omega)*1.5, 500)
    zeta_curve = (alpha / (2 * w_range)) + (beta * w_range / 2)

    plt.figure(figsize=(10, 5))
    plt.scatter(target_omega / (2*np.pi), target_zeta, color='red', label='Target Data')
    plt.plot(w_range / (2*np.pi), zeta_curve, label='Rayleigh Model Fit')
    plt.title(f"Damping Calibration: {filename.split('.')[0]}")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Damping Ratio (ζ)")
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    plt.show()
    plt.close() # Important to close the figure so they don't overlap in memory


def plot_rayleigh_fit_component(target_omega, target_zeta, alpha, beta, filename="rayleigh_fit.png"):
    # Generate frequencies from 0 to 1.5x the max target frequency
    # Starting at 10Hz instead of the first mode shows the U-bend
    w_range = np.linspace(10 * 2 * np.pi, max(target_omega) * 1.5, 1000)
    
    # Rayleigh components
    zeta_mass = alpha / (2 * w_range)
    zeta_stiff = (beta * w_range) / 2
    zeta_total = zeta_mass + zeta_stiff

    plt.figure(figsize=(10, 6))
    
    # Plot the components as dashed lines
    plt.plot(w_range/(2*np.pi), zeta_mass, 'g--', label=f'Mass Proportional (α={alpha:.2f})', alpha=0.6)
    plt.plot(w_range/(2*np.pi), zeta_stiff, 'b--', label=f'Stiffness Proportional (β={beta:.2e})', alpha=0.6)
    plt.plot(w_range/(2*np.pi), zeta_total, 'r-', linewidth=2, label='Total Rayleigh Fit')
    
    # Plot target points
    plt.scatter(target_omega/(2*np.pi), target_zeta, color='black', zorder=5, label='Target Data')

    plt.title(f"Damping Fit Analysis: {filename.replace('fit_', '').replace('.png', '')}")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Damping Ratio (ζ)")
    plt.ylim(0, max(target_zeta) * 1.5) # Keep the focus on the data
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend()
    plt.savefig(filename)
    plt.close()

def plot_figure_6_comparison(freqs_hz, d_ref, alpha, beta, label):
    """
    Replicates Figure 6 from the paper:
    (a) Original frequency and damping space
    (b) Transformed feature space
    """
    omega = 2 * np.pi * freqs_hz
    # d = alpha/2 + beta*omega^2 / 2
    d_est = (alpha / 2) + (beta * omega**2 / 2)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # --- Plot (a): (f, d)-space ---
    # Frequencies in kHz, Damping in 1/sec
    ax1.scatter(freqs_hz / 1000, d_ref, marker='x', color='blue', s=80, label='Reference (Blue Crosses)')
    ax1.scatter(freqs_hz / 1000, d_est, marker='o', facecolors='none', edgecolors='red', s=100, label='Estimated (Red Circles)')
    
    ax1.set_title(f"Fig 6(a): {label} in (f, d)-space")
    ax1.set_xlabel("Frequency (kHz)")
    ax1.set_ylabel("Damping (1/sec)")
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.6)

    # --- Plot (b): Transformed (x, y)-space ---
    # Using the common paper transformation: X = f, Y = 1/d or log-based
    # This helps visualize how the 'Point Set Matching' handles the distribution
    x_feat = freqs_hz / 1000
    y_ref_feat = 1000 / (d_ref + 1e-6) # Inverse transformation to cluster high values
    y_est_feat = 1000 / (d_est + 1e-6)

    ax2.scatter(x_feat, y_ref_feat, marker='x', color='blue', s=80)
    ax2.scatter(x_feat, y_est_feat, marker='o', facecolors='none', edgecolors='red', s=100)
    
    # Labeling the most energetic modes (Modes 1, 2, 3) as seen in the paper
    for i in range(3):
        ax2.annotate(f"{i+1}", (x_feat[i], y_ref_feat[i]), textcoords="offset points", xytext=(0,10), ha='center', color='blue', weight='bold')

    ax2.set_title(f"Fig 6(b): {label} in Transformed space")
    ax2.set_xlabel("X(f)")
    ax2.set_ylabel("Y(d)")
    ax2.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig(f"figure6_comparison_{label}.png")
    plt.show()

def plot_modal_shapes(eigenvectors, num_elements, label, num_modes_to_plot=3):
    """
    Plots the spatial displacement of the beam for the first few modes.
    """
    n_nodes = num_elements + 1
    x_coords = np.linspace(0, 1, n_nodes)
    
    plt.figure(figsize=(10, 6))
    
    for m in range(num_modes_to_plot):
        # Extract only the displacement DOFs (even indices: 0, 2, 4...)
        # Note: If you used BCs, the first nodes might be missing
        mode_data = eigenvectors[:, m]
        displacements = mode_data[0::2] 
        
        # Add the fixed-end (0,0) if it was removed for BCs
        full_displacements = np.insert(displacements, 0, 0)
        
        # Normalize for plotting
        full_displacements /= np.max(np.abs(full_displacements))
        
        plt.plot(x_coords, full_displacements, label=f"Mode {m+1}", linewidth=2)

    plt.title(f"FEM Mode Shapes: {label}")
    plt.xlabel("Normalized Length (x/L)")
    plt.ylabel("Relative Displacement")
    plt.axhline(0, color='black', lw=1, ls='--')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"modes_{label}.png")
    plt.show()
    plt.close()