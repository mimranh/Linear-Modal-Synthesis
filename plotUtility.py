import matplotlib.pyplot as plt
import numpy as np

def plot_rayleigh_fit(target_omega, target_zeta, alpha, beta):
    w_range = np.linspace(min(target_omega)*0.5, max(target_omega)*1.5, 500)
    zeta_curve = (alpha / (2 * w_range)) + (beta * w_range / 2)

    plt.figure(figsize=(10, 5))
    plt.scatter(target_omega / (2*np.pi), target_zeta, color='red', label='Target')
    plt.plot(w_range / (2*np.pi), zeta_curve, label='Rayleigh Fit')
    plt.title("Rayleigh Damping Calibration")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Damping Ratio (ζ)")
    plt.legend()
    plt.grid(True)
    plt.savefig("rayleigh_fit.png")
    print("Verification plot saved as rayleigh_fit.png")