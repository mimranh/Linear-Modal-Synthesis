import numpy as np
from scipy.optimize import minimize

def calculate_matching_loss(target_omega, target_d_ref, alpha, beta):
    """
    Calculates the L2 Matching Loss (RMSE) between 
    the Reference example and the Rayleigh Estimate.
    """
    d_est = (alpha / 2) + (beta * target_omega**2 / 2)
    mse = np.mean((d_est - target_d_ref)**2)
    rmse = np.sqrt(mse)
    
    # Also calculate Relative Error (%)
    relative_error = np.mean(np.abs(d_est - target_d_ref) / (target_d_ref + 1e-6)) * 100
    
    return rmse, relative_error

def run_rayleigh_optimization(target_omega, target_zeta):
    """
    Fits Rayleigh Damping parameters (alpha, beta) using Log-Residuals
    to ensure the U-shape is captured even at high frequencies.
    """
    
    def objective(params):
        alpha, beta = params
        # Standard Rayleigh Formula
        sim_zeta = (alpha / (2 * target_omega)) + (beta * target_omega / 2)
        
        # LOG-RESIDUALS: This is the key. 
        # It calculates error as a ratio rather than absolute distance.
        # This gives the 'Mass' term (alpha) a chance to compete with 'Stiffness' (beta).
        loss = np.sum((np.log(sim_zeta) - np.log(target_zeta))**2)
        
        # Physical constraints
        if alpha < 0 or beta < 0:
            return 1e10
        return loss

    # Starting guesses: We need a much higher alpha for your high frequencies
    init_guess = [10.0, 1e-7]
    
    # Use L-BFGS-B with strictly positive bounds
    res = minimize(
        objective, 
        init_guess, 
        method='L-BFGS-B', 
        bounds=[(1e-5, None), (1e-12, None)]
    )
    
    return res.x

def run_point_set_matching(target_omega, target_d_ref):
    """
    Matches the Estimated points (Red Circles) to Reference points (Blue Crosses)
    following the logic of Figure 6.
    target_d_ref: The decay rates (1/sec) from the 'Example' audio.
    """
    def objective(params):
        alpha, beta = params
        # The Rayleigh Estimate for decay rate 'd'
        d_est = (alpha / 2) + (beta * target_omega**2 / 2)
        
        # Point-set distance (L2 Norm) between Reference and Estimate
        # This is the 'Matching' error
        error = np.sum((d_est - target_d_ref)**2)
        return error

    # Constraints: alpha and beta must be positive
    res = minimize(objective, [1.0, 1e-8], bounds=[(1e-5, None), (1e-12, None)])
    return res.x


if __name__ == "__main__":
    # Your specific data
    test_omega = np.array([15392.35, 46219.27, 77172.93, 108338.18, 139800.33, 171645.25])
    test_zeta = np.array([0.02, 0.015, 0.012, 0.018, 0.025, 0.03])
    
    alpha, beta = run_point_set_matching(test_omega, test_zeta)
    print(f"--- Optimized for High Frequencies ---")
    print(f"Alpha: {alpha:.4f}")
    print(f"Beta:  {beta:.4e}")