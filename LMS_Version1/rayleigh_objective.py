import numpy as np
from scipy.optimize import minimize
from scipy.optimize import lsq_linear


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


def run_point_set_matching(omega_fem, d_ref, material_type="metal"):
    """
    Advanced Rayleigh matching with dynamic weights and physical constraints.
    Fixed to handle variable numbers of extracted modes.
    """
    n_modes = len(omega_fem)
    
    # 1. Dynamic Weights: Prioritize lower frequencies
    # We use linspace to ensure the weight array matches n_modes exactly.
    # For metal, we want the high frequencies to ring, so we keep weights high.
    if material_type == "metal":
        weights = np.linspace(1.0, 0.5, n_modes)
    else:
        # For wood, we weight the fundamental very heavily to ensure the 'thump'
        weights = np.linspace(1.0, 0.2, n_modes)
    
    # 2. Build the Linear System: d = alpha/2 + (beta * omega^2)/2
    A = np.zeros((n_modes, 2))
    A[:, 0] = 0.5
    A[:, 1] = (omega_fem**2) / 2
    
    # Matrix multiplication with Weighting
    W = np.diag(weights)
    A_weighted = W @ A
    b_weighted = W @ d_ref
    
    # 3. Set Physical Bounds
    # Note: These bounds override the 'zero' damping found in your wood recording
    if material_type == "metal":
        # Keep alpha/beta tiny to allow long sustain (the 'ping')
        lb = [0.0, 1e-11]  
        ub = [2.0, 1e-8]   
    else:
        # Wood: Force high alpha (mass damping) to kill the 'church bell' ring
        # Even if d_ref is 0.0001, alpha will be forced to at least 80.0
        lb = [5.0, 1e-5]  
        ub = [500.0, 1e-2]

    # 4. Solve Constrained Least Squares
    res = lsq_linear(A_weighted, b_weighted, bounds=(lb, ub))
    alpha, beta = res.x
    
    # Log the results so you can see what the optimizer decided
    print(f"--- Rayleigh Fit ({material_type}) ---")
    print(f"Modes matched: {n_modes}")
    print(f"Alpha: {alpha:.4f}, Beta: {beta:.2e}")
    
    return alpha, beta


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

def run_point_set_matching_old(target_omega, target_d_ref):
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