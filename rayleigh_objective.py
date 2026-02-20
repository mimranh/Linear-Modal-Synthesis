import numpy as np
from scipy.optimize import minimize

def run_rayleigh_optimization(target_omega, target_zeta):
    """
    Fits Rayleigh Damping parameters (alpha, beta) to observed damping ratios.
    
    Args:
        target_omega (np.array): Angular frequencies in rad/s
        target_zeta (np.array): Observed damping ratios (unitless)
        
    Returns:
        tuple: (alpha_optimized, beta_optimized)
    """
    
    def objective(params):
        """
        Internal loss function to minimize.
        zeta_i = alpha / (2 * omega_i) + (beta * omega_i / 2)
        """
        alpha, beta = params
        
        # Calculate simulated damping based on current alpha/beta guess
        sim_zeta = (alpha / (2 * target_omega)) + (beta * target_omega / 2)
        
        # Mean Squared Error between simulation and target
        loss = np.mean((sim_zeta - target_zeta)**2)
        
        # Physical constraint: alpha and beta must be non-negative
        if alpha < 0 or beta < 0:
            return 1e10
            
        return loss

    # Initial guesses: [alpha, beta]
    # alpha usually affects low frequencies, beta affects high frequencies
    init_guess = [0.1, 1e-6]
    
    # Use L-BFGS-B optimizer which allows for boundary constraints (0 to infinity)
    res = minimize(
        objective, 
        init_guess, 
        method='L-BFGS-B', 
        bounds=[(1e-8, None), (1e-12, None)]
    )
    
    if res.success:
        return res.x
    else:
        print("Optimization failed:", res.message)
        return init_guess

# This block allows you to run the file independently for testing
if __name__ == "__main__":
    # Test data
    test_omega = np.array([440, 880, 1760]) * 2 * np.pi
    test_zeta = np.array([0.02, 0.015, 0.03])
    
    alpha, beta = run_rayleigh_optimization(test_omega, test_zeta)
    print(f"--- Standalone Test ---")
    print(f"Optimized Alpha: {alpha:.4f}")
    print(f"Optimized Beta:  {beta:.4e}")