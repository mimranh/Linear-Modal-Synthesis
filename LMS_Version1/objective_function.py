import numpy as np
from scipy.optimize import minimize

# 1. Define our "Real World" target (e.g., a Steel Rod recording)
target_freqs = np.array([440, 880, 1320]) # Target harmonic series
target_damp = 2.5 # Target decay rate

def objective_function(params):
    """
    This is the core of "Example-Guided" synthesis.
    params[0] = Estimated Stiffness factor (E)
    params[1] = Estimated Damping factor (d)
    """
    E_guess, d_guess = params
    
    # Simulate sound based on the guess
    # Frequency is proportional to sqrt(E)
    sim_freqs = np.array([1, 2, 3]) * 100 * np.sqrt(E_guess) 
    sim_damp = d_guess
    
    # Calculate "Loss" (Difference between Sim and Target)
    freq_error = np.sum((sim_freqs - target_freqs)**2)
    damp_error = (sim_damp - target_damp)**2
    
    return freq_error + damp_error

# 2. Run the Optimizer
initial_guess = [1.0, 1.0] # Starting point for E and d
result = minimize(objective_function, initial_guess, method='Nelder-Mead')

print(f"Optimization Successful: {result.success}")
print(f"Estimated Stiffness (E): {result.x[0]:.2f}")
print(f"Estimated Damping (d): {result.x[1]:.2f}")