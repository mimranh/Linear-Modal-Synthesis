import numpy as np
from scipy.optimize import curve_fit

def rayleigh_model(f, alpha, beta):
    w = 2 * np.pi * f
    return 0.5 * (alpha / (w+ 1e-6) + beta * w)


def estimate_material_parameters(modes):
    # Filter out very low/high frequency modes that might be noise
    valid_modes = [m for m in modes if 10 < m["f"] < 20000]
    
    f_vals = np.array([m["f"] for m in valid_modes])
    d_vals = np.array([m["d"] for m in valid_modes])
    a_vals = np.array([m["A"] for m in valid_modes]) # Energies

    # Paper Suggestion: Weight the fit by the amplitude (A) 
    # This forces the optimizer to prioritize the modes you actually hear
    weights = np.sqrt(a_vals) 
    weights /= np.max(weights)

    # REFINED BOUNDS: Alpha for metal is usually 1-50. 
    # If it was hitting 1000, something was wrong with the scale.
    # p0: Start with low alpha and very low beta for metal
    try:
        popt, _ = curve_fit(
            rayleigh_model, 
            f_vals, 
            d_vals, 
            p0=[10, 1e-6], 
            sigma=1/weights, # Higher energy modes have more influence
            bounds=([0, 0], [2000, 0.1])
        )
        alpha, beta = popt
    except:
        print("Optimization failed, using defaults.")
        alpha, beta = 10, 1e-6

    return alpha, beta

def estimate_material_parameters_robust(modes):
    # 1. Filter out high-frequency outliers that ruin the 'Wood' slope
    # Real wood body rarely has stable modes above 2000Hz 
    filtered_modes = [m for m in modes if m['f'] < 2000]
    
    # 2. Filter out extreme damping outliers (noise)
    # If d > 30, it's likely a transient, not a material property
    filtered_modes = [m for m in filtered_modes if m['d'] < 30]

    f_vals = np.array([m['f'] for m in filtered_modes])
    d_vals = np.array([m['d'] for m in filtered_modes])
    
    # 3. Give the cluster 'Massive' weight
    weights = np.array([m['A'] for m in filtered_modes])
    weights = weights / np.max(weights)

    def rayleigh_model(f, alpha, beta):
        omega = 2 * np.pi * f
        return 0.5 * (alpha / omega + beta * omega)

    popt, _ = curve_fit(
        rayleigh_model, f_vals, d_vals, 
        p0=[100, 0.005], 
        sigma=1/weights,
        bounds=([0, 0], [5000, 0.1])
    )
    return popt # [alpha, beta]


import numpy as np

def estimate_material_linear_regression(modes):
    """
    Estimates alpha and beta using linear regression on transformed coordinates.
    This is more stable for high-damping materials like wood.
    """
    if not modes:
        return 10.0, 1e-6 # Default fallbacks

    f = np.array([m['f'] for m in modes])
    d = np.array([m['d'] for m in modes])
    a = np.array([m['A'] for m in modes])
    
    omega = 2 * np.pi * f
    
    # Coordinates for linear space
    X = omega**2
    Y = 2 * omega * d
    
    # Weighting: Use sqrt of amplitude to prioritize the 'wood body'
    weights = np.sqrt(a) / np.max(np.sqrt(a))
    
    # Perform weighted linear fit: Y = beta * X + alpha
    # polyfit returns [slope, intercept] -> [beta, alpha]
    params = np.polyfit(X, Y, 1, w=weights)
    
    beta = params[0]
    alpha = params[1]
    
    # Physical constraints: Wood must have positive damping
    # If regression gives negative values, we floor them to small positives
    alpha = max(0.1, alpha)
    beta = max(1e-10, beta)
    
    return float(alpha), float(beta)