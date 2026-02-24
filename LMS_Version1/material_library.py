import numpy as np

def get_material_properties(name):
    """
    Returns Physical constants and realistic damping profiles (zeta).
    Data sourced from architectural acoustics standards.
    """
    materials = {
        "steel": {
            "E": 210e9,
            "rho": 7800,
            # Steel has very low damping, almost constant across frequencies
            "target_zeta": np.array([0.002, 0.0018, 0.0015, 0.002, 0.0025, 0.003]),
            "description": "High stiffness, very low damping. Metallic ringing."
        },
        "concrete": {
            "E": 30e9,
            "rho": 2400,
            # Concrete has higher damping, especially at low frequencies (internal voids)
            "target_zeta": np.array([0.02, 0.015, 0.012, 0.018, 0.025, 0.03]),
            "description": "Heavy, medium damping. Damped 'thud' sound."
        },
        "wood_pine": {
            "E": 12e9,
            "rho": 500,
            # Wood has high damping due to its cellular/fibrous structure
            "target_zeta": np.array([0.05, 0.045, 0.04, 0.045, 0.055, 0.06]),
            "description": "High damping, low density. Warm, organic decay."
        },
        "glass": {
            "E": 70e9,
            "rho": 2500,
            # Glass is brittle with low damping but higher than steel
            "target_zeta": np.array([0.005, 0.004, 0.0035, 0.005, 0.007, 0.009]),
            "description": "High stiffness, sharp decay. Brittle ringing."
        }
    }
    
    return materials.get(name.lower(), materials["steel"])