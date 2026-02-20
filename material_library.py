def get_material_properties(name):
    """
    Returns Young's Modulus (E in Pa) and Density (rho in kg/m^3)
    for common building materials.
    """
    materials = {
        "steel": {
            "E": 210e9,
            "rho": 7800,
            "description": "High stiffness, high density. Bright, clear ringing sound."
        },
        "concrete": {
            "E": 30e9,
            "rho": 2400,
            "description": "Heavy, medium stiffness. Duller 'thud' sound."
        },
        "wood_pine": {
            "E": 12e9,
            "rho": 500,
            "description": "Low density, relatively stiff for its weight. Warm, resonant sound."
        },
        "glass": {
            "E": 70e9,
            "rho": 2500,
            "description": "High stiffness, medium density. Sharp, brittle ringing."
        },
        "aluminum": {
            "E": 69e9,
            "rho": 2700,
            "description": "Medium stiffness, light. Clean, metallic ring."
        }
    }
    
    return materials.get(name.lower(), materials["steel"])