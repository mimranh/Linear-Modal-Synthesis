import json
import numpy as np

def log_data_to_json(filename, validated_modes, alpha, beta, fs):
    """
    Logs modal parameters and Rayleigh constants to a JSON file for MATLAB.
    """
    # Convert numpy types to native Python types for JSON compatibility
    data = {
        "metadata": {
            "fs": int(fs),
            "alpha": float(alpha),
            "beta": float(beta)
        },
        "all_modes": [
            {"f": float(m['f']), "d": float(m['d']), "A": float(m['A'])} 
            for m in validated_modes
        ]
    }
    
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"Data successfully logged to {filename}")

# Usage in main():
# log_data_to_json("wood_modal_data.json", validated_modes, clean_modes, alpha, beta, fs)