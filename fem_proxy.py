import numpy as np
from scipy.linalg import eigh
from scipy.linalg import pinvh


def get_fem_parameters(E, rho, L=1.0, width=0.01, thickness=0.01, num_elements=30):
    """
    Simulates a structural beam with specific dimensions.
    width/thickness: in meters (default 1cm)
    """
    n_nodes = num_elements + 1
    dx = L / num_elements
    area = width * thickness # Cross-sectional area
    
    # Element Stiffness (K) and Mass (M) scaling with Area and Length
    k_el = (E * area / dx) * np.array([[1, -1], [-1, 1]])
    m_el = (rho * area * dx / 6) * np.array([[2, 1], [1, 2]])
    
    # ... (rest of the assembly logic stays the same) ...
    K = np.zeros((n_nodes, n_nodes))
    M = np.zeros((n_nodes, n_nodes))
    for i in range(num_elements):
        K[i:i+2, i:i+2] += k_el
        M[i:i+2, i:i+2] += m_el
    
    K_fixed = K[1:, 1:]
    M_fixed = M[1:, 1:]
    eigenvalues, eigenvectors = eigh(K_fixed, M_fixed)
    frequencies_hz = np.sqrt(np.maximum(eigenvalues, 0)) / (2 * np.pi)
    
    return frequencies_hz, eigenvectors


def compute_residual_profile(K, eigenvectors, eigenvalues, impact_node):
    """
    Calculates the residual displacement vector (R).
    """
    n_dofs = K.shape[0]
    F = np.zeros(n_dofs)
    F[impact_node] = 1.0  # Unit force at impact site

    # 1. Static response (Total deformation)
    # We use pinvh because K is symmetric and has 0-eigenvalues (Free-Free)
    u_static = pinvh(K) @ F 

    # 2. Subtract the first 6 modes to find what's missing
    u_modal_sum = np.zeros(n_dofs)
    for i in range(6):
        phi = eigenvectors[:, i]
        lam = eigenvalues[i] # This is omega^2
        # Contribution of mode i to the static shape
        u_modal_sum += (phi[impact_node] * phi) / lam

    return u_static - u_modal_sum


def get_fem_parameters1(E, rho, L=1.0, num_elements=10):
    """
    Simulates a structural beam using a Finite Element Proxy.
    E: Young's Modulus (Pa)
 main1   rho: Density (kg/m^3)
    L: Length (m)
    """
    # Discretization
    n_nodes = num_elements + 1
    dx = L / num_elements
    area = 0.01 * 0.01  # 1cm x 1cm cross section
    
    # Element Stiffness and Mass matrices
    k_el = (E * area / dx) * np.array([[1, -1], [-1, 1]])
    m_el = (rho * area * dx / 6) * np.array([[2, 1], [1, 2]])
    
    # Global Matrices
    K = np.zeros((n_nodes, n_nodes))
    M = np.zeros((n_nodes, n_nodes))
    
    # Assembly
    for i in range(num_elements):
        K[i:i+2, i:i+2] += k_el
        M[i:i+2, i:i+2] += m_el
        
    # Apply Boundary Conditions (Fix one end - Clamped)
    # We remove the first row and column
    K_fixed = K[1:, 1:]
    M_fixed = M[1:, 1:]
    
    # Solve Generalized Eigenvalue Problem
    # returns eigenvalues (w^2) and eigenvectors (phi)
    eigenvalues, eigenvectors = eigh(K_fixed, M_fixed)
    
    # Convert to Frequencies (Hz)
    frequencies_hz = np.sqrt(np.maximum(eigenvalues, 0)) / (2 * np.pi)
    
    return frequencies_hz, eigenvectors

if __name__ == "__main__":
    # Test with Steel properties
    f, phi = get_fem_parameters(E=210e9, rho=7800)
    print("First 5 Resonance Frequencies (Hz):")
    print(f[:5])