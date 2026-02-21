import numpy as np
from scipy.linalg import eigh

def solve_proper_fem(E, rho, L, width, thickness, num_elements=40):
    """
    Proper Finite Element Method for a cantilever beam.
    Uses 2-node beam elements with 2 Degrees of Freedom per node (Displacement & Rotation).
    """
    n_nodes = num_elements + 1
    dof_per_node = 2
    total_dof = n_nodes * dof_per_node
    
    dx = L / num_elements
    I = (width * thickness**3) / 12  # Second moment of area
    A = width * thickness            # Cross-section area
    
    K_global = np.zeros((total_dof, total_dof))
    M_global = np.zeros((total_dof, total_dof))
    
    # Element Stiffness Matrix (Hermitian Beam Element)
    k_e = (E * I / dx**3) * np.array([
        [12, 6*dx, -12, 6*dx],
        [6*dx, 4*dx**2, -6*dx, 2*dx**2],
        [-12, -6*dx, 12, -6*dx],
        [6*dx, 2*dx**2, -6*dx, 4*dx**2]
    ])
    
    # Element Mass Matrix (Consistent Mass Matrix)
    m_e = (rho * A * dx / 420) * np.array([
        [156, 22*dx, 54, -13*dx],
        [22*dx, 4*dx**2, 13*dx, -3*dx**2],
        [54, 13*dx, 156, -22*dx],
        [-13*dx, -3*dx**2, -22*dx, 4*dx**2]
    ])
    
    # Assembly Process
    for i in range(num_elements):
        idx = i * dof_per_node
        K_global[idx:idx+4, idx:idx+4] += k_e
        M_global[idx:idx+4, idx:idx+4] += m_e
        
    # Boundary Condition: Fixed at one end (Cantilever)
    # Remove first two DOFs (displacement and rotation at x=0)
    K_bc = K_global[2:, 2:]
    M_bc = M_global[2:, 2:]
    
    # Solve Generalized Eigenvalue Problem
    eigenvalues, eigenvectors = eigh(K_bc, M_bc)
    
    # Convert Eigenvalues to Hz: f = sqrt(lambda) / 2pi
    frequencies_hz = np.sqrt(np.maximum(eigenvalues, 0)) / (2 * np.pi)
    
    return frequencies_hz, eigenvectors