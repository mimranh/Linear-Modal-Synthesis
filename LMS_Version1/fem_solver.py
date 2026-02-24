import numpy as np
from scipy.linalg import eigh
from scipy.linalg import pinvh


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
    #K_bc = K_global[2:, 2:]
    #M_bc = M_global[2:, 2:]

    K_bc = K_global
    M_bc = M_global
    
    # Solve Generalized Eigenvalue Problem
    eigenvalues_raw, eigenvectors = eigh(K_bc, M_bc)
    
    # Convert Eigenvalues to Hz: f = sqrt(lambda) / 2pi
    frequencies_hz = np.sqrt(np.maximum(eigenvalues_raw, 0)) / (2 * np.pi)
    is_vibrational = frequencies_hz > 1.0
    frequencies_hz = frequencies_hz[is_vibrational]
    vibrational_vectors = eigenvectors[:, is_vibrational]
    
    
    return frequencies_hz, vibrational_vectors, K_bc, eigenvalues_raw



def compute_residual_profile(K, eigenvectors_full, eigenvalues_raw, impact_node_idx):
    """
    K: The full stiffness matrix returned by solve_proper_fem.
    eigenvectors_full: The raw eigenvectors from eigh.
    eigenvalues_raw: The raw eigenvalues (omega^2).
    impact_node_idx: The DOF index where the door is hit.
    """
    n_dofs = K.shape[0]
    F = np.zeros(n_dofs)
    # Applying unit force to the displacement DOF of the impact node
    F[impact_node_idx] = 1.0

    # Total static displacement using pseudo-inverse for Free-Free stability
    u_static = pinvh(K) @ F 

    # Subtract the contribution of the modes we already have (Modes 2 through 7)
    # We skip 0 and 1 because they are Rigid Body Modes.
    u_modal_sum = np.zeros(n_dofs)
    for i in range(2, 8): 
        phi = eigenvectors_full[:, i]
        lam = eigenvalues_raw[i] 
        u_modal_sum += (phi[impact_node_idx] * phi) / lam

    return u_static - u_modal_sum



def compute_residual_profile(K, eigenvectors_full, eigenvalues_raw, impact_dof):
    """
    K: Full global stiffness matrix.
    eigenvectors_full: All eigenvectors from eigh.
    eigenvalues_raw: All eigenvalues from eigh (omega squared).
    impact_dof: The specific Degree of Freedom index where the force is applied.
    """
    n_dofs = K.shape[0]
    
    # 1. Create a unit force vector at the impact DOF
    F = np.zeros(n_dofs)
    F[impact_dof] = 1.0 

    # 2. Compute Total Static Response (u_static)
    # pinvh is the Moore-Penrose pseudo-inverse for symmetric matrices.
    # It solves K*u = F while ignoring the zero-eigenvalue 'null space'.
    u_static = pinvh(K) @ F 

    # 3. Compute the Contribution of the Synthesized Modes
    # We must subtract the modes we ALREADY used in the 6-mode synthesis.
    # index 0, 1: Rigid Body Modes (Skip because they have no 'stiffness').
    # index 2-7: These are your 6 bending modes used in synthesis.
    u_modal_sum = np.zeros(n_dofs)
    
    for i in range(2, 8): 
        phi = eigenvectors_full[:, i]
        lam = eigenvalues_raw[i] # This is omega_i^2
        
        # Static contribution of mode i: (phi_i * phi_i^T * F) / omega_i^2
        # Since F has only one non-zero at [impact_dof], this simplifies:
        u_modal_sum += (phi[impact_dof] * phi) / lam

    # 4. Residual = Total - Modal Contribution
    u_residual = u_static - u_modal_sum
    
    return u_residual