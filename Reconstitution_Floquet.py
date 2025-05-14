import numpy as np
import matplotlib.pyplot as plt
import os

# Constants
Lx = 1.0  # Unit cell size in x-direction (meters)
Ly = 1.0  # Unit cell size in y-direction (meters)
f = 300e6  # Frequency (300 MHz)
c = 3e8  # Speed of light (m/s)
k0 = 2 * np.pi * f / c  # Free-space wavenumber
theta_incident = np.pi/4  # Incident angle (45 degrees, more realistic than 90)
phi_incident = 0  # Azimuthal angle
ki = np.array([k0 * np.sin(theta_incident) * np.cos(phi_incident),
               k0 * np.sin(theta_incident) * np.sin(phi_incident),
               k0 * np.cos(theta_incident)])
d_observation = 10 * c/f  # Observation distance

def parse_sparam_file(filename, target_freq=300e6):
    """
    Parse S-parameter file and extract data for target frequency with dynamic matrix sizing.

    Args:
        filename (str): Path to the S-parameter file (.tab)
        target_freq (float, optional): Target frequency to extract. Defaults to 300 MHz.

    Returns:
        np.ndarray: Complex S-parameter matrix
    """
    # Synthetic data generation for file not found
    if not os.path.exists(filename):
        print(f"File {filename} not found. Generating synthetic S-parameters.")
        # Determine matrix size dynamically or use a default
        default_size = 14
        s_matrix = np.zeros((default_size, default_size), dtype=complex)

        # Example synthetic initialization
        s_matrix[0, 0] = 0.1 * np.exp(1j * np.pi/4)  # Reflection of 00 TE mode
        s_matrix[2, 0] = 0.4 * np.exp(1j * np.pi/3)  # Strong 00->0(-1) coupling
        s_matrix[4, 0] = 0.3 * np.exp(-1j * np.pi/6)  # Strong 00->(-1)0 coupling
        return s_matrix

    # Read file contents
    with open(filename, 'r') as f:
        lines = f.readlines()

    # Find header row
    header_line = None
    for i, line in enumerate(lines):
        if line.strip().startswith('Frequency'):
            header_line = i
            break

    if header_line is None:
        raise ValueError("Could not find header line in S-parameter file")

    # Parse header to determine columns dynamically
    header_parts = lines[header_line].strip().split()
    column_map = {}
    max_index = 0

    for col_idx, header in enumerate(header_parts):
        if header.startswith('S[') and (header.endswith('_Mag') or header.endswith('_Phs')):
            # Extract indices and type from header like "S[1,2]_Mag"
            s_indices = header[header.find('[')+1:header.find(']')].split(',')
            i = int(s_indices[0])
            j = int(s_indices[1])

            # Track the maximum index to determine matrix size
            max_index = max(max_index, i, j)

            param_type = header.split('_')[1]  # 'Mag' or 'Phs'

            # Store the column index
            if (i, j) not in column_map:
                column_map[(i, j)] = {}
            column_map[(i, j)][param_type] = col_idx

    # Extract data lines
    data_lines = [line.strip().split() for line in lines[header_line + 1:] if line.strip()]

    # Find the row with our target frequency
    freq_idx = None
    for i, line in enumerate(data_lines):
        if len(line) > 0 and abs(float(line[0]) - target_freq) < 1e6:  # Within 1 MHz
            freq_idx = i
            break

    if freq_idx is None:
        raise ValueError(f"Target frequency {target_freq} not found in file")

    # Extract S-parameters for this frequency
    data_line = data_lines[freq_idx]

    # Dynamically size the S-matrix based on max indices found
    s_matrix = np.zeros((max_index, max_index), dtype=complex)

    # Fill in the S-matrix
    for (i, j), cols in column_map.items():
        if 'Mag' in cols and 'Phs' in cols:
            try:
                mag = float(data_line[cols['Mag']])
                phase_deg = float(data_line[cols['Phs']])
                # Convert to complex number and store in matrix (adjust indices to 0-based)
                s_matrix[i-1, j-1] = mag * np.exp(1j * np.deg2rad(phase_deg))
            except (IndexError, ValueError) as e:
                print(f"Warning: Could not process S-parameter S[{i},{j}]: {e}")

    return s_matrix

def calculate_wave_vector(m, n, k0, ki):
    """Calculate wave vector components for a given Floquet mode (m,n)."""
    # Extract incident wave components
    kx_inc = ki[0]
    ky_inc = ki[1]

    # Calculate Floquet mode wave vector components
    kx = kx_inc + 2 * np.pi * m / Lx
    ky = ky_inc + 2 * np.pi * n / Ly

    # Calculate z-component ensuring k² = kx² + ky² + kz²
    under_sqrt = k0**2 - kx**2 - ky**2

    if under_sqrt >= 0:
        # Propagating wave
        kz = np.sqrt(under_sqrt)
        is_evanescent = False
    else:
        # Evanescent wave
        kz = 1j * np.sqrt(-under_sqrt)
        is_evanescent = True

    k_vector = np.array([kx, ky, kz])
    return k_vector, is_evanescent

def calculate_polarization_vectors(k_vector, polarization):
    """Calculate polarization unit vectors for TE and TM modes."""
    # Normalize k vector
    k_norm = np.linalg.norm(k_vector)
    if k_norm == 0:
        raise ValueError("Zero wave vector encountered")

    k_hat = k_vector / k_norm

    # Handle special case where k is along z-axis
    if np.abs(k_hat[0]) < 1e-10 and np.abs(k_hat[1]) < 1e-10:
        # k is parallel to z-axis, choose arbitrary perpendicular vectors
        e_theta = np.array([1, 0, 0])
        e_phi = np.array([0, 1, 0])
    else:
        # Standard case
        # For TE mode: e_theta is perpendicular to both k and z
        z_axis = np.array([0, 0, 1])
        e_theta = np.cross(k_hat, z_axis)
        e_theta = e_theta / np.linalg.norm(e_theta)

        # For TM mode: e_phi is perpendicular to both k and e_theta
        e_phi = np.cross(k_hat, e_theta)
        e_phi = e_phi / np.linalg.norm(e_phi)

    if polarization == 'TE':
        return e_theta
    elif polarization == 'TM':
        return e_phi
    else:
        raise ValueError(f"Unknown polarization: {polarization}")

def print_mode_info(floquet_modes):
    """Print information about each Floquet mode."""
    print("\n===== Floquet Mode Information =====")
    for mode_num, mode in floquet_modes.items():
        if 'k' in mode:
            k_vector = mode['k']

            # Calculate propagation angle
            if not mode['is_evanescent']:
                # For propagating modes, calculate theta angle (from z-axis)
                kx, ky, kz = k_vector
                transverse_mag = np.sqrt(np.real(kx)**2 + np.real(ky)**2)
                theta = np.arctan2(transverse_mag, np.real(kz))
                theta_deg = np.rad2deg(theta)

                # Calculate phi angle (azimuthal)
                phi = np.arctan2(np.real(ky), np.real(kx))
                phi_deg = np.rad2deg(phi)

                angle_info = f"θ={theta_deg:.1f}°, φ={phi_deg:.1f}°"
            else:
                angle_info = "evanescent"

            print(f"Mode {mode_num}: ({mode['m']},{mode['n']}) {mode['polarization']}, {angle_info}")

            if 'amplitude' in mode:
                db_amplitude = 20 * np.log10(np.abs(mode['amplitude']))
                print(f"  Amplitude: {np.abs(mode['amplitude']):.4f} ({db_amplitude:.1f} dB)")
        else:
            print(f"Mode {mode_num}: ({mode['m']},{mode['n']}) {mode['polarization']}")
    print("=====================================\n")

def calculate_Esource(floquet_modes, resolution, s_matrix):

    Esource = np.zeros((resolution, resolution, 3), dtype=np.complex128)
    y, z = np.linspace(0, 1, resolution), np.linspace(0, 1, resolution)
    Y, Z = np.meshgrid(y, z, indexing='ij')

    for mode_num, mode in floquet_modes.items():
        k_vector, is_evanescent = calculate_wave_vector(mode['m'], mode['n'], k0, ki)
        polarization_vector = calculate_polarization_vectors(k_vector, mode['polarization'])

        # Add wave vector, polarization vector, and evanescent flag to the mode
        floquet_modes[mode_num]['k'] = k_vector
        floquet_modes[mode_num]['polarization_vector'] = polarization_vector
        floquet_modes[mode_num]['is_evanescent'] = is_evanescent

        if not is_evanescent and mode['polarization'] == 'TM':
            # Calculate the contribution of this mode to Esource
            mode_contribution = s_matrix[mode_num-1, 0] * np.exp(-1j * (k_vector[1] * Y + k_vector[2] * Z))
            Esource += mode_contribution[:, :, np.newaxis] * polarization_vector

    return Esource

def create_N_mask(Npoint):
    thickness = max(1, Npoint // 10)  # Ensure thickness is at least 1
    mask = np.zeros((Npoint, Npoint))
    
    # Left vertical line
    mask[:, 0:thickness] = 1
    
    # Right vertical line
    mask[:, Npoint-thickness:Npoint] = 1
    
    # Diagonal line - more precise implementation
    for i in range(Npoint):
        # Calculate start and end points of the diagonal
        start_x = thickness
        end_x = Npoint - thickness
        
        # Calculate the diagonal position with precise floating-point math
        # This gives the exact position where the diagonal should be for this row
        exact_pos = start_x + (i / (Npoint - 1)) * (end_x - start_x)
        
        # Apply the thickness centered around the exact position
        half_t = thickness // 2
        for t in range(-half_t, thickness - half_t):
            pos = int(exact_pos + t)
            if 0 <= pos < Npoint:  # Ensure we're within bounds
                mask[i, pos] = 1
    return mask

def calculate_currents_on_mask(Esource, mask):

    normal_vector = np.array([-1, 0, 0])
    mask_3d = mask[:, :, np.newaxis]
    currents = -np.cross(normal_vector, Esource)*mask_3d
      
    return currents

def calculate_fields(M, currents, mask, resolution, f, c, mu, epsilon):
    """
    Calculate the electric and magnetic fields at observation points using the Schelkunoff formulas.
    
    Parameters:
    - M: Observation point coordinates - can be a single point (3,) or multiple points (N,3)
    - currents: Current distribution on the mask (array-like, shape (resolution, resolution, 3))
    - mask: Mask defining the surface (array-like, shape (resolution, resolution))
    - resolution: Resolution of the mask
    - f: Frequency (Hz)
    - c: Speed of light (m/s)
    - mu: Permeability of free space (H/m)
    - epsilon: Permittivity of free space (F/m)
    
    Returns:
    - E: Electric field at the observation point(s)
    - H: Magnetic field at the observation point(s)
    """
    # Define constants
    k = 2 * np.pi * f / c  # Wavenumber
    zeta = np.sqrt(mu / epsilon)  # Intrinsic impedance of free space
    
    # Check if M is a single point or multiple points
    if M.ndim == 1:
        # Single observation point
        return calculate_field_single_point(M, currents, mask, resolution, k, zeta)
    else:
        # Multiple observation points
        E_list = []
        H_list = []
        for i in range(M.shape[0]):
            E_point, H_point = calculate_field_single_point(M[i], currents, mask, resolution, k, zeta)
            E_list.append(E_point)
            H_list.append(H_point)
        return np.array(E_list), np.array(H_list)

def calculate_field_single_point(M, currents, mask, resolution, k, zeta):
    """
    Helper function to calculate fields at a single observation point.
    """
    # Calculate the distance and unit vector from the origin to M
    r = np.linalg.norm(M)
    r_hat = M / r
    
    # Find the indices of the mask where the current is non-zero
    y, z = np.where(mask == 1)
    
    # Calculate the coordinates of the points on the mask in the OYZ plane
    M_prime = np.column_stack((np.zeros_like(y), y / resolution, z / resolution))
    
    # Extract the current values at these points
    J_e_M_prime = currents[y, z]
    
    # Calculate the phase factor for all points on the mask
    # For single point r_hat, use proper dot product approach
    phase_factors = np.exp(1j * k * np.sum(M_prime * r_hat, axis=1))
    
    # Calculate the integrals using vectorization
    integral_E = np.sum(J_e_M_prime * phase_factors[:, np.newaxis], axis=0)
    integral_H = np.sum(J_e_M_prime * phase_factors[:, np.newaxis], axis=0)
    
    # Calculate the electric field E(M)
    E = 1j * k * zeta * (np.exp(-1j * k * r) / (4 * np.pi * r)) * np.cross(r_hat, np.cross(r_hat, integral_E))
    
    # Calculate the magnetic field H(M)
    H = -1j * k * (np.exp(-1j * k * r) / (4 * np.pi * r)) * np.cross(r_hat, integral_H)
    
    return E, H

def plot_far_field_radiated_OXZ(field_source, resolution=100, num_angles=180):
    """
    Plot the far-field radiation pattern in the OXZ plane.
    
    Parameters:
    - field_source: Source electric field distribution
    - resolution: Resolution of the grid
    - num_angles: Number of angles to compute the far field
    """
    # Constants
    mu = 4 * np.pi * 1e-7  # Permeability of free space (H/m)
    epsilon = 8.85e-12     # Permittivity of free space (F/m)
    
    # Create the mask
    mask = create_N_mask(resolution)
    
    # Calculate currents on the mask (use the fixed version)
    currents = calculate_currents_on_mask(field_source, mask)
    
    # Define observation angles in the OXZ plane (phi = 0)
    theta_linspace = np.linspace(0, 2*np.pi, num_angles)
    
    # Calculate all observation points
    x = d_observation * np.cos(theta_linspace)
    z = d_observation * np.sin(theta_linspace)
    y = np.zeros_like(x)  # y=0 for OXZ plane
    
    # Observation points as a matrix
    M = np.column_stack((x, y, z))  # Shape: (num_angles, 3)
    
    # Calculate fields at each observation point
    E_magnitudes = []
    H_magnitudes = []
    
    # Compute fields for each point individually to avoid matrix dimension mismatch
    for i in range(len(M)):
        E, H = calculate_fields(M[i], currents, mask, resolution, f, c, mu, epsilon)
        E_magnitudes.append(np.linalg.norm(E))
        H_magnitudes.append(np.linalg.norm(H))
    
    # Convert to numpy arrays
    E_magnitudes = np.array(E_magnitudes)
    H_magnitudes = np.array(H_magnitudes)
    
    # Normalize field magnitudes for the plot
    E_max = np.max(E_magnitudes)
    H_max = np.max(H_magnitudes)
    
    E_norm = E_magnitudes / E_max if E_max > 0 else E_magnitudes
    H_norm = H_magnitudes / H_max if H_max > 0 else H_magnitudes
    
    # Convert to dB scale
    E_db = 20 * np.log10(E_norm + 1e-10)
    H_db = 20 * np.log10(H_norm + 1e-10)
    
    # Create polar plot
    fig, ax = plt.subplots(1, 1, subplot_kw={'projection': 'polar'}, figsize=(10, 8))
    
    # Plot electric field magnitude
    ax.plot(theta_linspace, E_db, 'r-', linewidth=2, label='Electric Field (E)')
    
    # Plot magnetic field magnitude
    ax.plot(theta_linspace, H_db, 'b--', linewidth=2, label='Magnetic Field (H)')
    
    # Set plot limits and labels
    ax.set_rticks([-30, -20, -10, 0])  # dB scale
    ax.set_rlim([-40, 5])
    ax.set_title('Far-Field Radiation Pattern (OXZ Plane)', pad=20)
    ax.grid(True)
    ax.legend(loc='upper right')
    plt.tight_layout()
    return fig

def plot_far_field_cartesian(field_source, resolution=100, num_angles=180):
    """
    Plot the far-field radiation pattern in the OXZ plane using Cartesian coordinates.
    
    Parameters:
    - field_source: Source electric field distribution
    - resolution: Resolution of the grid
    - num_angles: Number of angles to compute the far field
    
    Returns:
    - fig: The generated figure
    """
    # Constants
    mu = 4 * np.pi * 1e-7  # Permeability of free space (H/m)
    epsilon = 8.85e-12     # Permittivity of free space (F/m)
    
    # Create the mask
    mask = create_N_mask(resolution)
    
    # Calculate currents on the mask
    currents = calculate_currents_on_mask(field_source, mask)
    
    # Define observation angles in the OXZ plane (phi = 0)
    theta_linspace = np.linspace(0, 2*np.pi, num_angles)
    
    # Calculate all observation points
    x = d_observation * np.cos(theta_linspace)
    z = d_observation * np.sin(theta_linspace)
    y = np.zeros_like(x)  # y=0 for OXZ plane
    
    # Observation points as a matrix
    M = np.column_stack((x, y, z))  # Shape: (num_angles, 3)
    
    # Calculate fields at each observation point
    E_magnitudes = []
    H_magnitudes = []
    
    # Compute fields for each point individually
    for i in range(len(M)):
        E, H = calculate_fields(M[i], currents, mask, resolution, f, c, mu, epsilon)
        E_magnitudes.append(np.linalg.norm(E))
        H_magnitudes.append(np.linalg.norm(H))
    
    # Convert to numpy arrays
    E_magnitudes = np.array(E_magnitudes)
    H_magnitudes = np.array(H_magnitudes)
    
    # Normalize field magnitudes for the plot
    E_max = np.max(E_magnitudes)
    H_max = np.max(H_magnitudes)
    
    E_norm = E_magnitudes / E_max if E_max > 0 else E_magnitudes
    H_norm = H_magnitudes / H_max if H_max > 0 else H_magnitudes
    
    # Convert to dB scale
    E_db = 20 * np.log10(E_norm + 1e-10)
    H_db = 20 * np.log10(H_norm + 1e-10)
    
    # Create Cartesian plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Convert angles to degrees for better readability
    theta_degrees = np.rad2deg(theta_linspace)
    
    # Plot electric field magnitude
    ax.plot(theta_degrees, E_db, 'r-', linewidth=2, label='Electric Field (E)')
    
    # Plot magnetic field magnitude
    ax.plot(theta_degrees, H_db, 'b--', linewidth=2, label='Magnetic Field (H)')
    
    # Set plot limits and labels
    ax.set_xlim([0, 360])
    ax.set_ylim([-40, 5])
    ax.set_xlabel('Angle (degrees)', fontsize=12)
    ax.set_ylabel('Magnitude (dB)', fontsize=12)
    ax.set_title('Far-Field Radiation Pattern (OXZ Plane)', fontsize=14)
    ax.grid(True)
    ax.legend(loc='best')
    
    # Add vertical lines at notable angles
    for angle in [0, 90, 180, 270, 360]:
        ax.axvline(x=angle, color='gray', linestyle=':', alpha=0.7)
    
    # Add horizontal lines at notable dB levels
    for db_level in [-30, -20, -10, 0]:
        ax.axhline(y=db_level, color='gray', linestyle=':', alpha=0.7)
    
    plt.tight_layout()
    return fig