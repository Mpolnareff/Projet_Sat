import numpy as np
import matplotlib.pyplot as plt

# Constants
Lx = 1.0  # Unit cell size in x-direction (meters)
Ly = 1.0  # Unit cell size in z-direction (meters)
Lz = 2.0  # Unit cell size in y-direction (meters) - swapped with Lz from original
f = 200e6  # Frequency (200 MHz)
c = 3e8  # Speed of light (m/s)
k0 = 2 * np.pi * f / c  # Free-space wavenumber
# Changed incident angle to go in +Y direction
theta_incident = np.pi/2  # 90 degrees (along Y-axis)
phi_incident = np.pi/2   
ki = np.array([k0 * np.sin(theta_incident) * np.cos(phi_incident),
               k0 * np.sin(theta_incident) * np.sin(phi_incident),
               k0 * np.cos(theta_incident)])
d_observation = 10 * c/f  # Observation distance

def parse_sparam_file(filename, target_freq):
    """Parse S-parameter file or generate synthetic data if file not found."""
    import os
    import numpy as np
    
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
    
    # Detect frequency unit based on first frequency value and reasonable ranges
    if len(data_lines) > 0:
        first_freq = float(data_lines[0][0])
        if first_freq < 1000:  # Likely in GHz
            freq_scale = 1e9
        elif first_freq < 1000000:  # Likely in MHz
            freq_scale = 1e6
        else:  # Likely in Hz
            freq_scale = 1
        
        # Convert target frequency to the same unit as in the file
        scaled_target = target_freq / freq_scale
    else:
        raise ValueError("No data found in file")
    
    # Find the row with closest match to our target frequency
    closest_idx = None
    min_diff = float('inf')
    
    for i, line in enumerate(data_lines):
        if len(line) > 0:
            file_freq = float(line[0])
            diff = abs(file_freq - scaled_target)
            if diff < min_diff:
                min_diff = diff
                closest_idx = i
    
    if closest_idx is None:
        raise ValueError(f"No frequency data found in file")
    
    # Get the actual frequency found (for reporting)
    actual_freq = float(data_lines[closest_idx][0]) * freq_scale
    if abs(actual_freq - target_freq) > target_freq * 0.05:  # More than 5% different
        print(f"Warning: Requested frequency {target_freq/1e6:.2f} MHz not found. " 
              f"Using closest available: {actual_freq/1e6:.2f} MHz")
    
    # Extract S-parameters for this frequency
    data_line = data_lines[closest_idx]
    
    # Dynamically size the S-matrix based on max indices found
    s_matrix = np.zeros((max_index + 1, max_index + 1), dtype=complex)
    
    # Fill in the S-matrix (indices are 1-based in the file but 0-based in the matrix)
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
    kz_inc = ki[2]

    # Calculate Floquet mode wave vector components
    # Modified to handle OXZ plane for metasurface
    kx = kx_inc + 2 * np.pi * m / Lx
    kz = kz_inc + 2 * np.pi * n / Lz

    # Calculate y-component ensuring k² = kx² + ky² + kz²
    under_sqrt = k0**2 - kx**2 - kz**2

    if under_sqrt >= 0:
        # Propagating wave
        ky = np.sqrt(under_sqrt)
        is_evanescent = False
    else:
        # Evanescent wave
        ky = 1j * np.sqrt(-under_sqrt)
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

    # Changed for OXZ plane with normal in Y direction
    # Handle special case where k is along x-axis
    if np.abs(k_hat[0]) < 1e-10 and np.abs(k_hat[2]) < 1e-10:
        # k is parallel to y-axis, choose arbitrary perpendicular vectors
        e_theta = np.array([1, 0, 0])  # x-axis
        e_phi = np.array([0, 0, 1])    # z-axis
    else:
        # Standard case
        # For TE mode: e_theta is perpendicular to both k and y-axis (the normal)
        y_axis = np.array([0, 1, 0])
        e_theta = np.cross(k_hat, y_axis) 
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
                # For propagating modes, calculate angles
                kx, ky, kz = k_vector
                # Modified for Y as normal direction
                transverse_mag = np.sqrt(np.real(kx)**2 + np.real(kz)**2)
                theta = np.arctan2(transverse_mag, np.real(ky))
                theta_deg = np.rad2deg(theta)

                # Calculate phi angle (azimuthal in XZ plane)
                phi = np.arctan2(np.real(kz), np.real(kx))
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
    """Calculate the electric field at the source (metasurface)."""
    # For OXZ plane
    Esource = np.zeros((resolution, resolution, 3), dtype=np.complex128)
    x, z = np.linspace(0, Lx, resolution), np.linspace(0, Lz, resolution)
    X, Z = np.meshgrid(x, z, indexing='ij')

    for mode_num, mode in floquet_modes.items():
        k_vector, is_evanescent = calculate_wave_vector(mode['m'], mode['n'], k0, ki)
        polarization_vector = calculate_polarization_vectors(k_vector, mode['polarization'])

        # Add wave vector, polarization vector, and evanescent flag to the mode
        floquet_modes[mode_num]['k'] = k_vector
        floquet_modes[mode_num]['polarization_vector'] = polarization_vector
        floquet_modes[mode_num]['is_evanescent'] = is_evanescent

        if not is_evanescent and mode['polarization'] == 'TM':
            # Calculate the contribution of this mode to Esource
            # Modified for OXZ plane
            mode_contribution = s_matrix[mode_num-1, 0] * np.exp(-1j * (k_vector[0] * X + k_vector[2] * Z))
            Esource += mode_contribution[:, :, np.newaxis] * polarization_vector

    return Esource


def calculate_currents_on_mask(Esource, mask):
    """Calculate the induced electric currents on the metasurface."""
    # Changed normal vector to point in +Y direction (OXZ plane)
    normal_vector = np.array([0, 1, 0])
    mask_3d = mask[:, :, np.newaxis]
    currents = -np.cross(normal_vector, Esource) * mask_3d
      
    return currents

def calculate_fields(M, currents, mask, resolution, f, c, epsilon):
    """
    Calculate the electric field at observation points using the Schelkunoff formulas.
    
    Parameters:
    - M: Observation point coordinates - can be a single point (3,) or multiple points (N,3)
    - currents: Current distribution on the mask (array-like, shape (resolution, resolution, 3))
    - mask: Mask defining the surface (array-like, shape (resolution, resolution))
    - resolution: Resolution of the grid
    - f: Frequency (Hz)
    - c: Speed of light (m/s)
    - epsilon: Permittivity of free space (F/m)
    
    Returns:
    - E: Electric field at the observation point(s)
    """
    # Define constants
    k = 2 * np.pi * f / c  # Wavenumber
    mu = 4 * np.pi * 1e-7  # Permeability of free space (H/m)
    zeta = np.sqrt(mu / epsilon)  # Intrinsic impedance of free space
    
    # Check if M is a single point or multiple points
    if M.ndim == 1:
        # Single observation point
        return calculate_field_single_point(M, currents, mask, resolution, k, zeta)
    else:
        # Multiple observation points
        E_list = []
        for i in range(M.shape[0]):
            E_point = calculate_field_single_point(M[i], currents, mask, resolution, k, zeta)
            E_list.append(E_point)
        return np.array(E_list)

def calculate_field_single_point(M, currents, mask, resolution, k, zeta):
    """
    Helper function to calculate electric field at a single observation point.
    """
    # Calculate the distance and unit vector from the origin to M
    r = np.linalg.norm(M)
    r_hat = M / r
    
    # Find the indices of the mask where the current is non-zero
    x, z = np.where(mask == 1)
    
    # Calculate the coordinates of the points on the mask in the OXZ plane
    # Modified for OXZ plane
    M_prime = np.column_stack((x / resolution * Lx, np.zeros_like(x), z / resolution * Lz))
    
    # Extract the current values at these points
    J_e_M_prime = currents[x, z]
    
    # Calculate the phase factor for all points on the mask
    phase_factors = np.exp(1j * k * np.sum(M_prime * r_hat, axis=1))
    
    # Calculate the integrals using vectorization
    integral_E = np.sum(J_e_M_prime * phase_factors[:, np.newaxis], axis=0)
    
    # Calculate the electric field E(M)
    E = 1j * k * zeta * (np.exp(-1j * k * r) / (4 * np.pi * r)) * np.cross(r_hat, np.cross(r_hat, integral_E))
    
    return E

def calculate_fields_vectorized(M_points, currents, mask, resolution, f, c, epsilon):
    """
    Vectorized calculation of electric fields at multiple observation points.
    
    Parameters:
    - M_points: Array of observation point coordinates, shape (N, 3)
    - currents: Current distribution on the mask (array-like, shape (resolution, resolution, 3))
    - mask: Mask defining the surface (array-like, shape (resolution, resolution))
    - resolution: Resolution of the grid
    - f: Frequency (Hz)
    - c: Speed of light (m/s)
    - epsilon: Permittivity of free space (F/m)
    
    Returns:
    - E_fields: Electric fields at the observation points, shape (N, 3)
    """
    # Define constants
    k = 2 * np.pi * f / c  # Wavenumber
    mu = 4 * np.pi * 1e-7  # Permeability of free space (H/m)
    zeta = np.sqrt(mu / epsilon)  # Intrinsic impedance of free space
    
    # Find the indices of the mask where the current is non-zero
    x_indices, z_indices = np.where(mask == 1)
    
    # Calculate the coordinates of the points on the mask in the OXZ plane
    # Modified for OXZ plane
    M_prime = np.column_stack((x_indices / resolution * Lx, np.zeros_like(x_indices), z_indices / resolution * Lz))
    
    # Extract the current values at these points
    J_values = currents[x_indices, z_indices]  # Shape: (num_points, 3)
    
    # Initialize arrays for E fields
    num_obs_points = M_points.shape[0]
    E_fields = np.zeros((num_obs_points, 3), dtype=complex)
    
    # For each observation point
    for i in range(num_obs_points):
        M = M_points[i]
        
        # Calculate the distance from observation point to origin
        r = np.linalg.norm(M)
        r_hat = M / r
        
        # Calculate the phase factors for all source points
        # r_M - r_M'
        r_diff_vectors = M - M_prime  # Shape: (num_source_points, 3)
        r_diff_mags = np.linalg.norm(r_diff_vectors, axis=1)  # Shape: (num_source_points,)
        r_diff_hats = r_diff_vectors / r_diff_mags[:, np.newaxis]  # Shape: (num_source_points, 3)
        
        # For far field approximation, we can use:
        # |r - r'| ≈ r - r̂·r' for phase
        # |r - r'| ≈ r for amplitude
        phase_approx = r - np.sum(r_hat * M_prime, axis=1)  # Shape: (num_source_points,)
        phase_factors = np.exp(-1j * k * phase_approx)  # Shape: (num_source_points,)
        
        # Calculate the field contribution from each source point
        amplitude_factor = np.exp(-1j * k * r) / (4 * np.pi * r)
        
        # Sum the contributions
        J_phase = J_values * phase_factors[:, np.newaxis]  # Shape: (num_source_points, 3)
        integral_result = np.sum(J_phase, axis=0)  # Shape: (3,)
        
        # Calculate E field
        E = 1j * k * zeta * amplitude_factor * np.cross(r_hat, np.cross(r_hat, integral_result))
        
        E_fields[i] = E
    
    return E_fields

def plot_far_field_radiated_OYZ(field_source, mask, resolution=100, num_angles=180):
    """
    Plot the far-field radiation pattern in the OYZ plane with vectorized calculations.
    """
    # Constants
    epsilon = 8.85e-12     # Permittivity of free space (F/m)

    # Calculate currents on the mask
    currents = calculate_currents_on_mask(field_source, mask)

    # Define observation angles in the OYZ plane (phi = 90°)
    theta_linspace = np.linspace(0, np.pi, num_angles)

    # Calculate all observation points
    y = d_observation * np.cos(theta_linspace)
    z = d_observation * np.sin(theta_linspace)
    x = np.zeros_like(y)  # x=0 for OYZ plane

    # Observation points as a matrix
    M = np.column_stack((x, y, z))  # Shape: (num_angles, 3)

    # Vectorized field calculation
    E_fields = calculate_fields_vectorized(M, currents, mask, resolution, f, c, epsilon)

    # Calculate field magnitudes
    E_magnitudes = np.linalg.norm(E_fields, axis=1)

    # Print field magnitudes for debugging
    print("E_magnitudes:", E_magnitudes)

    # Normalize field magnitudes for the plot
    E_max = np.max(E_magnitudes)

    print("E_max:", E_max)

    E_norm = E_magnitudes / E_max if E_max > 0 else E_magnitudes

    # Convert to dB scale
    E_db = 20 * np.log10(E_norm + 1e-10)

    # Create polar plot
    fig, ax = plt.subplots(1, 1, subplot_kw={'projection': 'polar'}, figsize=(10, 8))

    # Plot electric field magnitude
    ax.plot(theta_linspace, E_db, 'r-', linewidth=2, label='Electric Field (E)')

    # Set plot limits and labels
    ax.set_rticks([-30, -20, -10, 0])  # dB scale
    ax.set_rlim([-40, 5])
    ax.set_title('Far-Field Radiation Pattern (OYZ Plane)', pad=20)
    ax.grid(True)
    ax.legend(loc='upper right')
    plt.tight_layout()
    return fig

def plot_far_field_cartesian(field_source, mask, resolution=100, num_angles=180):
    """
    Plot the far-field radiation pattern in the OYZ plane using Cartesian coordinates with vectorized calculations.
    """
    # Constants
    epsilon = 8.85e-12     # Permittivity of free space (F/m)

    # Calculate currents on the mask
    currents = calculate_currents_on_mask(field_source, mask)

    # Define observation angles in the OYZ plane (phi = 90°)
    theta_linspace = np.linspace(0, np.pi, num_angles)

    # Calculate all observation points for OYZ plane
    y = d_observation * np.cos(theta_linspace)
    z = d_observation * np.sin(theta_linspace)
    x = np.zeros_like(y)  # x=0 for OYZ plane

    # Observation points as a matrix
    M = np.column_stack((x, y, z))  # Shape: (num_angles, 3)

    # Vectorized field calculation
    E_fields = calculate_fields_vectorized(M, currents, mask, resolution, f, c, epsilon)

    # Calculate field magnitudes
    E_magnitudes = np.linalg.norm(E_fields, axis=1)

    # Print field magnitudes for debugging
    print("E_magnitudes:", E_magnitudes)

    # Normalize field magnitudes for the plot
    E_max = np.max(E_magnitudes)

    print("E_max:", E_max)

    E_norm = E_magnitudes / E_max if E_max > 0 else E_magnitudes

    # Convert to dB scale
    E_db = 20 * np.log10(E_norm + 1e-10)

    # Create Cartesian plot
    fig, ax = plt.subplots(figsize=(12, 8))

    # Convert angles to degrees for better readability
    theta_degrees = np.rad2deg(theta_linspace)

    # Plot electric field magnitude
    ax.plot(theta_degrees, E_db, 'r-', linewidth=2, label='Electric Field (E)')

    # Set plot limits and labels
    ax.set_xlim([0, 180])
    ax.set_ylim([-60, 5])
    ax.set_xlabel('Angle (degrees)', fontsize=12)
    ax.set_ylabel('Magnitude (dB)', fontsize=12)
    ax.set_title('Far-Field Radiation Pattern (OYZ Plane)', fontsize=14)
    ax.grid(True)
    ax.legend(loc='best')

    # Add vertical lines at notable angles
    for angle in [0, 90, 180]:
        ax.axvline(x=angle, color='gray', linestyle=':', alpha=0.7)

    # Add horizontal lines at notable dB levels
    for db_level in [-30, -20, -10, 0]:
        ax.axhline(y=db_level, color='gray', linestyle=':', alpha=0.7)

    plt.tight_layout()
    return fig

def calculate_stacked_array_field(field_source, mask, num_blocks, resolution=100, num_angles=180, element_spacing=None):
    """
    Optimized version to calculate the total E-field for multiple blocks stacked along the z-axis.
    
    Parameters:
    - field_source: Electric field at the source (single unit block)
    - mask: Mask defining the surface (array-like, shape (resolution, resolution))
    - num_blocks: Number of blocks to stack along z-axis
    - resolution: Resolution of the grid
    - num_angles: Number of angles to calculate in the far-field pattern
    - element_spacing: Spacing between elements in meters (default: 1.0)
    
    Returns:
    - total_E_fields: Total electric fields at observation points
    - M: Observation points
    - theta_degrees: Observation angles in degrees
    """
    # Constants
    epsilon = 8.85e-12     # Permittivity of free space (F/m)

    
    # If element spacing is not provided, use default
    if element_spacing is None:
        element_spacing = Lz  # Default to 1.0 meters
    
    # Calculate currents on the mask for a single block
    currents = calculate_currents_on_mask(field_source, mask)
    
    # Define observation angles in the OYZ plane (phi = 90°)
    theta_linspace = np.linspace(0, np.pi, num_angles)
    theta_degrees = np.rad2deg(theta_linspace)
    
    # Calculate all observation points for OYZ plane
    y = d_observation * np.cos(theta_linspace)
    z = d_observation * np.sin(theta_linspace)
    x = np.zeros_like(y)  # x=0 for OYZ plane
    
    # Observation points as a matrix
    M = np.column_stack((x, y, z))  # Shape: (num_angles, 3)
    
    # Calculate E-fields for the single block - do this only once!
    single_block_E_fields = calculate_fields_vectorized(M, currents, mask, resolution, f, c, epsilon)
    
    # Pre-calculate all phase shifts for all blocks and angles at once
    # Create array of block offsets [0, 1, 2, ..., num_blocks-1]
    block_indices = np.arange(num_blocks)
    # Calculate z-offset for each block: [0, spacing, 2*spacing, ...]
    z_offsets = block_indices * element_spacing
    
    # Use outer product to calculate path differences for all combinations of blocks and angles
    # sin_theta has shape (num_angles,)
    sin_theta = np.sin(theta_linspace)
    # This creates a matrix of shape (num_blocks, num_angles)
    path_differences = np.outer(z_offsets, sin_theta)
    
    # Calculate phase shifts for all blocks and angles: exp(j*k0*path_difference)
    # This has shape (num_blocks, num_angles)
    phase_shifts = np.exp(1j * k0 * path_differences)
    
    # Initialize total E-fields
    total_E_fields = np.zeros((num_angles, 3), dtype=complex)
    
    # Apply phase shifts to each block's fields and sum them
    for i in range(num_blocks):
        # For each block, we apply its phase shifts to all angles
        for angle_idx in range(num_angles):
            total_E_fields[angle_idx] += single_block_E_fields[angle_idx] * phase_shifts[i, angle_idx]
    
    return total_E_fields, M, theta_degrees

def plot_stacked_array_field(total_E_fields, M, theta_degrees, plot_type='polar'):
    """
    Plot the far-field radiation pattern for the stacked array.
    
    Parameters:
    - total_E_fields: Total electric fields at observation points
    - M: Observation points
    - theta_degrees: Observation angles in degrees
    - plot_type: Type of plot ('polar' or 'cartesian')
    
    Returns:
    - fig: Figure object
    """
    # Calculate field magnitudes
    E_magnitudes = np.linalg.norm(total_E_fields, axis=1)
    
    # Normalize field magnitudes for the plot
    E_max = np.max(E_magnitudes)
    E_norm = E_magnitudes / E_max if E_max > 0 else E_magnitudes
    
    # Convert to dB scale
    E_db = 20 * np.log10(E_norm + 1e-10)
    
    if plot_type == 'polar':
        # Create polar plot
        fig, ax = plt.subplots(1, 1, subplot_kw={'projection': 'polar'}, figsize=(10, 8))
        
        # Plot electric field magnitude
        ax.plot(np.deg2rad(theta_degrees), E_db, 'r-', linewidth=2, label='Electric Field (E)')
        
        # Set plot limits and labels
        ax.set_rticks([-30, -20, -10, 0])  # dB scale
        ax.set_rlim([-40, 5])
        ax.set_title('Far-Field Radiation Pattern (OYZ Plane) - Stacked Array', pad=20)
        ax.grid(True)
        ax.legend(loc='upper right')
        
    elif plot_type == 'cartesian':
        # Create Cartesian plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plot electric field magnitude
        ax.plot(theta_degrees, E_db, 'r-', linewidth=2, label='Electric Field (E)')
        
        # Set plot limits and labels
        ax.set_xlim([0, 180])
        ax.set_ylim([-60, 5])
        ax.set_xlabel('Angle (degrees)', fontsize=12)
        ax.set_ylabel('Magnitude (dB)', fontsize=12)
        ax.set_title('Far-Field Radiation Pattern (OYZ Plane) - Stacked Array', fontsize=14)
        ax.grid(True)
        ax.legend(loc='best')
        
        # Add vertical lines at notable angles
        for angle in [0, 90, 180]:
            ax.axvline(x=angle, color='gray', linestyle=':', alpha=0.7)
        
        # Add horizontal lines at notable dB levels
        for db_level in [-30, -20, -10, 0]:
            ax.axhline(y=db_level, color='gray', linestyle=':', alpha=0.7)
    
    plt.tight_layout()
    return fig