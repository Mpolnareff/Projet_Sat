import numpy as np

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

def calculate_far_field(floquet_modes, d_observation):
    """Calculate far-field pattern for each Floquet mode, excluding evanescent modes."""
    # Initialize far-field pattern
    theta_vals = np.linspace(0, np.pi, 180)  # Angle from 0 to 180 degrees
    E_far_field = np.zeros(len(theta_vals), dtype=complex)

    for mode_num, mode in floquet_modes.items():
        if 'k' in mode and not mode['is_evanescent']:  # Only include propagating modes
            k_vector = mode['k']
            polarization_vector = mode['polarization_vector']
            amplitude = mode['amplitude']

            # Ensure amplitude is a scalar
            scalar_amplitude = np.abs(amplitude)

            # Calculate far-field contribution
            for i, theta in enumerate(theta_vals):
                # Compute phase contributions (using real parts since it's propagating)
                phase = 1j * (np.real(k_vector[0]) * d_observation * np.sin(theta) +
                             np.real(k_vector[1]) * d_observation * np.sin(theta) * 0 +
                             np.real(k_vector[2]) * d_observation * np.cos(theta))
                
                # Calculate far-field contribution 
                E_far_field[i] += scalar_amplitude * np.exp(phase)

    return E_far_field
