import pandas as pd
import ast
import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from numba import cuda
import math
from time import time
import TestTab

# Declaration of constants
Lx = 1
Ly = 1
f = 300e6  # 300 MHz
pi = np.pi
c = 3e8    # Speed of light
theta_incident = pi/2
phi_incident = 0
k0 = 2*pi*f/c
ki = np.array([k0*np.sin(theta_incident)*np.cos(phi_incident),
              k0*np.sin(theta_incident)*np.sin(phi_incident),
              k0*np.cos(theta_incident)])

# Calculate wavelength for reference
wavelength = 2*pi/k0

# Cache for S matrix to avoid repeated file loading
smatrix_cache = {}

def magnitude_phase_to_complex(matrix):
    """Convert magnitude and phase to complex numbers."""
    # Extract magnitudes and phases into separate arrays
    magnitudes = np.array([[item[0] for item in row] for row in matrix])
    phases = np.array([[item[1] for item in row] for row in matrix])
    
    # Vectorized conversion to complex
    return 10**(magnitudes/20) * np.exp(1j * phases)

def get_s_matrix(file_name, freq_index=0):
    """Read S matrix from file with caching."""
    # Check cache first
    if file_name in smatrix_cache:
        return smatrix_cache[file_name]
    
    # Check file extension
    if file_name.lower().endswith('.tab'):
        # Parse tab file using TestTab
        try:
            frequencies, s_matrices = TestTab.parse_sparam_file(file_name)
            
            # Get the S-matrix for the specified frequency index
            # Default is the first frequency point
            s_matrix = s_matrices[freq_index]
            
            # Cache the result
            smatrix_cache[file_name] = s_matrix
            return s_matrix
        except Exception as e:
            print(f"Error reading tab file: {e}")
            return None
    elif file_name.lower().endswith(('.xlsx', '.xls')):
        # Keep the original Excel parsing for backward compatibility
        def str_to_tuple_float(s):
            try:
                t = ast.literal_eval(s)
                return tuple(float(x) for x in t)
            except (ValueError, SyntaxError):
                return None
        
        try:
            df = pd.read_excel(file_name)
            df = df.dropna()  # Remove rows with missing values
            df = df.drop_duplicates()  # Remove duplicate rows
            df = df.map(str_to_tuple_float)
            df = df.to_numpy()
            df = df[:, 1:]  # Remove first column
            
            result = magnitude_phase_to_complex(df)
            smatrix_cache[file_name] = result  # Cache the result
            return result
        except Exception as e:
            print(f"Error reading Excel file: {e}")
            return None
    else:
        print(f"Unsupported file format for {file_name}")
        return None

@cuda.jit
def calculate_field_grid_kernel(s_matrix_real, s_matrix_imag, x_vals, y_vals, z_val, 
                               magEincident, ki_0, ki_1, ki_2, k0, Lx, result_real, result_imag):
    """CUDA kernel for calculating field values across the grid with full evanescent wave handling."""
    # Get thread indices
    i, j = cuda.grid(2)
    
    # Check if indices are within bounds
    if i < result_real.shape[0] and j < result_real.shape[1]:
        # Extract coordinates
        x_val = x_vals[j]
        y_val = y_vals[i]
        
        # Initialize field value
        field_real = 0.0
        field_imag = 0.0
        
        # Loop through all modes
        for m in range(s_matrix_real.shape[0]):
            for n in range(s_matrix_real.shape[1]):
                # Calculate wave vector components with offsets for different modes
                kx = ki_0 + 2*math.pi*m/Lx
                ky = ki_1 + 2*math.pi*n/Lx
                
                # Check if wave is propagating or evanescent in z-direction
                under_sqrt = k0**2 - kx**2 - ky**2
                
                # Calculate propagation constants
                if under_sqrt >= 0:
                    # Propagating in z
                    kz = math.sqrt(under_sqrt)
                    
                    # Calculate propagating phase
                    phase = -(kx*x_val + ky*y_val + kz*z_val)
                    exp_real = math.cos(phase)
                    exp_imag = math.sin(phase)
                else:
                    # Evanescent in z
                    kz = math.sqrt(-under_sqrt)
                    
                    # Calculate xy phase component (propagating)
                    xy_phase = -(kx*x_val + ky*y_val)
                    
                    # Calculate exponential decay in z
                    z_decay = math.exp(-kz*z_val)
                    
                    # Combine for final complex exponential value
                    exp_real = math.cos(xy_phase) * z_decay
                    exp_imag = math.sin(xy_phase) * z_decay
                
                # Complex multiplication (a+bi)(c+di) = (ac-bd) + (ad+bc)i
                s_real = s_matrix_real[m, n]
                s_imag = s_matrix_imag[m, n]
                
                # Multiply phase factor by S-matrix element using complex multiplication
                term_real = exp_real * s_real - exp_imag * s_imag
                term_imag = exp_real * s_imag + exp_imag * s_real
                
                # Accumulate result
                field_real += term_real * magEincident
                field_imag += term_imag * magEincident
        
        # Store final result
        result_real[i, j] = field_real
        result_imag[i, j] = field_imag

def calculate_field_grid_cuda(s_matrix, x_range, y_range, z_val, magEincident, ki_vec, k0, Lx):
    """Calculate field grid using CUDA acceleration with proper evanescent wave handling."""
    # Convert to float32 for better GPU performance
    s_matrix_real = s_matrix.real.astype(np.float32)
    s_matrix_imag = s_matrix.imag.astype(np.float32)
    x_vals = x_range.astype(np.float32)
    y_vals = y_range.astype(np.float32)
    z_val = np.float32(z_val)
    magEincident = np.float32(magEincident)
    ki_0 = np.float32(ki_vec[0])
    ki_1 = np.float32(ki_vec[1])
    ki_2 = np.float32(ki_vec[2])
    k0 = np.float32(k0)
    Lx = np.float32(Lx)
    
    # Allocate memory for result
    result_real = np.zeros((len(y_vals), len(x_vals)), dtype=np.float32)
    result_imag = np.zeros((len(y_vals), len(x_vals)), dtype=np.float32)
    
    # Copy data to device
    d_s_matrix_real = cuda.to_device(s_matrix_real)
    d_s_matrix_imag = cuda.to_device(s_matrix_imag)
    d_x_vals = cuda.to_device(x_vals)
    d_y_vals = cuda.to_device(y_vals)
    d_result_real = cuda.to_device(result_real)
    d_result_imag = cuda.to_device(result_imag)
    
    # Configure grid and block dimensions
    threads_per_block = (16, 16)
    blocks_per_grid_x = math.ceil(result_real.shape[1] / threads_per_block[0])
    blocks_per_grid_y = math.ceil(result_real.shape[0] / threads_per_block[1])
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)
    
    # Launch kernel
    calculate_field_grid_kernel[blocks_per_grid, threads_per_block](
        d_s_matrix_real, d_s_matrix_imag, d_x_vals, d_y_vals, z_val,
        magEincident, ki_0, ki_1, ki_2, k0, Lx, d_result_real, d_result_imag
    )
    
    # Copy result back to host
    d_result_real.copy_to_host(result_real)
    d_result_imag.copy_to_host(result_imag)
    
    # Combine real and imaginary parts
    return result_real + 1j * result_imag

def calculate_field_grid_cupy(s_matrix, x_range, y_range, z_val, magEincident):
    """CuPy implementation with comprehensive evanescent wave handling."""
    # Move data to GPU
    d_s_matrix = cp.asarray(s_matrix)
    d_x_range = cp.asarray(x_range)
    d_y_range = cp.asarray(y_range)
    
    # Create coordinate grid
    X, Y = cp.meshgrid(d_x_range, d_y_range)
    
    # Initialize field array
    field_grid = cp.zeros((len(y_range), len(x_range)), dtype=cp.complex64)
    
    # Constants on device
    d_ki = cp.asarray(ki)
    d_k0 = cp.float32(k0)
    d_Lx = cp.float32(Lx)
    
    # Loop through all modes
    for m in range(s_matrix.shape[0]):
        for n in range(s_matrix.shape[1]):
            # Calculate wave vector components
            kx = d_ki[0] + 2*cp.pi*m/d_Lx
            ky = d_ki[1] + 2*cp.pi*n/d_Lx
            
            # Handle potential evanescent waves in z
            under_sqrt_z = d_k0**2 - kx**2 - ky**2
            
            if under_sqrt_z >= 0:
                # Propagating in z
                kz = cp.sqrt(under_sqrt_z)
                # Full propagating phase
                phase_delay = cp.exp(-1j*(kx*X + ky*Y + kz*z_val))
            else:
                # Evanescent in z
                kz = cp.sqrt(-under_sqrt_z)
                # Split into propagating (xy) and decaying (z) parts
                phase_delay = cp.exp(-1j*(kx*X + ky*Y)) * cp.exp(-kz*z_val)
            
            # Accumulate contribution from this mode
            field_grid += phase_delay * d_s_matrix[m, n] * magEincident
    
    # Return result as numpy array
    return cp.asnumpy(field_grid)

def calculate_field_grid_cpu(s_matrix, x_range, y_range, z_val, magEincident):
    """Comprehensive CPU implementation with full evanescent wave handling."""
    field_grid = np.zeros((len(y_range), len(x_range)), dtype=np.complex128)
    
    # Create coordinate grid
    X, Y = np.meshgrid(x_range, y_range)
    
    # Loop through all modes
    for m in range(s_matrix.shape[0]):
        for n in range(s_matrix.shape[1]):
            # Calculate wave vector components
            kx = ki[0] + 2*np.pi*m/Lx
            ky = ki[1] + 2*np.pi*n/Lx
            
            # Check for evanescent waves in z-direction
            under_sqrt_z = k0**2 - kx**2 - ky**2
            
            if under_sqrt_z >= 0:
                # Propagating in z
                kz = np.sqrt(under_sqrt_z)
                # Full propagating phase
                phase_delay = np.exp(-1j*(kx*X + ky*Y + kz*z_val))
            else:
                # Evanescent in z
                kz = np.sqrt(-under_sqrt_z)
                # Split into propagating (xy) and decaying (z) parts
                phase_delay = np.exp(-1j*(kx*X + ky*Y)) * np.exp(-kz*z_val)
            
            # Accumulate contribution from this mode
            field_grid += phase_delay * s_matrix[m, n] * magEincident
    
    return field_grid

def calculate_field_grid(file_name, magEincident, x_range, y_range, z_val=0, use_cuda=True, freq_index=0):
    """Calculate the electric field over a grid of points at fixed z with proper evanescent handling."""
    s_matrix = get_s_matrix(file_name, freq_index)
    if s_matrix is None:
        return None
    
    start_time = time()
    
    # Choose computational method based on preference and availability
    if use_cuda:
        try:
            # Try CuPy implementation first (simpler and handles complex numbers directly)
            field_grid = calculate_field_grid_cupy(s_matrix, x_range, y_range, z_val, magEincident)
            method = "CuPy"
        except Exception as e:
            print(f"CuPy failed with error: {e}. Falling back to CUDA kernel implementation.")
            try:
                # Fall back to Numba CUDA implementation
                field_grid = calculate_field_grid_cuda(s_matrix, x_range, y_range, z_val, 
                                                     magEincident, ki, k0, Lx)
                method = "Numba CUDA"
            except Exception as e2:
                print(f"CUDA implementation failed with error: {e2}. Falling back to CPU.")
                # Fall back to NumPy implementation
                field_grid = calculate_field_grid_cpu(s_matrix, x_range, y_range, z_val, magEincident)
                method = "CPU (fallback)"
    else:
        # Use CPU implementation
        field_grid = calculate_field_grid_cpu(s_matrix, x_range, y_range, z_val, magEincident)
        method = "CPU (selected)"
    
    end_time = time()
    print(f"Field calculation completed using {method} in {end_time - start_time:.3f} seconds")
    
    return field_grid

def plot_electric_field_3d(file_name, magEincident, resolution=150, z_val=0.1, use_cuda=True, freq_index=0):
    """Plot the electric field magnitude as a 3D surface."""
    # Create coordinate grid
    x_range = np.linspace(0, Lx, resolution)
    y_range = np.linspace(0, Ly, resolution)
    X, Y = np.meshgrid(x_range, y_range)
    
    print(f"Calculating field values for {resolution}×{resolution} grid...")
    # Calculate field values over the entire grid at once
    field_grid = calculate_field_grid(file_name, magEincident, x_range, y_range, z_val, use_cuda, freq_index)
    
    if field_grid is None:
        print("Error calculating field grid")
        return None
    
    # Calculate the magnitude for plotting
    field_magnitude = np.abs(field_grid)
    
    # Create the 3D plot
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the surface
    surf = ax.plot_surface(X, Y, field_magnitude, cmap=cm.plasma,
                          linewidth=0, antialiased=True, alpha=0.8)
    
    # Add color bar
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label='|E| Field Amplitude')
    
    # Add labels
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_zlabel('|E| Field Amplitude')
    ax.set_title(f'3D Electric Field Amplitude at z = {z_val:.2f} m (Resolution: {resolution}×{resolution})')
    print("Work completed")
    plt.savefig(f"Electric_Field_res{resolution}z{z_val:.2f}m.png", dpi=300)
    return fig

# Function to plot field variations along z-axis
def plot_field_vs_z(file_name, magEincident, x_point=0.5, y_point=0.5, z_range=None, 
                   num_points=100, use_cuda=True, freq_index=0):
    """Plot field magnitude variation along z-axis at a specific (x,y) point."""
    # Calculate wavelength for reference
    wavelength = 2*np.pi/k0
    
    if z_range is None:
        z_range = np.linspace(0, 5*wavelength, num_points)
    else:
        z_range = np.linspace(z_range[0], z_range[1], num_points)
    
    # Get S-matrix
    s_matrix = get_s_matrix(file_name, freq_index)
    if s_matrix is None:
        return None
    
    # Calculate field at each z value
    field_values = np.zeros(len(z_range), dtype=np.complex128)
    
    print(f"Calculating field values along z-axis at point ({x_point}, {y_point})...")
    
    # For efficiency, we'll use the CPU implementation directly
    # since we're only calculating for a single (x,y) point
    for i, z_val in enumerate(z_range):
        field_value = 0.0
        
        # Loop through all modes
        for m in range(s_matrix.shape[0]):
            for n in range(s_matrix.shape[1]):
                # Calculate wave vector components
                kx = ki[0] + 2*np.pi*m/Lx
                ky = ki[1] + 2*np.pi*n/Lx
                
                # Check for evanescent waves in z-direction
                under_sqrt_z = k0**2 - kx**2 - ky**2
                
                if under_sqrt_z >= 0:
                    # Propagating in z
                    kz = np.sqrt(under_sqrt_z)
                    # Full propagating phase
                    phase_delay = np.exp(-1j*(kx*x_point + ky*y_point + kz*z_val))
                else:
                    # Evanescent in z
                    kz = np.sqrt(-under_sqrt_z)
                    # Split into propagating (xy) and decaying (z) parts
                    phase_delay = np.exp(-1j*(kx*x_point + ky*y_point)) * np.exp(-kz*z_val)
                
                # Accumulate contribution from this mode
                field_value += phase_delay * s_matrix[m, n] * magEincident
        
        field_values[i] = field_value
    
    # Calculate magnitude
    field_magnitude = np.abs(field_values)
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(z_range, field_magnitude)
    ax.set_xlabel('z [m]')
    ax.set_ylabel('|E| Field Amplitude')
    ax.set_title(f'Electric Field Magnitude vs z at (x,y) = ({x_point}, {y_point})')
    ax.grid(True)
    
    # Add wavelength markers
    ax.axvline(x=wavelength, color='r', linestyle='--', alpha=0.5, label=f'λ = {wavelength:.2e} m')
    ax.axvline(x=2*wavelength, color='r', linestyle='--', alpha=0.3)
    ax.axvline(x=3*wavelength, color='r', linestyle='--', alpha=0.2)
    ax.legend()
    
    plt.savefig(f"Electric_Field_vs_z_at_x{x_point}_y{y_point}.png", dpi=300)
    print("Work completed")
    return fig
