import pandas as pd
import ast
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D
import scipy.linalg as la

def excel_to_python(file_name):
    def str_to_tuple_float(s):
        try:
            t = ast.literal_eval(s)
            return tuple(float(x) for x in t)
        except (ValueError, SyntaxError):
            return None  # or handle the error as needed

    df = pd.read_excel(file_name)
    df = df.dropna() # remove rows with missing values
    df = df.drop_duplicates() # remove duplicate rows
    df = df.map(str_to_tuple_float)
    df=df.to_numpy()
    df=df[:,1:]
    return(df)
def process_s_parameters(s_param_data):
    """
    Convert S-parameter data from (magnitude_dB, phase_degrees) tuples to complex numbers
    
    Parameters:
    s_param_data: numpy array of tuples (magnitude_dB, phase_degrees)
    
    Returns:
    s_matrix: numpy array of complex S-parameters
    """
    rows, cols = s_param_data.shape
    s_matrix = np.zeros((rows, cols), dtype=complex)
    
    for i in range(rows):
        for j in range(cols):
            mag_db, phase_deg = s_param_data[i, j]
            # Convert dB to linear magnitude
            magnitude = 10**(mag_db/20.0)
            # Convert degrees to radians
            phase_rad = np.deg2rad(phase_deg)
            # Convert to complex number
            s_matrix[i, j] = magnitude * np.exp(1j * phase_rad)
    
    return s_matrix

def compute_floquet_modes(s_matrix, frequency=None):
    """
    Compute Floquet modes from S-parameter matrix
    
    Parameters:
    s_matrix: Complex S-parameter matrix
    frequency: Frequency point (optional)
    
    Returns:
    eigenvalues: Floquet propagation constants
    eigenvectors: Floquet modes
    """
    # For a periodic structure, the Floquet modes are the eigenvectors
    # of the transfer matrix, which can be derived from the S-matrix
    
    # First, extract the block matrices
    n = s_matrix.shape[0] // 2  # Assuming square S-matrix with input/output ports
    print(np.shape(s_matrix))
    S11 = s_matrix[:n+1, :n+1]
    S12 = s_matrix[:n+1, n:]
    S21 = s_matrix[n:, :n+1]
    S22 = s_matrix[n:, n:]
    print(f"S11={S11}\n ,S12={S12}\n,S21={S21}\n,S22={S22}\n")
    # Calculate the transfer matrix T
    # T = [S21 - S22*S12^(-1)*S11,  S22*S12^(-1)]
    #     [-S12^(-1)*S11,          S12^(-1)    ]
    try:
        S12_inv = np.linalg.pinv(S12)
        
        T11 = S21 - np.dot(S22, np.dot(S12_inv, S11))
        T12 = np.dot(S22, S12_inv)
        T21 = -np.dot(S12_inv, S11)
        T22 = S12_inv
        
        T = np.block([[T11, T12], [T21, T22]])
        
        # Compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = la.eig(T)
        
        # The propagation constants are related to the eigenvalues
        propagation_constants = np.log(eigenvalues)
        
        # Sort by magnitude of propagation constant (attenuation)
        idx = np.argsort(np.abs(propagation_constants))
        propagation_constants = propagation_constants[idx]
        eigenvectors = eigenvectors[:, idx]
        
        return propagation_constants, eigenvectors
    
    except np.linalg.LinAlgError:
        print("Error: S12 matrix is singular and cannot be inverted.")
        return None, None

def analyze_floquet_modes(file_name, frequencies=None):
    """
    Analyze Floquet modes from S-parameter data loaded from Excel
    
    Parameters:
    file_name: Excel file containing S-parameter data
    frequencies: List of frequency points (optional)
    
    Returns:
    modes_data: Dictionary containing Floquet mode analysis results
    """
    # Load the S-parameter data
    s_param_data = excel_to_python(file_name)
    
    # Process S-parameters to complex form
    s_matrix = process_s_parameters(s_param_data)
    
    # Compute Floquet modes
    propagation_constants, modes = compute_floquet_modes(s_matrix)
    
    if propagation_constants is None:
        return None
    
    # Analyze results
    modes_data = {
        'propagation_constants': propagation_constants,
        'attenuation': np.abs(np.real(propagation_constants)),
        'phase_constants': np.imag(propagation_constants),
        'modes': modes
    }
    
    # Display results
    print("Floquet Mode Analysis Results:")
    print("-----------------------------")
    for i, prop_const in enumerate(propagation_constants):
        print(f"Mode {i+1}:")
        print(f"  Propagation constant: {prop_const}")
        print(f"  Attenuation: {np.abs(np.real(prop_const)):.4f} Np/unit")
        print(f"  Phase constant: {np.imag(prop_const):.4f} rad/unit")
        print(f"  Mode impedance: {compute_mode_impedance(modes[:,i], s_matrix):.2f} Ohms")
        print()
    
    return modes_data

def compute_mode_impedance(mode_vector, s_matrix):
    """
    Compute the impedance for a given Floquet mode
    
    Parameters:
    mode_vector: Eigenvector representing the mode
    s_matrix: S-parameter matrix
    
    Returns:
    impedance: Mode impedance in Ohms
    """
    # This is a simplified calculation - for actual implementation
    # you would need a more detailed approach based on your structure
    n = s_matrix.shape[0] // 2
    v_components = mode_vector[:n]
    i_components = mode_vector[n:]
    
    # Power calculation
    v_norm = np.linalg.norm(v_components)
    i_norm = np.linalg.norm(i_components)
    
    # Avoid division by zero
    if i_norm < 1e-10:
        return float('inf')
    
    return v_norm / i_norm * 50.0  # Assuming 50 Ohm reference impedance

def plot_floquet_modes(modes_data, frequencies=None):
    """
    Plot the Floquet mode analysis results
    
    Parameters:
    modes_data: Dictionary containing Floquet mode analysis results
    frequencies: List of frequency points (optional)
    """
    if modes_data is None:
        return
    
    # If no frequencies provided, use indices
    if frequencies is None:
        x_values = np.arange(len(modes_data['propagation_constants']))
        x_label = 'Mode Index'
    else:
        x_values = frequencies
        x_label = 'Frequency (GHz)'
    
    # Plot attenuation constants
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    plt.plot(x_values, modes_data['attenuation'], 'o-')
    plt.title('Floquet Mode Attenuation Constants')
    plt.xlabel(x_label)
    plt.ylabel('Attenuation (Np/unit)')
    plt.grid(True)
    
    # Plot phase constants
    plt.subplot(2, 1, 2)
    plt.plot(x_values, modes_data['phase_constants'], 'o-')
    plt.title('Floquet Mode Phase Constants')
    plt.xlabel(x_label)
    plt.ylabel('Phase (rad/unit)')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()



def calculate_floquet_wave(propagation_constants, eigenvectors, z_points, frequencies=None, amplitude_coefficients=None):
    """
    Calculate the field distribution of Floquet waves at different positions
    
    Parameters:
    propagation_constants: List of Floquet propagation constants
    eigenvectors: Matrix of Floquet mode eigenvectors
    z_points: Array of z-coordinate points to calculate field
    frequencies: Corresponding frequencies (optional)
    amplitude_coefficients: Mode excitation coefficients (optional)
    
    Returns:
    total_field: The calculated field at each z position
    """
    num_modes = len(propagation_constants)
    num_positions = len(z_points)
    
    # Default: equal amplitude for all modes if not specified
    if amplitude_coefficients is None:
        amplitude_coefficients = np.ones(num_modes)
    
    # Initialize field array
    total_field = np.zeros(num_positions, dtype=complex)
    
    # For each mode
    for i in range(num_modes):
        # Extract propagation constant
        gamma = propagation_constants[i]
        
        # Extract mode shape (use the first component for simplicity)
        # In practice, you might want a more sophisticated approach
        mode_shape = eigenvectors[0, i]
        
        # Calculate field contribution from this mode at each position
        for j, z in enumerate(z_points):
            # Apply Floquet theorem: F(z) = F(0) * exp(-gamma*z)
            field_contribution = amplitude_coefficients[i] * mode_shape * np.exp(-gamma * z)
            total_field[j] += field_contribution
    
    return total_field

def plot_floquet_wave(z_points, total_field, mode_info=None, title="Floquet Wave Propagation"):
    """
    Plot the Floquet wave field distribution
    
    Parameters:
    z_points: Array of z positions
    total_field: Complex field values at each position
    mode_info: Dictionary with mode information (optional)
    title: Plot title
    """
    # Extract magnitude and phase
    magnitude = np.abs(total_field)
    phase = np.angle(total_field)
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot magnitude
    ax1.plot(z_points, magnitude, 'b-', linewidth=2)
    ax1.set_title(f"{title} - Magnitude")
    ax1.set_xlabel("Position (z)")
    ax1.set_ylabel("Field Magnitude")
    ax1.grid(True)
    
    # Plot phase
    ax2.plot(z_points, phase, 'r-', linewidth=2)
    ax2.set_title(f"{title} - Phase")
    ax2.set_xlabel("Position (z)")
    ax2.set_ylabel("Phase (radians)")
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Also plot real and imaginary parts
    plt.figure(figsize=(12, 6))
    plt.plot(z_points, np.real(total_field), 'g-', label='Real Part')
    plt.plot(z_points, np.imag(total_field), 'm--', label='Imaginary Part')
    plt.title(f"{title} - Real and Imaginary Components")
    plt.xlabel("Position (z)")
    plt.ylabel("Field Value")
    plt.legend()
    plt.grid(True)
    plt.show()

def create_floquet_wave_animation(z_points, propagation_constants, eigenvectors, 
                                 time_points, amplitude_coefficients=None):
    """
    Create an animation of Floquet wave propagation over time
    
    Parameters:
    z_points: Array of z-coordinate positions
    propagation_constants: List of Floquet propagation constants
    eigenvectors: Matrix of Floquet mode eigenvectors
    time_points: Array of time points for animation
    amplitude_coefficients: Mode excitation coefficients (optional)
    
    Returns:
    Animation object
    """
    # Calculate spatial field distribution (without time factor)
    spatial_field = calculate_floquet_wave(propagation_constants, eigenvectors, 
                                          z_points, amplitude_coefficients)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    line, = ax.plot([], [], 'b-', linewidth=2)
    
    # Set axis limits
    ax.set_xlim(min(z_points), max(z_points))
    field_max = 1.5 * np.max(np.abs(spatial_field))
    ax.set_ylim(-field_max, field_max)
    
    ax.set_title("Floquet Wave Propagation")
    ax.set_xlabel("Position (z)")
    ax.set_ylabel("Field Value")
    ax.grid(True)
    
    # Animation function
    def animate(i):
        time = time_points[i]
        # Add time factor to the spatial distribution
        # For simplicity, assuming angular frequency omega=1
        omega = 1.0
        total_field_time = spatial_field * np.exp(1j * omega * time)
        # Plot real part
        line.set_data(z_points, np.real(total_field_time))
        return (line,)
    
    # Create animation
    anim = animation.FuncAnimation(fig, animate, frames=len(time_points),
                                  interval=50, blit=True)
    
    return anim

def plot_3d_floquet_field(x_points, y_points, z_points, propagation_constants, 
                         eigenvectors, amplitude_coefficients=None, time=0):
    """
    Plot 3D field distribution of Floquet modes
    
    Parameters:
    x_points, y_points: Meshgrid of x,y coordinates
    z_points: Array of z positions
    propagation_constants: Floquet propagation constants
    eigenvectors: Floquet mode eigenvectors
    amplitude_coefficients: Mode weights (optional)
    time: Time point for the plot (optional)
    """
    # Calculate field along z
    z_field = calculate_floquet_wave(propagation_constants, eigenvectors, 
                                    z_points, amplitude_coefficients)
    
    # Add time factor
    omega = 1.0  # Angular frequency (simplified)
    z_field_time = z_field * np.exp(1j * omega * time)
    
    # Create 3D field by assuming a simplified transverse distribution
    # (This would be replaced by actual mode shapes in a real implementation)
    X, Y = np.meshgrid(x_points, y_points)
    field_3d = np.zeros((len(y_points), len(x_points), len(z_points)), dtype=complex)
    
    # Create a simplified transverse distribution (Gaussian)
    x_center, y_center = np.mean(x_points), np.mean(y_points)
    sigma = min(np.ptp(x_points), np.ptp(y_points)) / 4
    
    for i, z_val in enumerate(z_points):
        # Gaussian transverse profile multiplied by z-propagation
        transverse = np.exp(-((X-x_center)**2 + (Y-y_center)**2) / (2*sigma**2))
        field_3d[:,:,i] = transverse * z_field_time[i]
    
    # Create 3D plot
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # For visualization, we'll plot real part using a color map on several z-slices
    num_slices = min(10, len(z_points))
    z_indices = np.linspace(0, len(z_points)-1, num_slices, dtype=int)
    
    for idx in z_indices:
        z_val = z_points[idx]
        
        # Get field slice and normalize for better visualization
        field_slice = np.real(field_3d[:,:,idx])
        
        # Plot as a surface at this z position
        surf = ax.plot_surface(X, Y, np.ones_like(X) * z_val, 
                              rstride=1, cstride=1,
                              facecolors=plt.cm.viridis(plt.Normalize(-1, 1)(field_slice)),
                              alpha=0.7)
    
    # Add colorbar
    m = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=plt.Normalize(-1, 1))
    m.set_array([])
    plt.colorbar(m, ax=ax, label='Field Value (Real Part)')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z (Propagation Direction)')
    ax.set_title('3D Floquet Field Distribution')
    
    plt.tight_layout()
    plt.show()

# Example usage
def demonstrate_floquet_waves(propagation_constants, eigenvectors):
    """
    Demonstrate different visualizations of Floquet waves
    
    Parameters:
    propagation_constants: Array of complex propagation constants
    eigenvectors: Matrix of eigenvectors (Floquet modes)
    """
    # Setup spatial domain
    z_points = np.linspace(0, 10, 500)  # 10 unit cells
    
    # 1. Basic field calculation and plotting
    print("Calculating total field from all modes...")
    total_field = calculate_floquet_wave(propagation_constants, eigenvectors, z_points)
    plot_floquet_wave(z_points, total_field, title="Total Floquet Wave (All Modes)")
    
    # 2. Focus on dominant mode
    # Find the mode with smallest attenuation
    idx_dominant = np.argmin(np.abs(np.real(propagation_constants)))
    dom_prop_const = [propagation_constants[idx_dominant]]
    dom_eigenvector = eigenvectors[:, [idx_dominant]]
    
    print(f"Calculating field for dominant mode (mode {idx_dominant})...")
    dominant_field = calculate_floquet_wave(dom_prop_const, dom_eigenvector, z_points)
    plot_floquet_wave(z_points, dominant_field, title=f"Dominant Floquet Mode (Mode {idx_dominant})")
    
    # 3. Create animation of wave propagation (only for real demonstrations)
    print("Creating animation of wave propagation...")
    time_points = np.linspace(0, 2*np.pi, 60)  # One complete cycle
    anim = create_floquet_wave_animation(z_points, propagation_constants, eigenvectors, time_points)
    
    # To save animation:
    # anim.save('floquet_wave.mp4', writer='ffmpeg', fps=15)
    
    # 4. 3D field visualization
    print("Creating 3D field visualization...")
    x_points = np.linspace(-2, 2, 50)
    y_points = np.linspace(-2, 2, 50)
    plot_3d_floquet_field(x_points, y_points, z_points[::10], 
                         propagation_constants, eigenvectors)
    
    return anim

# Connect to the previous Floquet mode computation
def analyze_and_visualize_floquet_waves(file_name):
    """
    Complete workflow: Analyze S-parameters and visualize the resulting waves
    
    Parameters:
    file_name: Excel file with S-parameter data
    """
    # Step 1: Load and process S-parameters
    s_param_data = excel_to_python(file_name)
    s_matrix = process_s_parameters(s_param_data)
    
    # Step 2: Compute Floquet modes
    propagation_constants, modes = compute_floquet_modes(s_matrix)
    
    if propagation_constants is None:
        print("Failed to compute Floquet modes.")
        return
    
    # Step 3: Visualize the resulting waves
    anim = demonstrate_floquet_waves(propagation_constants, modes)
    
    plt.show()  # Show all plots
    
    return propagation_constants, modes, anim


    # Assuming you have an Excel file with S-parameters
file_name = "C:\\Users\\maell\\Desktop\\ENAC\\Projet SAT\\Filtered_TE_SMatrix.xlsx"
results = analyze_and_visualize_floquet_waves(file_name)
