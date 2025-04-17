import pandas as pd
import ast
import numpy as np
import sympy as sm
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

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
eps = 1e-6

# Symbolic variables for analytical expressions
x, y, z = sm.symbols('x y z')

#Fonction
def magnitude_phase_to_complex(matrix):

    # Convert input to numpy array if it's not already
    matrix = np.asarray(matrix)
    
    # Check if input is a matrix with tuples
    if matrix.ndim != 2:
        raise ValueError("Input must be a 2D matrix")
    
    # Extract magnitudes and phases into separate matrices
    magnitudes = np.array([[item[0] for item in row] for row in matrix])
    phases = np.array([[item[1] for item in row] for row in matrix])
    
    # Vectorized conversion to complex
    complex_matrix = 10**(magnitudes/20) * np.exp(1j * phases)
    
    return complex_matrix

def excel_to_python(file_name):
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
        print(df)
        return magnitude_phase_to_complex(df)
    except Exception as e:
        print(f"Error reading Excel file: {e}")
        return None

def calculate_kr(m, n):
    kx = ki[0] + 2*pi*m/Lx
    ky = ki[1] + 2*pi*n/Lx
    
    # Handle potential evanescent waves
    under_sqrt = k0**2 - kx**2 - ky**2
    if under_sqrt >= 0:
        kz = sm.sqrt(under_sqrt)
    else:
        kz = 1j * sm.sqrt(-under_sqrt)
    
    Phase_delay_mn = sm.exp(-1j*(kx*x + ky*y + kz*z))
    return Phase_delay_mn

def calculate_E(file_name, magEincident):
    
    s_matrix = excel_to_python(file_name)
    if s_matrix is None:
        return None
    
    Er = 0
    a, b = s_matrix.shape
    for i in range(a):
        for j in range(b):
            Er = Er + calculate_kr(i, j) * s_matrix[i, j] * magEincident
    return Er

def evaluate_field_at_point(Er_expr, x_val, y_val, z_val):
   
    return complex(Er_expr.subs({x: x_val, y: y_val, z: z_val}))

def calculate_field_grid(file_name, magEincident, x_range, y_range, z_val=0):
   
    Er_expr = calculate_E(file_name, magEincident)
    if Er_expr is None:
        return None
    
    field_grid = np.zeros((len(y_range), len(x_range)), dtype=complex)
    
    for i, y_val in enumerate(y_range):
        for j, x_val in enumerate(x_range):
            field_grid[i, j] = evaluate_field_at_point(Er_expr, x_val, y_val, z_val)
    
    return field_grid

def plot_electric_field_3d(file_name, magEincident, resolution=20):
    Er_expr = calculate_E(file_name, magEincident)
    if Er_expr is None:
        print("Error: Could not calculate electric field expression")
        return None
    
    # Create coordinate grid
    x_range = np.linspace(0, Lx, resolution)
    y_range = np.linspace(0, Ly, resolution)
    X, Y = np.meshgrid(x_range, y_range)
    
    # For a 3D plot, we'll use a fixed height z or multiple z-planes
    # Option 1: Single z-plane
    z_val = 0.1  # Some distance above the surface
    
    # Calculate field magnitude on grid
    field_grid = np.zeros((resolution, resolution), dtype=float)
    
    print("Calculating field values...")
    for i in range(resolution):
        for j in range(resolution):
            field_value = evaluate_field_at_point(Er_expr, X[i, j], Y[i, j], z_val)
            field_grid[i, j] = np.abs(field_value)
    
    # Create the 3D plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the surface
    surf = ax.plot_surface(X, Y, field_grid, cmap=cm.plasma,
                          linewidth=0, antialiased=True, alpha=0.8)
    
    # Add color bar
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label='|E| Field Amplitude')
    
    # Add labels
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_zlabel('|E| Field Amplitude')
    ax.set_title('3D Electric Field Amplitude at z = {:.2f} m'.format(z_val))
    
    return fig

