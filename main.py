import numpy as np
import matplotlib.pyplot as plt
import os
# Import all functions from the paste.txt file
# Assuming it's saved as Reconstitution_Floquet.py
from Reconstitution_Floquet import *

# Define Floquet modes with polarization information
floquet_modes = {}

def main(filename):
    """
    Main function to analyze Floquet modes and plot radiation patterns.
   
    Args:
        filename (str): Path to the S-parameter file
    """
    # Parse the S-parameter file
    s_matrix = parse_sparam_file(filename)
    matrix_shape = np.shape(s_matrix)
   
    # Make sure we have enough Floquet modes defined
    while len(floquet_modes) < matrix_shape[0]:
        user_input = input(f"Please enter the polarisation and indices of the {len(floquet_modes)+1}th mode, with format TE or TM,m,n: ")
        polar, m, n = user_input.split(',')
        floquet_modes[len(floquet_modes) + 1] = {'polarization': polar.strip(), 'm': int(m), 'n': int(n)}
   
    # Calculate source field using Floquet modes and S-matrix
    resolution = 100  # Resolution for the field calculations
    Esource = calculate_Esource(floquet_modes, resolution, s_matrix)
   
    # Print information about each mode
    print_mode_info(floquet_modes)
   
    # Create the mask
    mask = create_N_mask(resolution)
   
    # Calculate currents on the mask
    currents = calculate_currents_on_mask(Esource, mask)
   
  # Generate the far-field radiation patterns
    polar_fig = plot_far_field_radiated_OXZ(Esource, resolution,1800)
    cartesian_fig = plot_far_field_cartesian(Esource, resolution,1800)
    
    # Show the figures
    plt.figure(polar_fig.number)
    plt.tight_layout()
    plt.savefig('polar_fig.png')
    plt.figure(cartesian_fig.number)
    plt.tight_layout()
    plt.savefig('cartesian_fig.png')
    plt.show()
    return polar_fig,cartesian_fig

if __name__ == "__main__":
    filename = input("Enter the filename of the S-parameter file: ")
    main(filename)