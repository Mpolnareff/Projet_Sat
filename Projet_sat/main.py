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
    s_matrix = parse_sparam_file(filename,f)
    matrix_shape = np.shape(s_matrix)
   
    # Make sure we have enough Floquet modes defined
    while len(floquet_modes)+1 < matrix_shape[0]:
        user_input = input(f"Please enter the polarisation and indices of the {len(floquet_modes)+1}th mode, with format TE or TM,m,n: ")
        polar, m, n = user_input.split(',')
        floquet_modes[len(floquet_modes) + 1] = {'polarization': polar.strip(), 'm': int(m), 'n': int(n)}
   
    # Calculate source field using Floquet modes and S-matrix
    resolution = 100  # Resolution for the field calculationss
    Esource = calculate_Esource(floquet_modes, resolution, s_matrix)
   
    # Print information about each mode
    print_mode_info(floquet_modes)
   
    # Create the mask
    mask = np.ones((resolution,resolution))
   
    # Calculate currents on the mask
    currents = calculate_currents_on_mask(Esource, mask)
   
  # Generate the far-field radiation patterns
    polar_fig_unitary = plot_far_field_radiated_OYZ(Esource,mask, resolution,1800)
    cartesian_fig_unitary = plot_far_field_cartesian(Esource,mask, resolution,1800)
    
    # Show the figures
    plt.figure(polar_fig_unitary.number)
    plt.tight_layout()
    plt.savefig('polar_fig_unitary.png')
    plt.figure(cartesian_fig_unitary.number)
    plt.tight_layout()
    plt.savefig('cartesian_fig_unitary.png')
    plt.show()


    # Number of blocks for stacked array
    num_blocks = 15  
    element_spacing = Lz  
    
    # Generate the stacked array radiation patterns
    total_E_fields, observation_points, theta_degrees = calculate_stacked_array_field(
        Esource, mask, num_blocks, resolution, 1800, element_spacing
    )
    
    # Create the polar plot
    fig_polar = plot_stacked_array_field(
        total_E_fields, observation_points, theta_degrees, plot_type='polar'
    )
    
    # Create the cartesian plot
    fig_cartesian = plot_stacked_array_field(
        total_E_fields, observation_points, theta_degrees, plot_type='cartesian'
    )
    
    # Show the plots
    plt.show()
    
    # Save the figures with improved settings
    plt.figure(fig_polar)
    plt.savefig('stacked_array_polar.png')
    
    plt.figure(fig_cartesian)
    plt.savefig('stacked_array_cartesian.png')
    return polar_fig_unitary,cartesian_fig_unitary,fig_polar,fig_cartesian


if __name__ == "__main__":
    filename = input("Enter the filename of the S-parameter file: ")
    main(filename)