from Reconstitution_Floquet import *
import matplotlib.pyplot as plt

# Floquet modes with polarization information - properly defined
floquet_modes = {
    1: {'polarization': 'TE', 'm': 0, 'n': 0},
    2: {'polarization': 'TM', 'm': 0, 'n': 0},
    3: {'polarization': 'TE', 'm': 0, 'n': -1},
    4: {'polarization': 'TE', 'm': 0, 'n': -2},
    5: {'polarization': 'TE', 'm': -2, 'n': -1},
    6: {'polarization': 'TE', 'm': 2, 'n': -1},
    7: {'polarization': 'TE', 'm': 0, 'n': 2},
    8: {'polarization': 'TE', 'm': 1, 'n': -1},
    9: {'polarization': 'TM', 'm': -1, 'n': -1},
    10: {'polarization': 'TM', 'm': 0, 'n': -1},
    11: {'polarization': 'TM', 'm': -2, 'n': -1},
    12: {'polarization': 'TM', 'm': 2, 'n': -1},
    13: {'polarization': 'TM', 'm': 1, 'n': -1},

}

def main(filename):
    with open(filename, 'r') as file:
        # Parse the S-parameter file
        s_matrix = parse_sparam_file(filename)
        a=np.shape(s_matrix)
        while len(floquet_modes)<a[0]:
            polar,m,n=input(f"Please enter the polarisation and indices of the {len(floquet_modes)+1}th mode, with format TE or TM,m,n :").split(',')
            floquet_modes[len(floquet_modes) + 1] = {'polarization':polar,'m':int(m),'n':int(n)}

    # Calculate wave vectors and polarization vectors for each mode
        for mode_num, mode in floquet_modes.items():
            k_vector, is_evanescent = calculate_wave_vector(mode['m'], mode['n'], k0, ki)
            polarization_vector = calculate_polarization_vectors(k_vector, mode['polarization'])

        # Add wave vector, polarization vector, and evanescent flag to the mode
            floquet_modes[mode_num]['k'] = k_vector
            floquet_modes[mode_num]['polarization_vector'] = polarization_vector
            floquet_modes[mode_num]['is_evanescent'] = is_evanescent

        # Use S-matrix to determine the amplitude for each mode
        # Example: Use the S-parameters to set the amplitude (simplified)
        # Here, we assume the S-matrix indices correspond to the mode indices
            if mode_num - 1 < s_matrix.shape[0] and mode_num - 1 < s_matrix.shape[1]:
                amplitude = s_matrix[mode_num - 1, mode_num - 1]  # Example: Use diagonal elements
            else:
                amplitude = 0.1 * np.exp(1j * np.pi/4)  # Default amplitude if index out of range

            floquet_modes[mode_num]['amplitude'] = amplitude

    # Print mode information
        print_mode_info(floquet_modes)

    # Calculate far-field pattern
        E_far_field = calculate_far_field(floquet_modes, d_observation)

    # Calculate far-field pattern for [0, 180] degrees
        theta_vals = np.linspace(0, np.pi, 180)
        E_far_field = calculate_far_field(floquet_modes, d_observation)
        
    # Create full 360 degree pattern by mirroring
        full_theta = np.linspace(0, 2*np.pi, 360)
        full_E = np.concatenate([E_far_field, np.flip(E_far_field)])
        
    # Create polar plot
        plt.figure(figsize=(10, 10))
        ax = plt.subplot(111, projection='polar')
        ax.plot(full_theta, np.abs(full_E), linewidth=2)
        ax.set_title('Far-Field Radiation Pattern', pad=20)
        ax.grid(True)
        
    # Also create the original Cartesian plot for comparison
        plt.figure(figsize=(10, 6))
        plt.plot(np.rad2deg(full_theta), np.abs(full_E))
        plt.title('Far-Field Pattern (0-360 degrees)')
        plt.xlabel('Theta (degrees)')
        plt.ylabel('|E|')
        plt.grid(True)
        
        plt.show()

main(input("Enter the filename of the S-parameter file: ")) 
