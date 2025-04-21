import numpy as np

def parse_sparam_file(filename):
    # Read the file
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    # Find the header row that begins with "Frequency"
    for i, line in enumerate(lines):
        if line.startswith('Frequency'):
            header_line = i
            break
    
    # Extract data lines (all frequencies)
    data_lines = [line.strip().split() for line in lines[header_line + 1:] if line.strip()]
    
    # Determine the number of ports from the header
    # Count how many "S[x,y]_Mag" patterns we have in the header
    header_parts = lines[header_line].strip().split()
    s_params = [part for part in header_parts if part.startswith('S[') and part.endswith('_Mag')]
    
    # Extract unique port numbers
    port_indices = set()
    for s_param in s_params:
        # Extract i,j from S[i,j]_Mag
        i_j = s_param[2:-5].split(',')
        port_indices.add(int(i_j[0]))
        port_indices.add(int(i_j[1]))
    
    num_ports = max(port_indices)
    
    # Extract frequencies
    frequencies = [float(line[0]) for line in data_lines]
    num_freq_points = len(frequencies)
    
    # Create a 3D array: (frequency points, num_ports, num_ports)
    s_matrix = np.zeros((num_freq_points, num_ports, num_ports), dtype=complex)
    
    # Parse data for each frequency point
    for freq_idx, data_line in enumerate(data_lines):
        for i in range(1, num_ports + 1):
            for j in range(1, num_ports + 1):
                # Find the column indices for S[i,j]_Mag and S[i,j]_Phs
                mag_col = None
                phase_col = None
                
                for col_idx, header in enumerate(header_parts):
                    if header == f'S[{i},{j}]_Mag':
                        mag_col = col_idx
                    if header == f'S[{i},{j}]_Phs':
                        phase_col = col_idx
                
                if mag_col is not None and phase_col is not None:
                    # Extract magnitude and phase
                    mag = float(data_line[mag_col])
                    phase_deg = float(data_line[phase_col])
                    
                    # Convert phase from degrees to radians and create complex number
                    phase_rad = np.deg2rad(phase_deg)
                    s_matrix[freq_idx, i-1, j-1] = mag * np.exp(1j * phase_rad)
    
    return np.array(frequencies), s_matrix

# Usage
frequencies, s_matrix = parse_sparam_file("C:\\Users\\maell\\Desktop\\ENAC\\Projet SAT\\Sources\\Sparam200MHz.tab")

if s_matrix is not None:
    print("Frequencies:")
    print(frequencies)
    print("S-parameters matrix:")
    print(s_matrix)
else:
    print("Failed to parse S-parameter file.")