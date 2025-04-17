from Fonctions import *


# Replace with your actual file path
file_name = "C:\\Users\\maell\\Desktop\\ENAC\\Projet SAT\\Filtered_TE_SMatrix.xlsx"
    
# Calculate the symbolic expression for the field
Er_symbolic = calculate_E(file_name, 1.0)
    
if Er_symbolic is not None:
    print("Symbolic expression for the field calculated successfully.")
        
        # Example: Evaluate field at a specific point
    field_value = evaluate_field_at_point(Er_symbolic, 0.5, 0.5, 0.0)
    print(f"Field value at (0.5, 0.5, 0.0): {field_value}")
        
        # Example: Plot the electric field in 3D
    fig = plot_electric_field_3d(file_name, 1.0, resolution=15)
    plt.show()
