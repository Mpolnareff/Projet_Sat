from Fonctions import *


# Replace with your actual file path
file_name = "C:\\Users\\maell\\Desktop\\ENAC\\Projet SAT\\Filtered_TE_SMatrix.xlsx"

# Example: Evaluate field at a specific point
field_value = calculate_field_at_point(file_name, 1.0, 0.5, 0.5, 0.0)
print(f"Field value at (0.5, 0.5, 0.0): {field_value}")

# Example: Plot the electric field in 3D
# Note: Higher resolutions will take longer to compute with either method
fig = plot_electric_field_3d(file_name, 1.0, resolution=100)
plt.show()