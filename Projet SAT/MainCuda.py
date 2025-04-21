from Fonctions_with_cuda import *
# Replace with your actual file path
file_name = "C:\\Users\\maell\\Desktop\\ENAC\\Projet SAT\\Sources\\Sparam200MHz.tab"
# Plot the electric field at different z-values
z_values = np.linspace(0, 1, 10)

# Use a more reasonable resolution for multiple plots
resolution = 1000  # Still detailed but much faster

for z in z_values:
    print(f"\nGenerating plot for z = {z:.2f}")
    fig = plot_electric_field_3d(file_name, 1.0, resolution=resolution, z_val=z, use_cuda=True)
    plt.close(fig)  # Close the figure to free memory

# Optionally, create a z-variation plot at a specific point
# This will show how the field varies with z at one (x,y) location
x_point, y_point = 0.5, 0.5  # Center of the domain
z_plot = plot_field_vs_z(file_name, 1.0, x_point, y_point, z_range=[0, 1], num_points=100)
plt.show()
