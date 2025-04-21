import Fonctions_tab_files 
from Fonctions_tab_files import *
z_values = np.linspace(0, 1, 10)
file_name="C:\\Users\\maell\\Desktop\\ENAC\\Projet SAT\\Sources\\Sparam200MHz.tab"
# Use a more reasonable resolution for multiple plots
resolution = 1000  # Reduced for faster processing while testing

# Select which frequency index to use from the tab file (0 = first frequency)
freq_index = 0

print("Processing multiple z-planes...")
for z in z_values:
    print(f"\nGenerating plot for z = {z:.2f}")
    fig = plot_electric_field_3d(
        file_name, 
        magEincident=1.0, 
        resolution=resolution, 
        z_val=z, 
        use_cuda=True, 
        freq_index=freq_index
    )
    plt.close(fig)  # Close the figure to free memory

# Create a z-variation plot at a specific point
# This will show how the field varies with z at one (x,y) location
print("\nGenerating z-variation plot...")
x_point, y_point = 0.5, 0.5  # Center of the domain
z_plot = plot_field_vs_z(
    file_name, 
    magEincident=1.0, 
    x_point=x_point, 
    y_point=y_point, 
    z_range=[0, 1], 
    num_points=100,
    freq_index=freq_index
)
plt.show()
