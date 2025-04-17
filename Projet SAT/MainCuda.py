from Fonctions_with_cuda import *

# Replace with your actual file path
file_name = "C:\\Users\\maell\\Desktop\\ENAC\\Projet SAT\\Filtered_TE_SMatrix.xlsx"

# Plot the electric field in 3D with high resolution
z_values=np.linspace(0,1,10)
for z in z_values:
    fig = plot_electric_field_3d(file_name, 1.0, resolution=1500,z_val=z, use_cuda=True)
    #plt.show()
