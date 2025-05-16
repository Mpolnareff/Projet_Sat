import numpy as np
import matplotlib.pyplot as plt

def create_N_mask(Npoint):
    thickness = max(1, Npoint // 10)  # Ensure thickness is at least 1
    mask = np.zeros((Npoint, Npoint))
    
    # Left vertical line
    mask[:, 0:thickness] = 1
    
    # Right vertical line
    mask[:, Npoint-thickness:Npoint] = 1
    
    # Diagonal line - more precise implementation
    for i in range(Npoint):
        # Calculate start and end points of the diagonal
        start_x = thickness
        end_x = Npoint - thickness
        
        # Calculate the diagonal position with precise floating-point math
        # This gives the exact position where the diagonal should be for this row
        exact_pos = start_x + (i / (Npoint - 1)) * (end_x - start_x)
        
        # Apply the thickness centered around the exact position
        half_t = thickness // 2
        for t in range(-half_t, thickness - half_t):
            pos = int(exact_pos + t)
            if 0 <= pos < Npoint:  # Ensure we're within bounds
                mask[i, pos] = 1
    
    return mask

# Create and display a 20x20 N mask
Npoint = 500
mask = create_N_mask(Npoint)

# Visualize the mask
plt.figure(figsize=(6, 6))
plt.imshow(mask, cmap='binary')
plt.title(f"N-shaped Mask ({Npoint}x{Npoint})")
plt.colorbar()
plt.tight_layout()
plt.show()

# Print smaller example for clarity
print("Example 10x10 N mask:")
print(create_N_mask(10))