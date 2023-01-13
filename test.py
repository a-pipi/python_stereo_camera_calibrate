import numpy as np

# Define the point to be rotated
point = np.array([[1], [2], [3]])

# Define the rotation matrix
R = np.array([[0.5, 0.5, -0.5],
              [-0.5, 0.5, 0.5],
              [0.5, 0.5, 0.5]])

# Multiply the rotation matrix by the point to get the new coordinates
new_coordinates = np.matmul(R, point)

# Print the new coordinates
print(new_coordinates)
