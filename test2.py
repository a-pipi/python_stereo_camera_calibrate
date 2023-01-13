import numpy as np
import matplotlib.pyplot as plt

# Set of 2D points
points1 = np.array([[1, 2], [3, 4], [5, 6]])
points2 = np.array([[2, 3], [4, 5], [6, 7]])

# Calculate the centroid of each set of points
centroid1 = np.mean(points1, axis=0)
centroid2 = np.mean(points2, axis=0)

# Shift the points so that the centroid is at the origin
points1_centered = points1 - centroid1
points2_centered = points2 - centroid2

# Calculate the rotation matrix using the Singular Value Decomposition (SVD) method
U, _, VT = np.linalg.svd(points1_centered.T @ points2_centered)

# Calculate the rotation matrix
rotation_matrix = U @ VT

# Apply the rotation matrix to the first set of points
rotated_points = points1_centered @ rotation_matrix + centroid2

plt.scatter(points1_centered[:,0], points1_centered[:,1], 20, color='r')
plt.scatter(points2_centered[:,0], points2_centered[:,1], 10, color='b')
plt.show()

np.meshgrid()