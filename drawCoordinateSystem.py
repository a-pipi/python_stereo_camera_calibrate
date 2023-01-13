import numpy as np

def draw_coordinate_system(point, name, ax, color='purple', R=np.eye(3)):
  directions = np.eye(3)
  d = np.dot(R, directions)
  ax.quiver(point[0], point[2], point[1], d[0][0], d[0][2], d[0][1], color='r', arrow_length_ratio=0.15, length=50)
  ax.quiver(point[0], point[2], point[1], d[1][0], d[1][2], d[1][1], color='g', arrow_length_ratio=0.15, length=50)
  ax.quiver(point[0], point[2], point[1], d[2][0], d[2][2], d[2][1], color='b', arrow_length_ratio=0.15, length=50)
  ax.scatter(point[0], point[2], point[1], color=color, marker='*', label=name)