import numpy as np

def intersect_line_plane(start_line, direction_line, center_plane, normal_plane):
  # extract tolerance if needed
  tol = 1e-14
  
  # get difference of center of the plane and strating point of the line
  diff = center_plane - start_line

  # get dot product of line direction with plane normal for the denominator
  denominator = np.dot(normal_plane, direction_line)

  # get relative position of intersection point on line
  relative_poi = np.dot(normal_plane, diff) / denominator
  relative_poi = np.ones((1,3)) * relative_poi

  # compute coordinates of interdection point
  point = start_line + relative_poi*direction_line
  return point




if __name__ == "__main__":
  start_line = np.array([46.1159, 37.1938, 997.2482])
  direction_line = np.array([0.0809, 0.2322, 0.9693])
  center_plane = np.array([0, 0, 473.2069])
  normal_plane = np.array([0.0809, 0.2322, 0.9693])
  point = intersect_line_plane(start_line, direction_line, center_plane, normal_plane) # 4.0248  -83.6270  492.9047
  print(point)