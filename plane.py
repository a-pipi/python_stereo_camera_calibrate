import numpy as np

def plane(points, ax, label, color, _flag_plot=True, _virtual_screen=False):
  points = points.T
  n = len(points)
  points = np.c_[points, np.ones(n)]
  
  if _flag_plot:
    # plot all screenpoints
    ax.scatter(points[0,:], points[2,:], points[1,:], color=color, label=label)

  U, S, V = np.linalg.svd(points)

  i = np.where(S==min(S))

  coeff = V[i][0]
  coeff = coeff/np.linalg.norm(coeff[0:3])

  normal_plane = coeff[0:3]

  point_plane = np.array([0,0,0])
  point_plane[2] = -coeff[3]/coeff[2]

  
  return point_plane, normal_plane

def norm_plane(points):
  # points = np.concatenate(points, axis=1)
  points = points.T
  n = len(points)
  points = np.c_[points, np.ones(n)]

  U, S, V = np.linalg.svd(points)

  i = np.where(S==min(S))

  coeff = V[i][0]
  coeff = coeff/np.linalg.norm(coeff[0:3])

  normal_plane = coeff[0:3]

  return normal_plane