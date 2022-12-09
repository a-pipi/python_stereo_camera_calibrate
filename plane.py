import numpy as np

def plane(points, ax, label, color, _flag_plot=True, _virtual_screen=False):
  points = points.T
  
  if _flag_plot:
    # plot all screenpoints
    # ax.scatter(points[0,:], points[1,:], points[2,:], color=color, label=label)
    ax.scatter(points[:,0], points[:,1], points[:,2], color=color, label=label)

  length = 10

  # position of center of plane 
  mean_point = np.mean(points, axis=1, keepdims=True)

  # svd of points - mean_point
  svd = np.linalg.svd(points - mean_point) 

  # left singular value
  left = svd[0]

  # normal to the points
  normal = left[:,-1]

  # normal should point to the camera
  if normal[2] > 0:
    normal = -normal

  if _flag_plot:
    ax.quiver(mean_point[0], mean_point[1], mean_point[2], normal[0], normal[1], normal[2], length=length, color=color)
  
  
  return mean_point, normal