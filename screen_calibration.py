import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation
from scipy.spatial import KDTree, procrustes
import glob

from drawCoordinateSystem import *
from plane import *
from intersectLinePlane import *
from find_dots import *


def screen_points(img_mirror_points1, img_mirror_points2, img_screen_points1, img_screen_points2):
  # triangulate 3d calibration pattern points
  mirror_points_3d = cv2.triangulatePoints(P1, P2, img_mirror_points1.T, img_mirror_points2.T)
  mirror_points_3d /= mirror_points_3d[3]
  mirror_points_3d = mirror_points_3d[0:3,:]

  # calculate plane of mirror point and plot point scatter, wireframe and normal vector
  center_mirror, normal_mirror = plane(mirror_points_3d, ax, "Mirror", "r", _flag_plot=False)

  # triangulate 3d calibration patern points
  virtual_screen_points_3d = cv2.triangulatePoints(P1, P2, img_screen_points1, img_screen_points2)
  virtual_screen_points_3d /= virtual_screen_points_3d[3]
  virtual_screen_points_3d = virtual_screen_points_3d[0:3,:]

  # get real screen points
  k=0
  intersect_points = np.zeros(virtual_screen_points_3d.shape)
  screen_points = np.zeros(virtual_screen_points_3d.shape)

  for point in virtual_screen_points_3d.T:
    temp = intersect_line_plane(point, normal_mirror, center_mirror, normal_mirror)
    intersect_points[:,k] = temp
    screen_points[:,k] = intersect_points[:,k] - (virtual_screen_points_3d[:,k] - intersect_points[:,k])
    k+=1

  return screen_points


def center_screen(images1, images2):
  # sort and read the synched frames
  c1_images_names = sorted(glob.glob(images1))
  c2_images_names = sorted(glob.glob(images2))

  c1_images = [cv2.imread(img1, 1) for img1 in c1_images_names]
  c2_images = [cv2.imread(img2, 1) for img2 in c2_images_names]

  # determine location of mirror and screen dots in image
  img_mirror_points1 = [mirror_dots(img1, (4,2), (100, 1800))[1] for img1 in c1_images]
  img_mirror_points2 = [mirror_dots(img2, (4,2), (100, 1800))[1] for img2 in c2_images]
  img_screen_points1 = [screen_dots(img1, (7,5), (660, 10000))[1] for img1 in c1_images]
  img_screen_points2 = [screen_dots(img2, (7,5), (660, 10000))[1] for img2 in c2_images]

  counter = 0
  for img1, img2 in zip(c1_images, c2_images):
    images = [img1, img2]
    titles = [c1_images_names[counter], c2_images_names[counter]]

    # plot the found dots on the images and show
    for i in range(len(images)):
      plt.subplot(1,2,i+1),plt.imshow(images[i],'gray')
      plt.title(titles[i])
      plt.xticks([]),plt.yticks([])
    plt.get_current_fig_manager().window.showMaximized()
    plt.show()

    counter += 1
  
  plt.close()
  _screen_points_3d = [screen_points(img_mirror_points1[i], img_mirror_points2[i], img_screen_points1[i], img_screen_points2[i]) for i in range(len(img_mirror_points1))]
  screen_points_3d = np.mean(np.asarray(_screen_points_3d), axis=0)

  # get calculated centers
  l = len(screen_points_3d[0])//2
  center_screen_3d = screen_points_3d[:,l]
  
  # calculate normale of the screen
  normal_screen = norm_plane(screen_points_3d)
  if normal_screen[2] < 0:
    normal_screen = -normal_screen

  return center_screen_3d, normal_screen, screen_points_3d


wait = 0
img1 = cv2.imread("images/screen/camera1/image4.png")
img2 = cv2.imread("images/screen/camera2/image4.png")

# get camera matrices and rotation and translation matrix from calibration file
calib_file = cv2.FileStorage('stereoCalibration.XML', cv2.FileStorage_READ)
mtx1 = calib_file.getNode("mtx1").mat()
mtx2 = calib_file.getNode("mtx2").mat()
dist1 = calib_file.getNode("dist1").mat()
dist2 = calib_file.getNode("dist2").mat()
R = calib_file.getNode("R").mat()
T = calib_file.getNode("T").mat()

RT1 = np.concatenate([np.eye(3), [[0],[0],[0]]], axis = -1)
P1 = mtx1 @ RT1 #projection matrix of camera 1
RT2 = np.concatenate([R, T], axis = -1)
P2 = mtx2 @ RT2 #projection matrix of camera 2


# plot 3d points and cameras
fig = plt.figure("Screen dots")
ax = fig.add_subplot(projection='3d')

center_screen_3d, normal_screen, screen_points_3d = center_screen("images/screen/camera1/*", "images/screen/camera2/*")

# plot 3d points and cameras
fig = plt.figure("Screen location")
ax = fig.add_subplot(projection='3d')

z_axis = [0, 0, 1]
# Calculate the rotation axis
u = np.cross(normal_screen, z_axis)
u = np.float64(u)

# Normalize the rotation axis to get a unit vector
u /= np.linalg.norm(u)

# Calculate the rotation angle
angle = np.arccos(np.dot(normal_screen, z_axis))

# Calculate the rotation vector
r = angle*u

# Normalize the rotation vector
r_norm = r / np.linalg.norm(r)

# Calculate the rotation angle
angle = np.linalg.norm(r)

# Create the rotation matrix using Rodrigues' rotation formula
rmtx_z = np.eye(3) + np.sin(angle) * np.array([[0, -r_norm[2], r_norm[1]], [r_norm[2], 0, -r_norm[0]], [-r_norm[1], r_norm[0], 0]]) + (1 - np.cos(angle)) * np.outer(r_norm, r_norm)

# put the screen points in the xy-plane
xypoints = np.dot((screen_points_3d.T-center_screen_3d), np.linalg.inv(rmtx_z))[:,0:2]

# set fixed and moving points
fixed = np.array([[3-j, 2-i] for i in range(5) for j in range(7)])
# Calculate the rotation matrix using the Singular Value Decomposition (SVD) method
U, _, VT = np.linalg.svd(fixed.T @ xypoints)

# calculate rotation matrix
rot_mtx = U @ VT

# calculate the in plane rotation angle
inplane_angle = np.arctan2(rot_mtx[1,0], rot_mtx[0,0])

# get inplane rotation matrix from rotation vector
r = Rotation.from_rotvec([0,0,inplane_angle])
rmtx_ip = r.as_matrix()

# Rotation matrix of the screen
rmtx_s = rmtx_z @ rmtx_ip

# plot vector cam2
R_cam2_vec = Rotation.from_matrix(R).as_rotvec()
T_cam2_vec = T.T[0]
nodalpoint_camera2 = -np.dot(T.T, R)[0]

# ax.quiver(center_screen_3d[0], center_screen_3d[2], center_screen_3d[1], normal_screen[0]*xscale, normal_screen[2]*zscale, normal_screen[1]*yscale, length=50, color='purple')
draw_coordinate_system([0,0,0], "camera1", ax)
draw_coordinate_system(nodalpoint_camera2, "camera2", ax, R=R, color='yellow')
draw_coordinate_system(center_screen_3d, "screen", ax, R=rmtx_s, color='cyan')
ax.scatter(-150, 700, 0, color='blue', label='distant point')
# ax.set_xlim([-200,1000])
# ax.set_ylim([-200,1000])
# ax.set_zlim([-200,1000])

# set axis labels and legens
ax.set_xlabel('x')
ax.set_ylabel('z')
ax.set_zlabel('y')
ax.invert_zaxis()
ax.legend()

# set plot fullscreen and show
plt.get_current_fig_manager().window.showMaximized()
plt.show()

# Save parameters to XML file
cv_file = cv2.FileStorage('screenCalibration.XML', cv2.FileStorage_WRITE)
cv_file.write("rmtx_screen", rmtx_s)
cv_file.write("location_screen", center_screen_3d)

print(np.linalg.norm(center_screen_3d), center_screen_3d)