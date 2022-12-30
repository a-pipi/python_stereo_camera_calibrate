import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation

from plane import *
from intersectLinePlane import *
from find_dots import *

wait = 0
img1 = cv2.imread("images/screen/camera1/image0.png")
img2 = cv2.imread("images/screen/camera2/image0.png")

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

# get the locations of the mirror dots from the images
image1, imgpoints1 = mirror_dots(img1, (4,2), (100, 1800))
image2, imgpoints2 = mirror_dots(img2, (4,2), (100, 1800))  

# get the locations of the screen dots from the images
image1, imgpointsVirtual1 = screen_dots(img1, (7,5), (660, 10000))
image2, imgpointsVirtual2 = screen_dots(img2, (7,5), (660, 10000))

images = [image1, image2]
titles = ['Camera 1', 'Camera 2']

# plot the found dots on the images and show
for i in range(len(images)):
  plt.subplot(1,2,i+1),plt.imshow(images[i],'gray')
  plt.title(titles[i])
  plt.xticks([]),plt.yticks([])
plt.get_current_fig_manager().window.showMaximized()
plt.show()

# plot 3d points and cameras
fig = plt.figure("Mirror")
ax = fig.add_subplot(projection='3d')

# plot vector cam1
ax.scatter(0,0,0, marker="*", color="g", linewidths=1.5, label='camera 1')

# plot vector cam2
R_cam2_vec = Rotation.from_matrix(R).as_rotvec()
T_cam2_vec = T.T[0]
nodalpoint_camera2 = -np.dot(T.T, R)[0]
ax.scatter(nodalpoint_camera2[0], nodalpoint_camera2[1], nodalpoint_camera2[2], marker="*", color="r", linewidths=1.5, label='camera2')

# triangulate 3d calibration pattern points
# mirror_points_3d = triangulate(mtx1, mtx2, R, T, imgpoints1_dst, imgpoints2_dst)
mirror_points_3d = cv2.triangulatePoints(P1, P2, imgpoints1.T, imgpoints2.T)
mirror_points_3d /= mirror_points_3d[3]
mirror_points_3d = mirror_points_3d[0:3,:]

# calculate plane of mirror point and plot point scatter, wireframe and normal vector
center_mirror, normal_mirror = plane(mirror_points_3d, ax, "Mirror", "r", _flag_plot=False)


# triangulate 3d calibration patern points
# virtual_screen_points_3d = triangulate(mtx1, mtx2, R, T, imgpointsVirtual1_dst, imgpointsVirtual2_dst)
virtual_screen_points_3d = cv2.triangulatePoints(P1, P2, imgpointsVirtual1, imgpointsVirtual2)
virtual_screen_points_3d /= virtual_screen_points_3d[3]
virtual_screen_points_3d = virtual_screen_points_3d[0:3,:]

# calculate plane of mirror point and plot point scatter, wireframe and normal vector
# center_virtual_screen, normal_virtual_screen = plane(virtual_screen_points_3d, ax, "Virtual screen", "g", _virtual_screen=True, _flag_plot=True)

# get real screen points
k=0
intersect_points = np.zeros(virtual_screen_points_3d.shape)
screen_points = np.zeros(virtual_screen_points_3d.shape)

for point in virtual_screen_points_3d.T:
  temp = intersect_line_plane(point, normal_mirror, center_mirror, normal_mirror)
  intersect_points[:,k] = temp
  screen_points[:,k] = intersect_points[:,k] - (virtual_screen_points_3d[:,k] - intersect_points[:,k])
  k+=1

# get center of the screen
l = len(screen_points[0])//2
center_screen = screen_points[:,l]

# calculate normale of the screen
normal_screen = norm_plane(screen_points)


# scatter intersection and screen points
ax.scatter(mirror_points_3d[0,:], mirror_points_3d[2,:], mirror_points_3d[1,:], color='gray', label='mirror points')
ax.scatter(intersect_points[0,:], intersect_points[2,:], intersect_points[1,:], color='black', label="intersection points")
ax.scatter(screen_points[0,:], screen_points[2,:], screen_points[1,:], color='blue', label='screen points')
ax.scatter(virtual_screen_points_3d[0,:], virtual_screen_points_3d[2,:], virtual_screen_points_3d[1,:], color='cyan', label='virtual screen points')
ax.scatter(center_screen[0], center_screen[2], center_screen[1], color='pink', linewidths=10, label='center screen')

# set axis labels and legens
ax.set_xlabel('x')
ax.set_ylabel('z')
ax.set_zlabel('y')
ax.invert_zaxis()
ax.legend()

# set plot fullscreen and show
plt.get_current_fig_manager().window.showMaximized()
plt.show()
print("einde")

#NOTE: save paramters to XML file