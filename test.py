import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation

from triangulate import *
from find_circles import *
from intersect import *
from plane import *

wait = 0
img1 = cv2.imread("images/screen/stereoLeft/imageL0.png")
img2 = cv2.imread("images/screen/stereoRight/imageR0.png")

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
fig = plt.figure("Mirror")
ax = fig.add_subplot(projection='3d')

# plot vector cam1
ax.quiver(0,0,0,0,0,1, length=10)

# plot vector cam2
R_cam2_vec = Rotation.from_matrix(R).as_rotvec()
T_cam2_vec = T.T[0]
ax.quiver(T_cam2_vec[0], T_cam2_vec[1], T_cam2_vec[2], R_cam2_vec[0], R_cam2_vec[1], R_cam2_vec[2], length=10)

# get virtual calibration circle grid of screen and middel of screen from camera 1
circleGridScreen1 = getCircleGrid(img1, 4, 6, 10, True)
cv2.waitKey(wait)
cv2.destroyAllWindows()

# convert keypoints to imgpoints1
imgpointsVirtual1 = circleGridScreen1.centers[:,0]
imgpointsVirtual1_dst = cv2.undistortPoints(imgpointsVirtual1, mtx1, dist1)[:,0]

# get virtual calibration circle grid of scren and middel of screen from camera 2
circleGridScreen2 = getCircleGrid(img2, 4, 6, 10, True)
cv2.waitKey(wait)
cv2.destroyAllWindows()

# convert keypoints to imgpoints2
imgpointsVirtual2 = circleGridScreen2.centers[:,0]
imgpointsVirtual2_dst = cv2.undistortPoints(imgpointsVirtual2, mtx2, dist2)[:,0]

# triangulate 3d calibration patern points
virtual_screen_points_3d = triangulate(mtx1, mtx2, R, T, imgpointsVirtual1, imgpointsVirtual2)

# calculate plane of mirror point and plot point scatter, wireframe and normal vector
center_virtual_screen, normal_virtual_screen = plane(virtual_screen_points_3d, ax, "Virtual screen", "g", _virtual_screen=True, _flag_plot=True)

ax.set_xlim(-30,30)
ax.set_ylim(-30,30)
ax.set_zlim(0, 60)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.legend()
plt.show()


print(virtual_screen_points_3d)