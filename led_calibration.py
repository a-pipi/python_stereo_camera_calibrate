import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation
import glob

from plane import *
from intersectLinePlane import *
from find_dots import *

wait = 0

# import images camera 1
images_folder1 = 'images/led/camera1/*'
images_names1 = sorted(glob.glob(images_folder1))
images1 = []
for imname in images_names1:
  im = cv2.imread(imname, 1)
  images1.append(im)

#import images camera 2
images_folder2 = 'images/led/camera2/*'
images_names2 = sorted(glob.glob(images_folder2))
images2 = []
for imname in images_names2:
  im = cv2.imread(imname, 1)
  images2.append(im)

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
imgpoints1 = []
images1_ = []
led_locations1 = []
for image in images1:
  temp_img, temp_imgpoints = mirror_dots(image, (4,2), (100, 1800))

  # get the locations of the screen dots from the images
  image1, temp_location = led_dots(image, (100, 10000))

  images1_.append(temp_img)
  imgpoints1.append(temp_imgpoints)
  led_locations1.append(temp_location)

imgpoints2 = []
images2_ = []
led_locations2 = []
for image in images2:
  temp_img, temp_imgpoints = mirror_dots(image, (4,2), (50, 10000))

  # get the locations of the screen dots from the images
  image2, temp_location = led_dots(image, (100, 10000))

  images2_.append(temp_img)
  imgpoints2.append(temp_imgpoints)
  led_locations2.append(temp_location)



titles = ['Camera 1', 'Camera 2']

# plot the found dots on the images and show
for i in range(len(images1)):
  plt.subplot(1,2,1),plt.imshow(images1[i],'gray')
  plt.subplot(1,2,2),plt.imshow(images2[i],'gray')
  # plt.title(titles[i])
  plt.xticks([]),plt.yticks([])
  plt.get_current_fig_manager().window.showMaximized()
  plt.show()

# plot 3d points and cameras
fig = plt.figure("LED sources")
ax = fig.add_subplot(projection='3d')

# plot vector cam1
ax.scatter(0,0,0, marker="*", color="g", linewidths=1.5, label='camera 1')

# plot vector cam2
R_cam2_vec = Rotation.from_matrix(R).as_rotvec()
T_cam2_vec = T.T[0]
nodalpoint_camera2 = -np.dot(T.T, R)[0]
ax.scatter(nodalpoint_camera2[0], nodalpoint_camera2[1], nodalpoint_camera2[2], marker="*", color="r", linewidths=1.5, label='camera2')

# triangulate 3d calibration pattern points on the mirror
mirror_points_3d = []
for i in range(len(imgpoints1)):
  temp_points = cv2.triangulatePoints(P1, P2, imgpoints1[i].T, imgpoints2[i].T)
  temp_points /= temp_points[3]
  temp_points = temp_points[0:3,:]
  mirror_points_3d.append(temp_points)



# calculate centers and normals of the mirrors
center_mirror=[]
normal_mirror=[]
for i in range(len(mirror_points_3d)):
  temp_center, temp_normal = plane(mirror_points_3d[i], ax, "Mirror", "r", _flag_plot=False)
  center_mirror.append(temp_center)
  normal_mirror.append(temp_normal)

# calculate led positions
virtual_led_points_3d = []
for i in range(len(led_locations1)):
  temp_led = cv2.triangulatePoints(P1, P2, led_locations1[i], led_locations2[i])
  temp_led /= temp_led[3]
  temp_led = temp_led[0:3,:]
  virtual_led_points_3d.append(temp_led)

virtual_led_points_3d = np.array(virtual_led_points_3d)[:,:,0]
# mirror_points_3d = np.array(mirror_points_3d)[:,:,0]

# calculate led_points
intersect_points = np.zeros(virtual_led_points_3d.shape)
led_points = np.zeros(virtual_led_points_3d.shape)
for i in range(len(virtual_led_points_3d)):
  temp = intersect_line_plane(virtual_led_points_3d[i], normal_mirror[i], center_mirror[i], normal_mirror[i])
  intersect_points[:,i] = temp
  led_points[:,i] = intersect_points[:,i] - (virtual_led_points_3d.T[:,i] - intersect_points[:,i])

virtual_led_points_3d = virtual_led_points_3d.T

ax.scatter(led_points[0,:], led_points[2,:], led_points[1,:], marker="^", label='LED sources')
ax.scatter(mirror_points_3d[0][0,:], mirror_points_3d[0][2,:], mirror_points_3d[0][1,:], label='Mirror1')
ax.scatter(mirror_points_3d[1][0,:], mirror_points_3d[1][2,:], mirror_points_3d[1][1,:], label='Mirror2')
ax.scatter(mirror_points_3d[2][0,:], mirror_points_3d[2][2,:], mirror_points_3d[2][1,:], label='Mirror3')
ax.scatter(intersect_points[0,0], intersect_points[2,0], intersect_points[1,0], marker="v", label='intersect1')
ax.scatter(intersect_points[0,1], intersect_points[2,1], intersect_points[1,1], marker="v", label='intersect2')
ax.scatter(intersect_points[0,2], intersect_points[2,2], intersect_points[1,2], marker="v", label='intersect3')
ax.scatter(virtual_led_points_3d[0,0], virtual_led_points_3d[2,0], virtual_led_points_3d[1,0], marker="v", label='virtual1')
ax.scatter(virtual_led_points_3d[0,1], virtual_led_points_3d[2,1], virtual_led_points_3d[1,1], marker="v", label='virtual2')
ax.scatter(virtual_led_points_3d[0,2], virtual_led_points_3d[2,2], virtual_led_points_3d[1,2], marker="v", label='virtual3')


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
cv_file = cv2.FileStorage('ledCalibration.XML', cv2.FileStorage_WRITE)
cv_file.write("LED_locations", led_points)