import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation

from triangulate import *
from find_circles import *



def find_marker_grid(image, rows, columns, world_scaling):
  # load the image pair to process
  img = image

  #Change this if the code can't find the checkerboard
  criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

  #coordinates of squares in the checkerboard world space
  objp = np.zeros((rows*columns,3), np.float32)
  objp[:,:2] = np.mgrid[0:rows,0:columns].T.reshape(-1,2)
  objp = world_scaling* objp

  #frame dimensions. Frames should be the same size.
  width = img[0].shape[1]
  height = img[0].shape[0]

  #Pixel coordinates of checkerboards
  imgpoints = np.zeros((rows*columns, 2)) # 2d points in image plane.

  #coordinates of the checkerboard in checkerboard world space.
  objpoints = [] # 3d point in real world space

  # convert color to gray image
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

  #find the checkerboard
  ret, corners = cv2.findChessboardCorners(gray, (rows, columns), None)

  if ret:
    # Convolution size used to improve corner detection. Don't make this too large.
    conv_size = (11, 11)

    #opencv can attempt to improve the checkerboard coordinates
    corners = cv2.cornerSubPix(gray, corners, conv_size, (-1, -1), criteria)
    cv2.drawChessboardCorners(img, (rows,columns), corners, ret)

    # save object and image points
    objpoints.append(objp)

    i = 0
    for corner in corners:
      imgpoints[i] = corner
      i += 1
  
  return img, imgpoints


def find_marker_circle(image):
  # Set filtering parameters
  # Initialize parameter setting using cv2.SimpleBlobDetector
  params = cv2.SimpleBlobDetector_Params()

  # Set Area filtering parameters
  params.filterByArea = True
  params.minArea = 100

  # Set Circularity filtering parameters
  params.filterByCircularity = True
  params.minCircularity = 0.7

  # Create a detector with the parameters
  detector = cv2.SimpleBlobDetector_create(params)

  # Detect blobs
  keypoints = detector.detect(image)

  # Draw blobs on our image as red circles
  blank = np.zeros((1, 1))
  image = cv2.drawKeypoints(image, keypoints, blank, (0, 0, 255),
  cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
  cv2.drawMarker(image, (int(keypoints[0].pt[0]), int(keypoints[0].pt[1])), (0,255,0))

  return image, np.array([keypoints[0].pt])

# get calibration circle grid of mirror from camera 1
circleGrid1 = getCircleGrid(cv2.imread("images/screen/stereoLeft/imageL0.png"))
cv2.waitKey(0)

# convert keypoints to imgpoints1
imgpoints1 = np.zeros((len(circleGrid1.keypoints),2))
idx = 0
for keypoint in circleGrid1.keypoints:
  imgpoints1[idx] = np.array([keypoint.pt])
  idx += 1

# get calibration circle grid of mirror from camera 2
circleGrid2 = getCircleGrid(cv2.imread("images/screen/stereoRight/imageR0.png"))
cv2.waitKey(0)
cv2.destroyAllWindows()

# convert keypoints to imgpoints2
imgpoints2 = np.zeros((len(circleGrid2.keypoints),2))
idx = 0
for keypoint in circleGrid2.keypoints:
  imgpoints2[idx] = np.array([keypoint.pt])
  idx += 1

# get camera matrices and rotation and translation matrix from calibration file
calib_file = cv2.FileStorage('stereoCalibration.XML', cv2.FileStorage_READ)
mtx1 = calib_file.getNode("mtx1").mat()
mtx2 = calib_file.getNode("mtx2").mat()
R = calib_file.getNode("R").mat()
T = calib_file.getNode("T").mat()

# triangulate 3d calibration patern points
mirror_points_3d = triangulate(mtx1, mtx2, R, T, imgpoints1, imgpoints2)

# plot 3d points and cameras
fig = plt.figure("Mirror")
ax = fig.add_subplot(projection='3d')

# plot vector cam1
ax.quiver(0,0,0,0,0,1, length=10)

# plot vector cam2
R_cam2_vec = Rotation.from_matrix(R).as_rotvec()
T_cam2_vec = T.T[0]
ax.quiver(T_cam2_vec[0], T_cam2_vec[1], T_cam2_vec[2], R_cam2_vec[0], R_cam2_vec[1], R_cam2_vec[2], length=10)

# plot all mirrorpoints
ax.scatter(mirror_points_3d[:,0], mirror_points_3d[:,1], mirror_points_3d[:,2])

# extrapolate xs, ys and zs
xs = mirror_points_3d[:,0]
ys = mirror_points_3d[:,1]
zs = mirror_points_3d[:,2]

# do fit
tmp_A = []
tmp_b = []
for i in range(len(xs)):
    tmp_A.append([xs[i], ys[i], 1])
    tmp_b.append(zs[i])
b = np.matrix(tmp_b).T
A = np.matrix(tmp_A)

# Manual solution
fit = (A.T * A).I * A.T * b
errors = b - A * fit
residual = np.linalg.norm(errors)


# plot plane
xlim = ax.get_xlim()
ylim = ax.get_ylim()
X,Y = np.meshgrid(np.arange(xlim[0], xlim[1]),
                  np.arange(ylim[0], ylim[1]))
Z = np.zeros(X.shape)
for r in range(X.shape[0]):
    for c in range(X.shape[1]):
        Z[r,c] = fit[0] * X[r,c] + fit[1] * Y[r,c] + fit[2]
ax.plot_wireframe(X,Y,Z, color='k')

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.show()


rows = 6
columns = 10
world_scaling = 25

# find markers on the screen and get image and imagepoints
img1, imgpoints1 = find_marker_grid(cv2.imread("images/screen/stereoLeft/imageL0.png"), rows, columns, world_scaling)
img2, imgpoints2 = find_marker_grid(cv2.imread("images/screen/stereoRight/imageR0.png"), rows, columns, world_scaling)

# triangulate screen markers
screen_points_3d = triangulate(mtx1, mtx2, R, T, imgpoints1, imgpoints2)

# # find the central circular screen marker
# img1, center1 = find_marker_circle(img1)
# img2, center2 = find_marker_circle(img2)

# # triangulate central screen maker
# screen_central_point_3d = triangulate(mtx1, mtx2, R, T, center1, center2)

cv2.imshow("window", img1)
cv2.waitKey()
cv2.destroyAllWindows()
