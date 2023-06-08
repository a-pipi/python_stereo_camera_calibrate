import cv2
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

from drawCoordinateSystem import *

# load calibration files
camera_file = cv2.FileStorage('stereoCalibration.XML', cv2.FileStorage_READ)
screen_file = cv2.FileStorage('screenCalibration.XML', cv2.FileStorage_READ)
led_file = cv2.FileStorage('ledCalibration.XML', cv2.FileStorage_READ)

# import camera calibration parameters
mtx1 = camera_file.getNode("mtx1").mat()
mtx2 = camera_file.getNode("mtx2").mat()
dist1 = camera_file.getNode("dist1").mat()
dist2 = camera_file.getNode("dist2").mat()
R = camera_file.getNode("R").mat()
T = camera_file.getNode("T").mat()

# transformation matrix from camera1 to camera2
tmtx_cam1_cam2 = np.vstack([np.hstack([R, T]), [0,0,0,1]])


# new coordinate system origin
ori = np.vstack([0,0,0])

# import screen calibration parameters
rmtx_screen = screen_file.getNode("rmtx_screen").mat()
location_screen = screen_file.getNode("location_screen").mat()

# import led calibration parameters
led_locations = led_file.getNode("LED_locations").mat()


# plot setup
fig = plt.figure("setup")
ax = fig.add_subplot(projection='3d')

trans = np.concatenate([rmtx_screen, location_screen], axis=-1)
trans = np.vstack([trans, [0,0,0,1]])

tmtx1 = np.linalg.inv(trans)
rmtx1 = np.eye(3) #tmtx1[0:3,0:3] #np.linalg.inv(rmtx_screen)
nodalpoint_camera1 = np.array([0,0,0]) #tmtx1[0:3,3] #np.vstack([-location_screen, 1])


tmtx_1_2 = np.concatenate([R, T], axis=-1)
tmtx_1_2 = np.vstack([tmtx_1_2, [0,0,0,1]])
tmtx2 = np.linalg.inv(tmtx_1_2)
rmtx2 = R#rmtx1 @ R
nodalpoint_camera2 =  R.T @ -T#tmtx2[0:3,3] + nodalpoint_camera1

led1 = np.vstack([led_locations[:,0][...,None], 1]) 
led3 = np.vstack([led_locations[:,1][...,None], 1]) 
led2 = np.vstack([led_locations[:,2][...,None], 1]) 


draw_coordinate_system(location_screen, "Screen", ax, R=rmtx_screen, color='blue')
draw_coordinate_system(nodalpoint_camera1, "Camera1", ax, R=rmtx1, color='red')
draw_coordinate_system(nodalpoint_camera2, "Camera2", ax, R=rmtx2, color='green')
ax.scatter(led1[0], led1[2], led1[1], color='yellow', marker='^', label="LED1")
ax.scatter(led2[0], led2[2], led2[1], color='cyan', marker='^', label="LED2")
ax.scatter(led3[0], led3[2], led3[1], color='purple', marker='^', label="LED3")
ax.scatter(0, 700, 0)

# set axis labels and legens
ax.set_xlabel('x')
ax.set_ylabel('z')
ax.set_zlabel('y')
ax.invert_zaxis()
ax.legend()

ax.set_xlim((-550,150))
ax.set_ylim((0, 700))
ax.set_zlim((600,-100))

print(f"screen: {np.linalg.norm(location_screen)}\n camera1: {nodalpoint_camera1}\n camera2: {np.linalg.norm(nodalpoint_camera2)}\n LED1: {np.linalg.norm(led1)}\n LED2: {np.linalg.norm(led2)}\n LED3: {np.linalg.norm(led3)}")

# set plot fullscreen and show
plt.get_current_fig_manager().window.showMaximized()
plt.show()

# Save parameters to XML file
cv_file = cv2.FileStorage('systemSetupParameters.XML', cv2.FileStorage_WRITE)
cv_file.write("rmtx1", rmtx1)
cv_file.write("rmtx2", rmtx2)
cv_file.write("nodalpoint_camera1", nodalpoint_camera1)
cv_file.write("nodalpoint_camera2", nodalpoint_camera2)
cv_file.write("LED1_location", led1)
cv_file.write("LED2_location", led2)
cv_file.write("LED3_location", led3)