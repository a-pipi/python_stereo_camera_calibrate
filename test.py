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
rmtx_cam1_cam2 = camera_file.getNode("R").mat()
tvct_cam1_cam2 = camera_file.getNode("T").mat()

# transformation matrix from camera1 to camera2
tmtx_cam1_cam2 = np.vstack([np.hstack([rmtx_cam1_cam2, tvct_cam1_cam2]), [0,0,0,1]])


# new coordinate system origin
ori = np.vstack([0,0,0])

# import screen calibration parameters
rmtx_screen = screen_file.getNode("rmtx_screen").mat()
location_screen = screen_file.getNode("location_screen").mat()

# import led calibration parameters
led_locations = led_file.getNode("LED_locations").mat()
led1 = np.vstack([led_locations[:,0][...,None], 1]) 
led2 = np.vstack([led_locations[:,1][...,None], 1]) 
led3 = np.vstack([led_locations[:,2][...,None], 1]) 


T_inv = np.vstack([np.hstack((rmtx_screen.T, -rmtx_screen.T.dot(location_screen))), [0,0,0,1]])

light_source_1_screen = T_inv.dot(led1)
light_source_2_screen = T_inv.dot(led2)
light_source_3_screen = T_inv.dot(led3)



print("klaas")