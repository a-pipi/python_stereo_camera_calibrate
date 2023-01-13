import cv2
import numpy as np

img = cv2.imread("images/led/camera2/image0.png")
ret, bin = cv2.threshold(img, 20, 40, cv2.THRESH_BINARY)

# Set filtering parameters
# Initialize parameter setting using cv2.SimpleBlobDetector
params = cv2.SimpleBlobDetector_Params()
params.minArea = 150
params.maxArea = 10000
params.minCircularity = 0.1
params.minThreshold = 20
params.maxThreshold = 255
params.thresholdStep = 5
params.filterByConvexity = False
params.minConvexity = 0.8
params.filterByInertia = False

detector = cv2.SimpleBlobDetector_create(params)

keypoints = detector.detect(img)

img = cv2.drawKeypoints(img, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

bin = cv2.resize(bin, (1080,720))
img = cv2.resize(img, (1080,720))
cv2.imshow("LED", img)
cv2.waitKey()
cv2.destroyAllWindows()