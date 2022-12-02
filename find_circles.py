import cv2
import numpy as np

from math import dist

class getCircleGrid():
  def __init__(self, image, _flag_screen=False):
    self.image = image

    # get size of image
    self.scale = 2
    width = int(image.shape[1]/self.scale)
    height = int(image.shape[0]/self.scale)
    self.size = (width, height)

    # Set filtering parameters
    # Initialize parameter setting using cv2.SimpleBlobDetector
    self.params = cv2.SimpleBlobDetector_Params()

    self.params.minArea = 50
    self.params.maxArea = 10000
    self.params.minCircularity = 0.7
    self.params.minThreshold = 0

    self.keypoints = []
    self.center_screen = (0, 0)

    self.windowName = 'image'

    res_img = cv2.resize(self.image, self.size)
    cv2.imshow(self.windowName, res_img)
    cv2.createTrackbar('Area', self.windowName, 0, 10000, self.area)
    cv2.createTrackbar('Max Area', self.windowName, 10000, 10000, self.max_area)
    cv2.createTrackbar('Circulariy', self.windowName, 0, 100, self.circularity)
    cv2.createTrackbar('Threshold', self.windowName, 0, 255, self.binarize)
    if _flag_screen:
      cv2.setMouseCallback(self.windowName, self.mouse_event)

  def mouse_event(self, event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
      temp_img = self.image
      distances = []
      # find the shortest distance from the mouseclick to the keypoints
      for keypoint in self.keypoints:
        distance = dist((self.scale*x, self.scale*y), keypoint.pt)
        distances.append(distance)
        idx = distances.index(min(distances))

      # draw a marker for the keypoint closest to the mouseclick
      temp_img = cv2.drawMarker(temp_img, (int(self.keypoints[idx].pt[0]), int(self.keypoints[idx].pt[1])), (0,255,0))
      temp_img = cv2.resize(temp_img, self.size)
      cv2.imshow(self.windowName, temp_img)

      # safe the keypoint as a variable
      self.center_screen = self.keypoints[idx].pt

  def area(self, value):
    self.params.minArea = value
    self.calculate_keypoints()

  def max_area(self, value):
    self.params.maxArea = value
    self.calculate_keypoints()

  def circularity(self, value):
    self.params.minCircularity = value
    self.calculate_keypoints()

  def binarize(self, value):
    self.params.minThreshold = value
    self.calculate_keypoints()

  def calculate_keypoints(self):
    temp_img = self.image
    # Create a detector with the parameters
    detector = cv2.SimpleBlobDetector_create(self.params)

    # Detect blobs
    self.keypoints = detector.detect(self.image)

    # Draw blobs on our image as red circles
    blank = np.zeros((1, 1))
    temp_img = cv2.drawKeypoints(temp_img, self.keypoints, blank, (0, 0, 255),
    cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # resize image and show it
    temp_img = cv2.resize(temp_img, self.size)
    cv2.imshow(self.windowName, temp_img)

if __name__ == "__main__":
  circleGrid = getCircleGrid(cv2.imread("images/screen/stereoRight/imageR0.png"), True)

  cv2.waitKey(0)
  cv2.destroyAllWindows()
  print(circleGrid.center_screen)