import cv2
import numpy as np



class getCircleGrid():  
  def __init__(self):
    # Set filtering parameters
    # Initialize parameter setting using cv2.SimpleBlobDetector
    self.params = cv2.SimpleBlobDetector_Params()

    self.params.minArea = 50
    self.params.minCircularity = 0.7

  def area(self, value):
    image = self.image
    self.params.minArea = value

    # Create a detector with the parameters
    detector = cv2.SimpleBlobDetector_create(self.params)

    # Detect blobs
    self.keypoints = detector.detect(self.image)

    # Draw blobs on our image as red circles
    blank = np.zeros((1, 1))
    image = cv2.drawKeypoints(image, self.keypoints, blank, (0, 0, 255),
    cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    for keypoint in self.keypoints:
      cv2.drawMarker(image, (int(keypoint.pt[0]), int(keypoint.pt[1])), (0,255,0))

    windowName = 'image'

    res_img = cv2.resize(image, (1080,720))
    cv2.imshow(windowName, res_img)

  def circularity(self, value):
    image = self.image
    self.params.minCircularity = value/100

    # Create a detector with the parameters
    detector = cv2.SimpleBlobDetector_create(self.params)

    # Detect blobs
    self.keypoints = detector.detect(self.image)

    # Draw blobs on our image as red circles
    blank = np.zeros((1, 1))
    image = cv2.drawKeypoints(image, self.keypoints, blank, (0, 0, 255),
    cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    for keypoint in self.keypoints:
      cv2.drawMarker(image, (int(keypoint.pt[0]), int(keypoint.pt[1])), (0,255,0))

    windowName = 'image'

    res_img = cv2.resize(image, (1080,720))
    cv2.imshow(windowName, res_img)

  def find_marker_circle(self, image):
    self.image = image

    # Set Area filtering parameters
    self.params.filterByArea = True
    self.params.minArea = self.params.minArea

    # Set Circularity filtering parameters
    self.params.filterByCircularity = True
    self.params.minCircularity = self.params.minCircularity

    # Create a detector with the parameters
    detector = cv2.SimpleBlobDetector_create(self.params)

    # Detect blobs
    self.keypoints = detector.detect(self.image)

    # Draw blobs on our image as red circles
    blank = np.zeros((1, 1))
    image = cv2.drawKeypoints(image, self.keypoints, blank, (0, 0, 255),
    cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.drawMarker(image, (int(self.keypoints[0].pt[0]), int(self.keypoints[0].pt[1])), (0,255,0))

    windowName = 'image'

    res_img = cv2.resize(image, (1080,720))
    cv2.imshow(windowName, res_img)
    cv2.createTrackbar('Area', windowName, 0, 100, self.area)
    cv2.createTrackbar('Circulariy', windowName, 0, 100, self.circularity)

circles = getCircleGrid()
img = cv2.imread("images/screen/stereoLeft/imageL0.png")

circles.find_marker_circle(img)



cv2.waitKey(0)
cv2.destroyAllWindows()

print(len(circles.keypoints))