import cv2
import numpy as np
from matplotlib import pyplot as plt

def mirror_dots(image, pattern_size, area):
  # get shape of the image
  shape = image.shape

  # Set filtering parameters
  # Initialize parameter setting using cv2.SimpleBlobDetector
  params = cv2.SimpleBlobDetector_Params()
  params.minArea = area[0]
  params.maxArea = area[1]
  params.minCircularity = 0.1
  params.minThreshold = 0
  params.filterByConvexity = False
  params.filterByInertia = False

  # find the dots on of the mirror for camera 1 and camera 2
  upper_image = image[0:int(shape[0]/2), :]
  lower_image = image[int(shape[0]/2):int(shape[0]), :]

  detector = cv2.SimpleBlobDetector_create(params)

  upper_ret, upper_centers = cv2.findCirclesGrid(upper_image, pattern_size, flags=(cv2.CALIB_CB_SYMMETRIC_GRID+cv2.CALIB_CB_CLUSTERING), blobDetector=detector)
  lower_ret, lower_centers = cv2.findCirclesGrid(lower_image, pattern_size, flags=(cv2.CALIB_CB_SYMMETRIC_GRID+cv2.CALIB_CB_CLUSTERING), blobDetector=detector)

  for center in lower_centers[:,0]:
    center[1] = center[1]+shape[0]/2

  count = 0
  centers = np.zeros((pattern_size[0]*pattern_size[1]*2, 2))
  for center in upper_centers[:,0]:
    centers[count,:] = center
    count += 1

  for center in lower_centers[:,0]:
    centers[count,:] = center
    count += 1

  image = cv2.putText(image, str(1), (int(upper_centers[0,0][0]), int(upper_centers[0,0][1])), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 4, (255,0,0), 5,
                      cv2.LINE_AA)
  image = cv2.putText(image, str(2), (int(lower_centers[0,0][0]), int(lower_centers[0,0][1])), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 4, (255,0,0), 5, 
                      cv2.LINE_AA)

  image = cv2.drawChessboardCorners(image, pattern_size, upper_centers, upper_ret)
  image = cv2.drawChessboardCorners(image, pattern_size, lower_centers, lower_ret)
  
  return image, centers

def screen_dots(image, pattern_size, area):
  # get shape of the image
  shape = image.shape

  # Set filtering parameters
  # Initialize parameter setting using cv2.SimpleBlobDetector
  params = cv2.SimpleBlobDetector_Params()
  params.minArea = area[0]
  params.maxArea = area[1]
  params.minCircularity = 0.1
  params.minThreshold = 0
  params.filterByConvexity = False
  params.filterByInertia = False

  detector = cv2.SimpleBlobDetector_create(params)

  centers = []
  ret, centers = cv2.findCirclesGrid(image, pattern_size, flags=(cv2.CALIB_CB_SYMMETRIC_GRID+cv2.CALIB_CB_CLUSTERING), blobDetector=detector)

  image = cv2.drawChessboardCorners(image, pattern_size, centers, ret)
  
  return image, centers

def show_images(images):
  # create titles of images and put them in an array
  titles = ['Camera 1', 'Camera 2']
  
  for i in range(len(images)):
    plt.subplot(1,2,i+1),plt.imshow(images[i],'gray')
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])
  plt.show()

if __name__ == "__main__":
  # read the images
  img1 = cv2.imread("images/screen/camera1/image0.png")
  img2 = cv2.imread("images/screen/camera2/image0.png")
  
  image1, mirror_dots1 = mirror_dots(img1, (4,2), (100, 1800))
  image2, mirror_dots2 = mirror_dots(img2, (4,2), (100, 1800))  

  image1, screen_dots2 = screen_dots(img1, (7,5), (660, 10000))
  image2, screen_dots3 = screen_dots(img2, (7,5), (660, 10000))

  show_images([image1, image2])