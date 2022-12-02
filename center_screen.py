import cv2
import numpy as np

center = (960,540)
size_square = (100,100)
p1 = (int(center[0]-size_square[0]/2), int(center[1]-size_square[1]/2))
p2 = (int(p1[0]+size_square[0]), int(p1[1]+size_square[1]))

print(p1, p2)

img = np.zeros([1080,1920,3],dtype=np.uint8)
img.fill(255)
img = cv2.circle(img, center, 10, (0,0,0), -1)
img = cv2.rectangle(img, p1, p2, (0,0,0), 1)

cv2.namedWindow("window", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("window", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
cv2.moveWindow("window", 1920, 0)
cv2.imshow("window", img)
cv2.waitKey()