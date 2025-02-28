import cv2 as cv
import numpy as np

img = cv.imread('assets/messi.png', cv.IMREAD_GRAYSCALE)

sift = cv.SIFT_create()
kp, des = sift.detectAndCompute(img, None)

print(len(kp))
cv.imshow('Sift keypoints', img)
cv.waitKey(0)
cv.destroyAllWindows()
