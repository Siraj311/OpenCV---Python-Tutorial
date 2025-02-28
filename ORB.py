import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img = cv.imread('assets/messi.png', cv.IMREAD_GRAYSCALE)

orb = cv.ORB_create()
kp, des = orb.detectAndCompute(img, None)

print(len(kp))
cv.imshow('image', img)
cv.waitKey(0)
cv.destroyAllWindows()