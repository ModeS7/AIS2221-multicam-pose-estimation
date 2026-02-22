import cv2 as cv
import numpy as np



img = cv.imread('photos/phone1.png',0)
cv.imshow("img", img)
cv.waitKey(0)
rotV = np.array([-0.08243321, 0.39044282, 0.05824072])

rotM = cv.Rodrigues(-rotV)[0]
print(rotM)