import numpy as np      # noqa
import cv2
from matplotlib import pyplot as plt

image =  'PLGlnWj.png'
img = cv2.imread(image)
imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(imgray, 127, 255, 0)

kernel = np.ones((7,7),np.uint8)
dilation = cv2.dilate(img,kernel,iterations = 10)

edges = cv2.Canny(img,200,300)
contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(img, contours, -1, (0,255,0), 1)
plt.imshow(edges)
plt.imshow(img)
