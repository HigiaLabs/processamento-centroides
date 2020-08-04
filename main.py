import numpy as np      # noqa
import cv2
from matplotlib import pyplot as plt

image =  'PLGlnWj.png'

img = cv2.imread(image,0)

hist = cv2.calcHist([img],[0],None,[256],[0,256])

cols, rows = img.shape
brightness = np.sum(img) / (255 * cols * rows)

M = cv2.moments(img)

cX =    int(M['m10'] /  M['m00'])
cY =   int( M['m01'] / M['m00'])

cv2.circle(img, (cX, cY), 5, (255, 255, 255), -1)
responses = np.random.randint(0,2,(25,1)).astype(np.float32)


criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
ret,label,center=cv2.kmeans(img,2,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)


cv2.imshow('Aaoba', img)

cv2.waitKey(0)
cv2.destroyAllWindows()
exit()