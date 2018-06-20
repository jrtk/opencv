import cv2
import numpy as np
from matplotlib import pyplot as plt

#Choose any image
img1 = cv2.imread("./img1.jpeg", cv2.IMREAD_COLOR)

print ("Image shape is ", img1.shape)
rows, cols, _ = img1.shape


M = cv2.getRotationMatrix2D((cols/2, rows/2), 5,1)
dst = cv2.warpAffine(img1, M, (cols, rows))



img2 = cv2.imread("./img2.jpeg", cv2.IMREAD_COLOR)

img1Gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img2Gray = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
#img2Gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

#Let's find ORB ( This includes locator and desctriptor)

orbObj  = cv2.ORB_create(50)

img1Locator, img1Descriptor = orbObj.detectAndCompute(img1Gray, None)
img2Locator, img2Descriptor = orbObj.detectAndCompute(img1Gray, None)

matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = matcher.match(img1Descriptor, img2Descriptor, None)

matches.sort(key=lambda x: x.distance, reverse=False)
img3 = cv2.drawMatches(img1, img1Locator, dst, img2Locator,matches[:10], None, flags=2)
plt.imshow(img3), plt.show()
cv2.drawMatches()






