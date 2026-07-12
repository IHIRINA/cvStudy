import cv2
import numpy as np

kernel = np.ones((30,30), np.uint8)
erosion1 = cv2.erode(img, kernel, iterations=1)

dilation = cv2.dilate(img, kernel, iterations=1)

erosion2 = cv2.erode(img, kernel, iterations=2)
erosion3 = cv2.erode(img, kernel, iterations=3)
res = np.hstack((img, erosion1, erosion2, erosion3))
cv2.imshow('erosion', res)
cv2.waitKey(0)
cv2.destroyAllWindows()


# 开运算：先腐蚀再膨胀
opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

# 闭运算：先膨胀再腐蚀
closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)