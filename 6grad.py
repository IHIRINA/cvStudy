# 梯度= 膨胀-腐蚀
import cv2
import numpy as np

pie = cv2.imread('pie.png')
kernel = np.ones((5,5), np.uint8)
dilate = cv2.dilate(pie, kernel, iterations=1)
erode = cv2.erode(pie, kernel, iterations=1)
res = np.hstack((dilate, erode))

gradient = cv2.morphologyEx(pie, cv2.MORPH_GRADIENT, kernel)

# 礼帽
top_hat = cv2.morphologyEx(pie, cv2.MORPH_TOPHAT, kernel) # 原图像与开运算结果的差值, 只剩刺了
# 黑帽
black_hat = cv2.morphologyEx(pie, cv2.MORPH_BLACKHAT, kernel) # 闭运算结果与原图像的差值，只剩轮廓了


# sobel算子
img = cv2.imread('pie.png', cv2.IMREAD_GRAYSCALE)
sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5) # x方向
sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5) # y方向


# scharr算子，更加敏感，细节更丰富
scharrx = cv2.Scharr(img, cv2.CV_64F, 1, 0)
scharry = cv2.Scharr(img, cv2.CV_64F, 0, 1)


# laplacian算子
laplacian = cv2.Laplacian(img, cv2.CV_64F)     # 中间点和边缘减的感觉
