import cv2 #opencv 读取的格式是BGR
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread('test.jpg')   #HWC

cv2.imshow('image',img)

img.shape #(H,W,C)

img2 = cv2.imread('test.jpg', cv2.IMREAD_GRAYSCALE)

cv2.imwrite('test_gray.jpg',img2)