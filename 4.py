import cv2

img = cv2.resize(img, (500, 414))
img.shape

res = cv2.resize(img, (0,0), fx=0.5, fy=0.5) 

ress = cv2.addWeighted(img, 0.5, res, 0.5, 0)