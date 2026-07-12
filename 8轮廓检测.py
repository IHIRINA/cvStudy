img = cv2.imread('./images/1.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)   # 二值化

contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)     

cv2.drawContours(img, contours, -1, (0, 0, 255), 3) # BGR, 3是线条宽度
cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 轮廓 Approximation
img = cv2.imread('./images/1.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
epsilon = 0.01 * cv2.arcLength(contours[0], True)
approx = cv2.approxPolyDP(contours[0], epsilon, True)

draw_img = img.copy()
cv2.drawContours(draw_img, [approx], -1, (0, 0, 255), 3)
cv2.imshow('draw_img', draw_img)