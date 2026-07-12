# 阈值
ret, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
ret, thresh1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
ret, thresh2 = cv2.threshold(img, 127, 255, cv2.THRESH_TRUNC)
ret, thresh3 = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO)
ret, thresh4 = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO_INV)


# 滤波，去掉噪音点
blur = cv2.blur(img, (5, 5)) # 均值滤波，简单的平均卷积操作               

box = cv2.boxFilter(img, -1, (5, 5), normalize=True) # true和均值滤波一模一样，false是方框滤波

blur = cv2.GaussianBlur(img, (5, 5), 0) # 高斯滤波，模糊程度取决于高斯核的标准差

blur = cv2.medianBlur(img, 5) # 中值滤波，去掉椒盐噪声



# 边缘检测canny
# 1. 高斯滤波，降噪 
# 2. 计算梯度
# 3. 非极大值抑制    线性插值法
# 4. 滞后阈值
img = cv2.imread('messi5.jpg', 0)
v1 = cv2.Canny(img, 100, 200)
v2 = cv2.Canny(img, 200, 400)   # 越小越精细
res = np.hstack((v1, v2))
