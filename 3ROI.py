import cv2

b, g, r = cv2.split(img)

img2 = cv2.merge([r,g,b])


# padding
top_size,bottom_size, left_size, right_size = (50,50,50,50)
replicate = cv2.copyMakeBorder(img, top_size, bottom_size, left_size, right_size, borderType=cv2.BORDER_REPLICATE) #复制最边缘的像素
reflect = cv2.copyMakeBorder(img, top_size, bottom_size, left_size, right_size, borderType=cv2.BORDER_REFLECT) #镜像最边缘的像素
reflect_101 = cv2.copyMakeBorder(img, top_size, bottom_size, left_size, right_size, borderType=cv2.BORDER_REFLECT_101) #和上面一样，但最边缘不反射
wrap = cv2.copyMakeBorder(img, top_size, bottom_size, left_size, right_size, borderType=cv2.BORDER_WRAP) #像素被复制到最边缘的延伸部分
constant = cv2.copyMakeBorder(img, top_size, bottom_size, left_size, right_size, borderType=cv2.BORDER_CONSTANT, value=0) #用0填充


