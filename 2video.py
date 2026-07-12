import cv2

vc = cv2.VideoCapture('2.mp4')
if vc.isOpened():
    open, frame = vc.read()
else:
    open = False

while open:
    ret, frame = vc.read()
    if ret == True:
        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)