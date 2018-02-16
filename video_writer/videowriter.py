import cv2
import numpy as np

#convert from RGB of PIL to BGR of OpenCV
img_out_cv2 = cv2.imread('out_300.png')
frameH, frameW, channels = img_out_cv2.shape

fourcc =  cv2.cv.CV_FOURCC(*'XVID')
video = cv2.VideoWriter("output.avi", fourcc, 15, (frameW,frameH), 1)

for i in range(300,501):
	img_out_cv2 = cv2.imread("out_"+str(i)+".png")
	video.write(img_out_cv2)

video.release()