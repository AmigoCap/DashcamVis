import cv2
vidcap = cv2.VideoCapture('p0HmIdAW7uo.mp4')
success,image = vidcap.read()
count = 0
success = True
while success:
  success,image = vidcap.read()
  cv2.imshow('img',image)    # save frame as JPEG file
  cv2.waitKey(0)            # exit if Escape is hit
