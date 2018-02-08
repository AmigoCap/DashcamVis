import cv2
from matplotlib import pyplot as plt
vidcap = cv2.VideoCapture('p0HmIdAW7uo.mp4')
success,image = vidcap.read()
count = 0
success = True
while success:
  success,image = vidcap.read()
  fig, ax = plt.subplots()
  ax.imshow(image, extent=[0,800,0,600])
  ax.imshow(image, extent=[0, 400, 0, 300])
  x=range(100)
  ax.plot(x, x, '--', linewidth=5, color='firebrick')
  plt.show()
  #cv2.imshow('img',image)    # save frame as JPEG file
  #cv2.waitKey(0)            # exit if Escape is hit
