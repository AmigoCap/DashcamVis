# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 12:56:46 2018

@author: Achraf Baiz
"""

import cv2 as cv
import numpy as np

def video_images():
    vidcap = cv.VideoCapture('video_1.avi')
    success,image = vidcap.read()
    count = 0
    success = True
    while success:
      success,image = vidcap.read()
      print('Read a new frame: ', success)
      cv.imwrite("frame%d.jpg" % count, image)     # save frame as JPEG file
      count += 1
      
      
if __name__=='__main__': 
    
    white_pix = 0
    black_pix = 0
    # Load an color image in color or grayscale
    img   = cv.imread('bikers.jpg',0)   
        
    #img   = cv.medianBlur(img,5)
    _,thimg = cv.threshold(img,127,255,cv.THRESH_BINARY)
    
    for i in range(thimg.shape[0]):
        for j in range(thimg.shape[1]):
            if thimg.item(i,j)==0:
                black_pix=black_pix+1
            else:
                white_pix=white_pix+1
    
    avg = black_pix/(white_pix+black_pix)
    if(avg>0.70):
        print("night")
    elif(avg>0.40 and avg<0.60):
        print("afternon")
    elif(avg<0.40):
        print("day")
    
    
    #img = np.float32(img) / 255.0
    """
    Rappel sur les Constantes de codage en opencv
    CV_8U is unsigned 8bit/pixel - ie a pixel can have values 0-255, this is the normal range for most image and video formats.
    CV_32F is float - the pixel can have any value between 0-1.0, this is useful for some sets of calculations on data - but it has to be converted into 8bits to save or display by multiplying each pixel by 255.
    CV_32S is a signed 32bit integer value for each pixel - again useful of you are doing integer maths on the pixels, but again needs converting into 8bits to save or display. This is trickier since you need to decide how to convert the much larger range of possible values (+/- 2billion!) into 0-255

    """
    # Calculate gradient 
    #gx = cv.Sobel(img, cv.CV_32F, 1, 0, ksize=1)
    
    #gy = cv.Sobel(img, cv.CV_32F, 0, 1, ksize=1)
    #Calculate
    #laplacian = cv.Laplacian(img,cv.CV_32F)
    
    
    cv.namedWindow('img') 
    cv.startWindowThread()
    # Display an image
    cv.imshow('img',thimg)
    cv.waitKey(0) 
    cv.destroyAllWindows()