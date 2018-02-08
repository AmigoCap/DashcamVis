# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 19:44:45 2018

@author: Achraf Baiz
"""

import cv2 as cv
import numpy as np
import matplotlib.patches as mpatches
from matplotlib import pyplot as plt


def reconstitue():
    for i in range(300,501):
        img1 = cv.imread('output/frame%d.jpg'%i)
        img2 = cv.imread('informations_output/image_%d.png'%i)
        x_offset=y_offset=50
        img1[y_offset:y_offset+img2.shape[0], x_offset:x_offset+img2.shape[1]] = img2
        cv.imwrite('final/out_%d.png'%i, img1)

def histogramme(img,cannaux,_id):
    if cannaux==3:
        color = ('b','g','r')
        for i,col in enumerate(color):
            histr = cv.calcHist([img],[i],None,[256],[0,256])
            plt.plot(histr,color = col)
            plt.xlim([0,256])
            plt.ylim([0,25100])
        #plt.show()
        '''
        red_patch = mpatches.Patch(color='red', label='red intensity')
        blue_patch = mpatches.Patch(color='blue', label='blue intensity')
        green_patch = mpatches.Patch(color='green', label='green intensity')
        plt.legend(handles=[red_patch,blue_patch,green_patch])
        '''
        plt.savefig('informations_output/image_%d.png'%_id)
        plt.gcf().clear()
def daylight(img):
    white_pix = 0
    black_pix = 0
    gray_image = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    _,thimg = cv.threshold(img,127,255,cv.THRESH_BINARY)
    for i in range(gray_image.shape[0]):
        for j in range(gray_image.shape[1]):
            if gray_image.item(i,j)==0:
                black_pix=black_pix+1
            else:
                white_pix=white_pix+1
    avg = black_pix/(white_pix+black_pix)
    if(avg>0.70):
        print("night")
    elif(avg>0.40 and avg<0.60):
        print("afternon/early morning")
    elif(avg<0.40):
        print("day")

if __name__=='__main__': 
    
    for i in range(300,501): # 100 images
        img   = cv.imread(('output/frame%d.jpg'%i),1)
        histogramme(img,3,i)
        daylight(img)
    
    reconstitue()
    
    '''
    cv.namedWindow('img') 
    cv.startWindowThread()
    # Display an image
    cv.imshow('img',img)
    cv.waitKey(0) 
    cv.destroyAllWindows()
    '''