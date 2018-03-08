# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 19:44:45 2018

@author: Achraf Baiz
"""

import cv2 as cv
import numpy as np
import matplotlib.patches as mpatches
from matplotlib import pyplot as plt
import os

def weather(weath):
	if(weath=="day"):
		return cv.imread("Sun-icon.png",3)
	if(weath=="afternon/earlymorning"):
		return cv.imread("Clouds-icon.png",3)
	if(weath=="night"):
		return cv.imread("Moon-icon.png",3)

def blend_offsets(img1,img2,x_offset,y_offset):
	#return img1[y_offset:y_offset+img2.shape[0], x_offset:x_offset+img2.shape[1]] = blend_transparent(img1[y_offset:y_offset+img2.shape[0], x_offset:x_offset+img2.shape[1],::],img2)
    img1[y_offset:y_offset+img2.shape[0], x_offset:x_offset+img2.shape[1]] = img2
    return  img1

def blend_transparent(face_img, overlay_t_img):
    # Split out the transparency mask from the colour info
    overlay_img = overlay_t_img[:,:,:3] # Grab the BRG planes
    overlay_mask = overlay_t_img[:,:,3:]  # And the alpha plane

    # Again calculate the inverse mask
    background_mask = 255 - overlay_mask

    # Turn the masks into three channel, so we can use them as weights
    overlay_mask = cv.cvtColor(overlay_mask, cv.COLOR_GRAY2BGR)
    background_mask = cv.cvtColor(background_mask, cv.COLOR_GRAY2BGR)

    # Create a masked out face image, and masked out overlay
    # We convert the images to floating point in range 0.0 - 1.0
    face_part = (face_img * (1 / 255.0)) * (background_mask * (1 / 255.0))
    overlay_part = (overlay_img * (1 / 255.0)) * (overlay_mask * (1 / 255.0))

    # And finally just add them together, and rescale it back to an 8bit integer image    
    return np.uint8(cv.addWeighted(face_part, 255.0, overlay_part, 255.0, 0.0))

def fig2data ( fig ):
    """
    @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGBA values
    """
    # draw the renderer
    fig.canvas.draw ( )
 
    # Get the RGBA buffer from the figure
    w,h = fig.canvas.get_width_height()
    buf = np.fromstring ( fig.canvas.tostring_argb(), dtype=np.uint8 )
    buf.shape = ( w, h,4 )
 
    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll ( buf, 3, axis = 2 )

    bufbgr = cv.cvtColor(buf,cv.COLOR_RGB2BGR)

    return bufbgr

def reconstitue(listing):
	j=0
	for i in listing:
		img1 = cv.imread('output_presentation/frame%d.jpg'%j)
		img2 = cv.imread('informations_output_presentation/image_%d.png'%j)
		x_offset=y_offset=50
		#imgf = face_recognection(img1)
		img1[y_offset:y_offset+img2.shape[0], x_offset:x_offset+img2.shape[1]] = img2
		cv.imwrite('final_presentation/out_%d.png'%j, img1)
		j+=1
        
def face_recognection(img):
    #xml face shape recognition path
    cascPath = "haarcascade_frontalface_default.xml"
    # Create the haar cascade
    faceCascade = cv.CascadeClassifier(cascPath)
    
    # Read the image
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    
    # Detect faces in the image
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags = cv.CASCADE_SCALE_IMAGE    
    )
    
    print("Found {0} faces!".format(len(faces)))
    
    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    return img,len(faces)

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
        plt.savefig('informations_output_presentation/image_%d.png'%_id)
        plt.gcf().clear()

def daylight(img):
    white_pix = 0
    black_pix = 0
    gray_image = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    _,thimg    = cv.threshold(img,127,255,cv.THRESH_BINARY)
    for i in range(gray_image.shape[0]):
        for j in range(gray_image.shape[1]):
            if gray_image.item(i,j)==0:
                black_pix=black_pix+1
            else:
                white_pix=white_pix+1
    avg = black_pix/(white_pix+black_pix)
    if(avg>0.70):
        return "night"
    elif(avg>0.40 and avg<0.60):
        return "afternon/earlymorning"
    elif(avg<0.40):
        return "day"

def timeline(array_images):
    x=[]
    y=[]
    Y=[]
    color = ('b','g','r')
   
    #boucle sur les frames pour retirer des informations
    for i in range(len(array_images)):
        figure1 = plt.figure()
        plot1    = figure1.add_subplot (111) 
        weth        = daylight(array_images[i])
        img_weather = weather(weth)
        for j,col in enumerate(color):
            histr = cv.calcHist([array_images[i]],[j],None,[256],[0,256])
            plot1.plot(histr,color = col)
            #plot1.xlim([0,256])
            #plot1.ylim([0,25100])
        histoimg = fig2data(figure1)
        array_images[i],nf= face_recognection(array_images[i])
        array_images[i]   = blend_offsets(array_images[i], histoimg, 50, 50)
        array_images[i]   = blend_offsets(array_images[i], img_weather, 200, 200)
        x.append(i)
        y.append(nf)
        #figure1.clf()
	#lissage du nombre de visage
    for i in range(len(array_images)):
        if i not in [0,1,len(array_images)-1,len(array_images)-2]:
            Y.append(int((y[i]+y[i-1]+y[i-2]+y[i+1]+y[i+2])/5))
        else:
            Y.append(y[i])

    for i in range(len(array_images)):
        figure2 = plt.figure()
        plot2   = figure2.add_subplot (111) 
        plot2.plot(x,Y,'b-')
        plot2.plot([x[i]],[Y[i]],'ro')
        timelineimg = fig2data(figure2)
        array_images[i] = blend_offsets(array_images[i], timelineimg, 300, 300)
        #figure2.clf()
        cv.imwrite('final_presentation/out_%d.png'%i, array_images[i])
    
    # Read the image
    

        
def video_images(path):
    assert type(path)==str
    vidcap = cv.VideoCapture(path)
    success,image = vidcap.read()
    count = 0
    success = True
    while success:
      success,image = vidcap.read()
      print('Read a new frame: ', success)
      cv.imwrite("output_presentation/frame_%d.jpg" % count, image)     # save frame as JPEG file
      count += 1
      if count==10:
          break        

if __name__=='__main__': 
    #video_images("accident.mp4")
    file_batch= sorted(os.listdir("output_presentation"),key=lambda x : int(x.replace(".","_").split("_")[1]))
    img_batch = list(map(lambda x: cv.imread("output_presentation/"+x,3),file_batch))
    timeline(img_batch[:10])

    #listing=os.listdir("informations_output_presentation")
    '''
    for i in range(560,1254): # 100 images
        img   = cv.imread(('output_presentation/frame%d.jpg'%i),1)
        histogramme(img,3,i)
        daylight(img)
    '''
    #reconstitue(listing)
    
    
    '''
    cv.namedWindow('img') 
    cv.startWindowThread()
    # Display an image
    cv.imshow('img',img)
    cv.waitKey(0) 
    cv.destroyAllWindows()
    '''