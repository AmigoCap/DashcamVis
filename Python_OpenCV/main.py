# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 19:44:45 2018

@author: Achraf Baiz
"""
from __future__ import unicode_literals
import argparse
import subprocess
import numpy as np
import matplotlib.patches as mpatches
from matplotlib import pyplot as plt
import os
import cv2
import youtube_dl




plt.ioff()

def weather(weath):
	if(weath=="day"):
		return cv2.imread("Sun-icon.png",cv2.IMREAD_UNCHANGED)
	if(weath=="afternon/earlymorning"):
		return cv2.imread("Clouds-icon.png",cv2.IMREAD_UNCHANGED)
	if(weath=="night"):
		return cv2.imread("Moon-icon.png",cv2.IMREAD_UNCHANGED)

def blend_offsets(img1,img2,x_offset,y_offset,resize_img=1):
	#return img1[y_offset:y_offset+img2.shape[0], x_offset:x_offset+img2.shape[1]] =
    w_local=img1.shape[1]
    h_local=img1.shape[0]
    w_mask=img2.shape[1]
    h_mask=img2.shape[0]
    if (w_mask+x_offset*w_local>w_local ):
        coeffx=(1-x_offset)*w_local/(w_mask+10)
    else:
        coeffx=1
    if(h_mask+y_offset*h_local>h_local):
        coeffy=(1-y_offset)*h_local/(h_mask+10)
    else:
        coeffy=1
    coeff=min(coeffx,coeffy)
    img2=cv2.resize(img2, (0,0), fx=coeff*resize_img ,fy=coeff*resize_img) 
    y_off_pix=int(img1.shape[0]*y_offset)
    x_off_pix=int(img1.shape[1]*x_offset)
    img1[y_off_pix:y_off_pix+img2.shape[0], x_off_pix:x_off_pix+img2.shape[1],::]=blend_transparent(img1[y_off_pix:y_off_pix+img2.shape[0], x_off_pix:x_off_pix+img2.shape[1],::],img2)

    #img1[y_offset:y_offset+img2.shape[0], x_offset:x_offset+img2.shape[1]] = img2
    return  img1

def blend_transparent(face_img, overlay_t_img):
    # Split out the transparency mask from the colour info
    overlay_img = overlay_t_img[:,:,:3] # Grab the BRG planes
    overlay_mask = overlay_t_img[:,:,3:]  # And the alpha plane

    # Again calculate the inverse mask
    background_mask = 255 - overlay_mask

    # Turn the masks into three channel, so we can use them as weights
    overlay_mask = cv2.cvtColor(overlay_mask, cv2.COLOR_GRAY2BGR)
    background_mask = cv2.cvtColor(background_mask, cv2.COLOR_GRAY2BGR)

    # Create a masked out face image, and masked out overlay
    # We convert the images to floating point in range 0.0 - 1.0
    face_part = (face_img * (1 / 255.0)) * (background_mask * (1 / 255.0))
    overlay_part = (overlay_img * (1 / 255.0)) * (overlay_mask * (1 / 255.0))

    # And finally just add them together, and rescale it back to an 8bit integer image    
    return np.uint8(cv2.addWeighted(face_part, 255.0, overlay_part, 255.0, 0.0))

def fig2data ( fig ,transparency=False):
    """
    @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGBA values
    """
    fig.savefig("./fig.png",transparent=transparency)
    # draw the renderer
#    fig.canvas.draw ( )
# 
#    # Get the RGBA buffer from the figure
#    w,h = fig.canvas.get_width_height()
#    buf = np.fromstring ( fig.canvas.tostring_argb(), dtype=np.uint8 )
#    buf.shape = ( w, h,4 )
#    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
#    buf = np.roll ( buf, 3, axis = 2 )
#
#    bufbgr = cv.cvtColor(buf,cv.COLOR_RGB2BGR)
    
    imgoutput=cv2.imread("fig.png",cv2.IMREAD_UNCHANGED)
    return imgoutput

def reconstitue(listing):
	j=0
	for i in listing:
		img1 = cv2.imread('output_presentation/frame_%d.jpg'%j)
		img2 = cv2.imread('informations_output_presentation/image_%d.png'%j,cv2.IMREAD_UNCHANGED)
		x_offset=y_offset=50
		#imgf = face_recognection(img1)
		img1[y_offset:y_offset+img2.shape[0], x_offset:x_offset+img2.shape[1]] = img2
		cv2.imwrite('final_presentation/out_%d.png'%j, img1)
		j+=1
        
def face_recognection(img):
    #xml face shape recognition path
    cascPath = "haarcascade_frontalface_default.xml"
    # Create the haar cascade
    faceCascade = cv2.CascadeClassifier(cascPath)
    
    # Read the image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the image
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags = cv2.CASCADE_SCALE_IMAGE    
    )
    
    #print("Found {0} faces!".format(len(faces)))
    
    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    return img,len(faces)

def histogramme(img,cannaux,_id):
    if cannaux==3:
        color = ('b','g','r')
        for i,col in enumerate(color):
            histr = cv2.calcHist([img],[i],None,[256],[0,256])
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
        plt.savefig('informations_output_presentation/image_%d.png'%_id,bbox_inches='tight')
        #plt.gcf().clear()

def daylight(img):
    white_pix = 0
    black_pix = 0
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _,thimg    = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
    for i in range(gray_image.shape[0]):
        for j in range(gray_image.shape[1]):
            if gray_image.item(i,j)==0:
                black_pix=black_pix+1
            else:
                white_pix=white_pix+1
    avg = black_pix/(white_pix+black_pix)
    if(avg>0.50):
        return "night"
    elif(avg>0.30 and avg<=0.50):
        return "afternon/earlymorning"
    elif(avg<=0.30):
        return "day"

def timeline(array_images):
    x=[]
    y=[]
    Y=[]
    color = ('b','g','r')
   
    #boucle sur les frames pour retirer des informations
    for i in range(len(array_images)):
        local_image = cv2.imread(array_images[i],cv2.IMREAD_UNCHANGED)
        figure1     = plt.figure()
        plot1       = figure1.add_subplot (111)
        weth        = daylight(local_image)
        img_weather = weather(weth)
        for j,col in enumerate(color):
            histr = cv2.calcHist([local_image],[j],None,[256],[0,256])
            plot1.plot(histr,color = col)
            plot1.set_xlim([0,256])
            plot1.set_ylim([0,26000])
            plt.close(figure1)
        histoimg = fig2data(figure1,transparency=False)
        local_image,nf= face_recognection(local_image)
        local_image   = blend_offsets(local_image, histoimg, 0.1, 0.1,0.5)
        local_image   = blend_offsets(local_image, img_weather, 0.5,0.7)
        cv2.imwrite('informations_output_presentation/out_%d.png'%i, local_image)
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
        local_image = cv2.imread('informations_output_presentation/out_%d.png'%i,cv2.IMREAD_UNCHANGED)
        figure2 = plt.figure()
        plot2   = figure2.add_subplot (111) 
        plot2.plot(x,Y,'b-')
        plot2.plot([x[i]],[Y[i]],'ro')
        plt.close(figure2)
        timelineimg = fig2data(figure2)
        local_image = blend_offsets(local_image, timelineimg, 0.3, 0.8,0.5)
        #figure2.clf()
        cv2.imwrite('final_presentation/out_%d.png'%i, local_image)
    
    # Read the image
    

        
def video_images(path,maxframes=1200):
    vidcap = cv2.VideoCapture(path)
    success,image = vidcap.read()
    count = 0
    success = True
    while success:
      success,image = vidcap.read()
      print('Read a new frame: ', success)
      cv2.imwrite("output_presentation/frame_%d.jpg" % count, image)     # save frame as JPEG file
      count += 1
      if count==maxframes:
          break        

def images_video(array_images,framerate=24): # ------------------> A modifier
    local_image = cv2.imread("final_presentation/"+array_images[0],cv2.IMREAD_UNCHANGED)
    #convert from RGB of PIL to BGR of OpenCV
    frameH, frameW, channels = local_image.shape
    fourcc =  cv2.VideoWriter_fourcc(*'MJPG')
    #fourcc  = cv.VideoWriter_fourcc(*)
    video = cv2.VideoWriter("output.avi", fourcc, framerate, (frameW,frameH), 1)
    
    for i in range(len(array_images)):
        video.write( cv2.imread("final_presentation/"+array_images[i],cv2.IMREAD_UNCHANGED))
    video.release()      
def upload_video_youtube():
    request="""python upload_youtube.py --file="output.avi" --title="Dashcamvideo" --description="No description" --keywords="dashcam" --category="22" --privacyStatus="unlisted" """
    subprocess.call(request,shell=True)

def create_parser():
    parser=argparse.ArgumentParser(description='Image processer')
    parser.add_argument("--link", required=True, help="Link video to download")
    # nombre de frames à traiter
    parser.add_argument('--maxframes',metavar='frames', type=int, help='Maximum Number of frames to use in video processing')
    #framerate à utiliser lors de la reconstitution de la video
    parser.add_argument('--rate', metavar='framerate' , type=int, help='define frame rate of the reconstitued video (24 by default)')
    #position des graphes [gauche ou à droit]
    parser.add_argument('--pos', metavar='position' , type=str, help='define position')
    #configuration directe à partir d'un fichier configuration
    parser.add_argument('--json', metavar='True or False' , type=str, help='the json file of config')
    return parser
    
def create_folders(path="."):
    if(os.path.isdir(path+"/output_presentation/") is not True):
        os.makedirs(path+"/output_presentation/")
    if(os.path.isdir(path+"/informations_output_presentation/") is not True):
        os.makedirs(path+"/informations_output_presentation/")
    if(os.path.isdir(path+"/final_presentation/") is not True):
        os.makedirs(path+"/final_presentation/")

def delete_folders(f1,f2,f3):
    for file in f1+f2+f3:
        os.remove(file)


if __name__=='__main__': 
    parser=create_parser()
    args = parser.parse_args()

    ydl_opts = {'outtmpl':'output_vid'}
    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        ydl.download([args.link])
    args = parser.parse_args()
    create_folders()
    if(args.link==None):
        exit()
    elif(args.maxframes!=None and args.rate!=None):
            video_images(ydl_opts["outtmpl"],args.maxframes)
            out_batch  = sorted(os.listdir("informations_output_presentation"),key=lambda x : int(x.replace(".","_").split("_")[1]))
            final_img_batch = list(map(lambda x:"informations_output_presentation/"+x,out_batch))
            file_batch = sorted(os.listdir("output_presentation"),key=lambda x : int(x.replace(".","_").split("_")[1]))
            img_batch = list(map(lambda x:"output_presentation/"+x,file_batch))
            timeline(img_batch[:args.maxframes])
            file_batch_final = sorted(os.listdir("final_presentation"),key=lambda x : int(x.replace(".","_").split("_")[1]))

            images_video(file_batch_final,args.rate)
            file_batch_final=list(map(lambda x:"final_presentation/"+x, file_batch_final))
            delete_folders(img_batch,final_img_batch,file_batch_final)
            upload_video_youtube() 
    elif(args.maxframes==None and args.rate!=None):
            video_images(ydl_opts["outtmpl"])
            out_batch  = sorted(os.listdir("informations_output_presentation"),key=lambda x : int(x.replace(".","_").split("_")[1]))
            final_img_batch = list(map(lambda x:"informations_output_presentation/"+x,out_batch))
            file_batch = sorted(os.listdir("output_presentation"),key=lambda x : int(x.replace(".","_").split("_")[1]))
            img_batch = list(map(lambda x:"output_presentation/"+x,file_batch))
            timeline(img_batch)
            file_batch_final = sorted(os.listdir("final_presentation/"),key=lambda x : int(x.replace(".","_").split("_")[1]))
            images_video(file_batch_final,args.rate)
            file_batch_final=list(map(lambda x:"final_presentation/"+x, file_batch_final))
            delete_folders(img_batch,final_img_batch,file_batch_final)
            upload_video_youtube() 
    elif(args.maxframes!=None and args.rate==None):
            video_images(ydl_opts["outtmpl"],args.maxframes)
            out_batch  = sorted(os.listdir("informations_output_presentation"),key=lambda x : int(x.replace(".","_").split("_")[1]))
            final_img_batch = list(map(lambda x:"informations_output_presentation/"+x,out_batch))
            file_batch = sorted(os.listdir("output_presentation"),key=lambda x : int(x.replace(".","_").split("_")[1]))
            img_batch = list(map(lambda x:"output_presentation/"+x,file_batch))
            timeline(img_batch[:args.maxframes])
            file_batch_final = sorted(os.listdir("final_presentation/"),key=lambda x : int(x.replace(".","_").split("_")[1]))
            images_video(file_batch_final,24)
            file_batch_final=list(map(lambda x:"final_presentation/"+x, file_batch_final))
            delete_folders(img_batch,final_img_batch,file_batch_final) 
            upload_video_youtube()
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