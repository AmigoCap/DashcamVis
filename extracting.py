#programme pour avoir un visage
import cv2
import numpy as np
from matplotlib import pyplot as plt

if __name__=='__main__':

        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
        cv2.namedWindow("img", cv2.WINDOW_NORMAL)
        img = cv2.imread('0001.jpg')
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x,y,w,h) in faces:
                cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
                roi_gray = gray[y:y+h, x:x+w]
                roi_color = img[y:y+h, x:x+w]
                eyes = eye_cascade.detectMultiScale(roi_gray)
                for (ex,ey,ew,eh) in eyes:
                        cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
        imS = cv2.resize(img, (960, 540))
        cv2.imshow('img',imS)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        """
        img = cv2.imread('0001.jpg',0)

        # global thresholding
        ret1,th1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)

        # Otsu's thresholding
        ret2,th2 = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

        # Otsu's thresholding after Gaussian filtering
        blur = cv2.GaussianBlur(img,(5,5),0)
        ret3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

        # plot all the images and their histograms
        images = [img, 0, th1,
                  img, 0, th2,
                  blur, 0, th3]
        titles = ['Original Noisy Image','Histogram','Global Thresholding (v=127)',
                  'Original Noisy Image','Histogram',"Otsu's Thresholding",
                  'Gaussian filtered Image','Histogram',"Otsu's Thresholding"]

        for i in xrange(3):
            plt.subplot(3,3,i*3+1),plt.imshow(images[i*3],'gray')
            plt.title(titles[i*3]), plt.xticks([]), plt.yticks([])
            plt.subplot(3,3,i*3+2),plt.hist(images[i*3].ravel(),256)
            plt.title(titles[i*3+1]), plt.xticks([]), plt.yticks([])
            plt.subplot(3,3,i*3+3),plt.imshow(images[i*3+2],'gray')
            plt.title(titles[i*3+2]), plt.xticks([]), plt.yticks([])
        plt.show()
        """
