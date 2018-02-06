# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 10:14:26 2018

@author: Achraf Baiz
"""

import cv2
import sys

# Get user supplied values
imagePath = "frenchteam.jpg"
cascPath = "haarcascade_frontalface_default.xml"

# Create the haar cascade
faceCascade = cv2.CascadeClassifier(cascPath)

# Read the image
image = cv2.imread(imagePath)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces in the image
faces = faceCascade.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(30, 30),
    flags = cv2.CASCADE_SCALE_IMAGE    
)

print("Found {0} faces!".format(len(faces)))

# Draw a rectangle around the faces
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

cv2.namedWindow('img') 
cv2.startWindowThread()
cv2.imshow("Faces found", image)
cv2.waitKey(0)
cv2.destroyAllWindows()