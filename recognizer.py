#!/usr/bin/env python2.6
import random
from PIL import Image
import subprocess
from imutils.video import VideoStream
import cv2
import numpy as np
import datetime
import argparse
import imutils
import time

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d","--dual", type=int,choices=[0,1], 
        help="whether or not the internal camera should be used")
args = ap.parse_args()

faceDetect = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
if (args.dual == 1):
	faceDetect1 = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
vs = VideoStream(src=0).start()
if (args.dual == 1):
	web = VideoStream(src=1).start()

recognizer = cv2.face.createLBPHFaceRecognizer()
if (args.dual == 1):
	recognizer1 = cv2.face.createLBPHFaceRecognizer()
recognizer.load("trainingData.yml")
if (args.dual == 1):
	recognizer1.load("trainingData.yml")

id = 0
if (args.dual == 1):
	id1 = 0
font = cv2.FONT_HERSHEY_COMPLEX
nfaces = 0
if (args.dual == 1):
	nfaces1 = 0
unknown = 0
if (args.dual == 1):
	unknown1 = 0
time.sleep(1.0)

while (True):
	frame = vs.read()
	if (args.dual == 1):
		frame1 = web.read()
	if (args.dual == 1):
		gray1 = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
	gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
	if (args.dual == 1):
		predict_image1 = np.array(gray1, 'uint8')
	predict_image = np.array(gray, 'uint8')
	faces=faceDetect.detectMultiScale(predict_image,1.3,5)
	if (args.dual == 1):
		faces1=faceDetect1.detectMultiScale(predict_image1,1.3,5)
	for (x,y,w,h) in faces:
		cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
		id, conf = recognizer.predict(predict_image[y:y+h,x:x+w])
		if (id == 1):	
			id = str("Andrei")
		if id == 22:
			id = str("Rares")
		if id == 3:
			id = str("Carmen")
		if id == 4:
			id = str("Ionut")
		print conf
		if ((conf < 40) or ((conf > 40 and conf <50) and (w<200 and w>150))) or (((conf > 50 and conf < 70) and (w>100 and w<150)) or ((conf > 70 and conf<80) and (w>50 and w<100))):
			nfaces = nfaces + 1
			if (nfaces > 5):
				cv2.putText(frame,str(w)+" "+str(h),(x+w,y), font, 1,(255,255,255),1,1)
				cv2.putText(frame,str(id),(x,y), font, 1,(255,255,255),1,1)	
				cv2.putText(frame,str(conf),(x+w,y+h), font, 1,(255,255,255),1,1)		
		else:
			unknown = unknown + 1
			if unknown > 5:
				cv2.putText(frame,str(w)+" "+str(h),(x+w,y), font, 1,(255,255,255),1,1)
				cv2.putText(frame,'Unkown',(x,y), font, 1,(255,255,255),1,1)
				cv2.putText(frame,str(conf),(x+w,y+h), font, 1,(255,255,255),1,1)
	cv2.moveWindow('face', 80, 50)
	cv2.imshow("face",frame)
	if (args.dual == 1):
		for (x,y,w,h) in faces1:
       		        cv2.rectangle(frame1,(x,y),(x+w,y+h),(255,0,0),2)
       		        id1, conf1 = recognizer1.predict(predict_image1[y:y+h,x:x+w])
       		        if (id1 == 1):
				id1 = str("Andrei")
  		        print conf1
			if ((conf1 < 40) or ((conf1 > 40 and conf1 <50) and (w<200 and w>150))) or (((conf1 > 50 and conf1 < 70) and (w>100 and w<150)) or ((conf1 > 70 and conf1<80) and (w>50 and w<100))):
        	               	nfaces1 = nfaces1 + 1
				if (nfaces1 > 5):
					cv2.putText(frame1,str(w)+" "+str(h),(x+w,y), font, 1,(255,255,255),1,1)	
					cv2.putText(frame1,str(conf1),(x+w,y+h), font, 1,(255,255,255),1,1)
        		   	        cv2.putText(frame1,str(id1),(x,y), font, 1,(255,255,255),1,1)
               		else:
				unknown1 = unknown1 + 1
                	       	if (unknown1 > 5):
					cv2.putText(frame1,str(w)+" "+str(h),(x+w,y), font, 1,(255,255,255),1,1)
					cv2.putText(frame1,str(conf1),(x+w,y+h), font, 1,(255,255,255),1,1)
                	       	        cv2.putText(frame1,'Unkown',(x,y), font, 1,(255,255,255),1,1)
		frame1 = imutils.resize(frame1,800)
		cv2.imshow("face1",frame1)
	if (cv2.waitKey(1) == ord('q')):
		break
vs.stop()
if (args.dual == 1):
	web.stop()
cv2.destroyAllWindows() 

