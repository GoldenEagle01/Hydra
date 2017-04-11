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
from math import sin, cos, radians

faceDetect = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
faceDetect1 = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
vs = VideoStream(src=0).start()
web = VideoStream(src=1).start()
time.sleep(1)

recognizer = cv2.face.createLBPHFaceRecognizer()
recognizer1 = cv2.face.createLBPHFaceRecognizer()
recognizer.load("trainingData.yml")
recognizer1.load("trainingData.yml")

id = 0
id1 = 0
font = cv2.FONT_HERSHEY_COMPLEX
nfaces = 0
nfaces1 = 0
unknown = 0
unknown1 = 0

while (True):
	frame = vs.read()
	#frame = imutils.resize(frame, width=500)
	frame1 = web.read()
	frame1 = imutils.resize(frame1, width=800)
	gray1 = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
	gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
	predict_image1 = np.array(gray1, 'uint8')
	predict_image = np.array(gray, 'uint8')
	faces1=faceDetect.detectMultiScale(predict_image1,1.2,5)
	faces=faceDetect.detectMultiScale(predict_image,1.2,5)
	for (x,y,w,h) in faces:
		cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
		id, conf = recognizer.predict(predict_image[y:y+h,x:x+w])
		if (id == 1):	
			id = str("Andrei")
		#result = cv2.face.MinDistancePredictCollector()
		#recognizer.predict(predict_image[y: y + h, x: x + w],result, 0)
		#id = result.getLabel()
		#conf = result.getDist()
		print conf 
		if (conf < 80) or (conf > 100 and conf < 110 and w < 55 and h <95) or (conf > 90 and conf <100 and w>100 and w<120) or (conf > 80 and conf <90 and conf  and w>120 and w< 150):
			nfaces = nfaces + 1
			if (nfaces > 5):
				cv2.putText(frame,str(w)+" "+str(h),(x+w,y), font, 1,(255,255,255),1,1)
				cv2.putText(frame,str(id),(x,y), font, 1,(255,255,255),1,1)	
				cv2.putText(frame,str(conf),(x+w,y+h), font, 1,(255,255,255),1,1)		
		else:
			unknown = unknown + 1
                        if (unknown > 5):
				cv2.putText(frame,str(w)+" "+str(h),(x+w,y), font, 1,(255,255,255),1,1)
                                cv2.putText(frame,'Unkown',(x,y), font, 1,(255,255,255),1,1)
				cv2.putText(frame,str(conf),(x+w,y+h), font, 1,(255,255,255),1,1)
			#if (nfaces > 5):
			#	cv2.putText(frame,str(w)+" "+str(h),(x+w,y), font, 1,(255,255,255),1,1)
			#	cv2.putText(frame,str(conf),(x+w,y+h), font, 1,(255,255,255),1,1)
			#	cv2.putText(frame,'Unkown',(x,y), font, 1,(255,255,255),1,1)
	cv2.imshow("face",frame)
	for (x,y,w,h) in faces1:
                cv2.rectangle(frame1,(x,y),(x+w,y+h),(255,0,0),2)
                id1, conf1 = recognizer1.predict(predict_image1[y:y+h,x:x+w])
                if (id1 == 1):
			id1 = str("Andrei")
		#result = cv2.face.MinDistancePredictCollector()
                #recognizer.predict(predict_image[y: y + h, x: x + w],result, 0)
                #id = result.getLabel()
                #conf = result.getDist()
                print conf1
                if (conf1 < 80) or (conf1 > 100 and conf1 < 110 and w > 55 and w < 95) or (conf1>90 and conf1 < 100 and w>100 and w<120) or (conf1>80 and conf1 < 90 and w>120 and w<190):
                        nfaces1 = nfaces1 + 1
			if (nfaces1 > 5):
				cv2.putText(frame1,str(w)+" "+str(h),(x+w,y), font, 1,(255,255,255),1,1)
				cv2.putText(frame1,str(conf1),(x+w,y+h), font, 1,(255,255,255),1,1)
                	        cv2.putText(frame1,str(id1),(x,y), font, 1,(255,255,255),1,1)
                else:
                        if (unknown1 > 5):
				cv2.putText(frame1,str(w)+" "+str(h),(x+w,y), font, 1,(255,255,255),1,1)
				cv2.putText(frame1,str(conf1),(x+w,y+h), font, 1,(255,255,255),1,1)
                                cv2.putText(frame1,'Unkown',(x,y), font, 1,(255,255,255),1,1)
			#if (nfaces1 > 5):
			#	cv2.putText(frame1,str(w)+" "+str(h),(x+w,y), font, 1,(255,255,255),1,1)
			#	cv2.putText(frame1,str(conf1),(x+w,y+h), font, 1,(255,255,255),1,1)
                        #       cv2.putText(frame1,'Unkown',(x,y), font, 1,(255,255,255),1,1)
	cv2.imshow("face1",frame1)
	if (cv2.waitKey(1) == ord('q')):
		break
vs.stop()
web.stop()
cv2.destroyAllWindows() 

