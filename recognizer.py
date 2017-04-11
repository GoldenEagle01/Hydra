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

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--picamera", type=int, default=-1,
        help="whether or not the Raspberry Pi camera should be used")
args = vars(ap.parse_args())

faceDetect = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
vs = VideoStream(usePiCamera=args["picamera"] > 0).start()
time.sleep(1)

recognizer = cv2.face.createLBPHFaceRecognizer()
recognizer.load("trainingData.yml")

id = 0
font = cv2.FONT_HERSHEY_COMPLEX
nfaces = 0

while (True):
	frame = vs.read()
	frame = imutils.resize(frame, width=400)
	
	gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
	predict_image = np.array(gray, 'uint8')
	faces=faceDetect.detectMultiScale(predict_image,1.2,5)
	for (x,y,w,h) in faces:
		cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
		#id, conf = recognizer.predict(predict_image[y:y+h,x:x+w])
		result = cv2.face.MinDistancePredictCollector()
		recognizer.predict(predict_image[y: y + h, x: x + w],result, 0)
		id = result.getLabel()
		conf = result.getDist()
		print conf 
		if (conf < 100) or (conf < 120 and w < 55 and h <55):
			nfaces = nfaces + 1
			cv2.putText(frame,str(id),(x,y), font, 1,(255,255,255),1,1)			
		else:
			if (nfaces > 10):
				#print "##############"
				cv2.putText(frame,'Unkown',(x,y), font, 1,(255,255,255),1,1)
	cv2.imshow("face",frame)
	if (cv2.waitKey(1) == ord('q')):
		break
vs.stop()
cv2.destroyAllWindows() 

