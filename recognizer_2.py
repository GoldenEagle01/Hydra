#!/usr/bin/env python2.6
import random
from scipy.spatial import distance as dist
from PIL import Image
import subprocess
from imutils.video import VideoStream
from imutils import face_utils
import cv2
import numpy as np
import datetime
import argparse
import dlib
import imutils
import time

def eye_aspect_ratio(eye):
        # compute the euclidean distances between the two sets of
        # vertical eye landmarks (x, y)-coordinates
        A = dist.euclidean(eye[1], eye[5])
        B = dist.euclidean(eye[2], eye[4])

        # compute the euclidean distance between the horizontal
        # eye landmark (x, y)-coordinates
        C = dist.euclidean(eye[0], eye[3])

        # compute the eye aspect ratio
        ear = (A + B) / (2.0 * C)

        # return the eye aspect ratio
        return ear

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d","--dual", type=int,choices=[0,1], 
        help="whether or not the internal camera should be used")
args = ap.parse_args()

EYE_AR_THRESH = 0.25
EYE_AR_CONSEC_FRAMES = 3

COUNTER = 0
TOTAL = 0

detector = dlib.get_frontal_face_detector()

predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
# face recognizer builder
faceDetect = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
# for second camera (can be removed, first object can be used)
if (args.dual == 1):
	faceDetect1 = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
# open default camera
videoStreamDefault = VideoStream(src=0).start()
if (args.dual == 1):
	# open web camera
	web = VideoStream(src=1).start()

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# create face recognizer
recognizer = cv2.face.createLBPHFaceRecognizer()
	
# Load recognizer
recognizer.load("trainingData.yml")

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
	# Take frame
	frame = videoStreamDefault.read()
	if (args.dual == 1):
		frame1 = web.read()
	
	# Convert frame to grayscale
	gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
	if (args.dual == 1):
		gray1 = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
	
	rects = detector(gray, 0)
	if (args.dual == 1):
		rects1 = detector(gray1, 0)
		
	predict_image = np.array(gray, 'uint8')	
	if (args.dual == 1):
		predict_image1 = np.array(gray1, 'uint8')

	# Retrive detected faces
	faces=faceDetect.detectMultiScale(predict_image,1.2,5)
	if (args.dual == 1):
		faces1=faceDetect1.detectMultiScale(predict_image1,1.2,5)
	
	# For blinks
	for rect in rects:
		shape = predictor(gray, rect)
                shape = face_utils.shape_to_np(shape)
		leftEye = shape[lStart:lEnd]
                rightEye = shape[rStart:rEnd]
                leftEAR = eye_aspect_ratio(leftEye)
                rightEAR = eye_aspect_ratio(rightEye)
		ear = (leftEAR + rightEAR) / 2.0
		leftEyeHull = cv2.convexHull(leftEye)
                rightEyeHull = cv2.convexHull(rightEye)
                cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
                cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
		if ear < EYE_AR_THRESH:
                        COUNTER += 1
		else:
                        if COUNTER >= EYE_AR_CONSEC_FRAMES:
                                TOTAL += 1
                        COUNTER = 0
                cv2.putText(frame, "Blinks: {}".format(TOTAL), (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
				
	for (x,y,w,h) in faces:
		# Insert rectangle for face detected
		x = x + int(0.1*w)
		w = w - int(0.2*w)
	
		cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
		# Take ID and confidance value
		face = predict_image[y:y+h,x:x+w]
		id, conf = recognizer.predict(face)
		if (id == 1):
                        id = 'Andrei'
                if (id == 3):
                        id = 'Ionut'
                if (id == 4):
                        id = 'Ioana D'
		print conf, id
		if ((conf < 40) or ((conf > 40 and conf <50) and (w<200 and w>150))) or (((conf > 50 and conf < 70) and (w>100 and w<150)) or ((conf > 70 and conf<80) and (w>50 and w<100))):
			#cv2.putText(frame,str(w)+" "+str(h),(x+w,y), font, 1,(255,255,255),1,1)
			cv2.putText(frame,str(id),(x,y), font, 1,(255,255,255),1,1)	
			cv2.putText(frame,str(int(conf)),(x+w,y+h), font, 1,(255,255,255),1,1)		
		else:
			#cv2.putText(frame,str(w)+" "+str(h),(x+w,y), font, 1,(255,255,255),1,1)
			cv2.putText(frame,'Unkown',(x,y), font, 1,(255,255,255),1,1)
			cv2.putText(frame,str(int(conf)),(x+w,y+h), font, 1,(255,255,255),1,1)
	cv2.moveWindow('face', 80, 50)
	cv2.imshow("face",frame)
	if (args.dual == 1):
		for rect in rects1:
 	                shape = predictor(gray, rect)
        	        shape = face_utils.shape_to_np(shape)
        	        leftEye = shape[lStart:lEnd]
        	        rightEye = shape[rStart:rEnd]
        	        leftEAR = eye_aspect_ratio(leftEye)
        	        rightEAR = eye_aspect_ratio(rightEye)
        	        ear = (leftEAR + rightEAR) / 2.0
        	        leftEyeHull = cv2.convexHull(leftEye)
        	        rightEyeHull = cv2.convexHull(rightEye)
        	        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        	        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
        	        if ear < EYE_AR_THRESH:
        	                COUNTER += 1
        	        else:
        	                if COUNTER >= EYE_AR_CONSEC_FRAMES:
        	                        TOTAL += 1
        	                COUNTER = 0
       		        if (TOTAL > 0):
				cv2.rectangle(frame1,(x,y),(x+w,y+h),(255,0,0),2)
       		        id1, conf1 = recognizer.predict(predict_image1[y:y+h,x:x+w])
       		        if (id1 == 1):
				id1 = str("Andrei")
  		        print conf1
			if ((conf1 < 40) or ((conf1 > 40 and conf1 <50) and (w<200 and w>150))) or (((conf1 > 50 and conf1 < 70) and (w>100 and w<150)) or ((conf1 > 70 and conf1<80) and (w>50 and w<100))):
        	               	nfaces1 = nfaces1 + 1
				if (nfaces1 > 5 and TOTAL > 0):
					cv2.putText(frame1,str(w)+" "+str(h),(x+w,y), font, 1,(255,255,255),1,1)	
					cv2.putText(frame1,str(conf1),(x+w,y+h), font, 1,(255,255,255),1,1)
        		   	        cv2.putText(frame1,str(id1),(x,y), font, 1,(255,255,255),1,1)
               		else:
				unknown1 = unknown1 + 1
                	       	if (unknown1 > 5 and TOTAL > 0):
					cv2.putText(frame1,str(w)+" "+str(h),(x+w,y), font, 1,(255,255,255),1,1)
					cv2.putText(frame1,str(conf1),(x+w,y+h), font, 1,(255,255,255),1,1)
                	       	        cv2.putText(frame1,'Unkown',(x,y), font, 1,(255,255,255),1,1)
		frame1 = imutils.resize(frame1,800)
		cv2.imshow("face1",frame1)
	if (cv2.waitKey(1) == ord('q')):
		break
		
# Close video stream
videoStreamDefault.stop()
# If web camera was opened, close it
if (args.dual == 1):
	web.stop()
	
cv2.destroyAllWindows() 

