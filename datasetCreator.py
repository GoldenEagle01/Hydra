from imutils.video import VideoStream
import cv2
import numpy as np
import datetime
import argparse
import imutils
import time

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--picamera", type=int, default=-1,
        help="whether or not the Raspberry Pi camera should be used")
args = vars(ap.parse_args())

faceDetect = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
vs = VideoStream(usePiCamera=args["picamera"] > 0).start()
time.sleep(1)

id = input("enter user id= ")
sampleNr = 0

while (True):
	frame = vs.read()
	frame = imutils.resize(frame, width=250)
	gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
	faces = faceDetect.detectMultiScale(gray,1.3,5)
	for (x,y,w,h) in faces:
		sampleNr = sampleNr + 1
		cv2.imwrite("dataSet/User."+str(id)+"."+str(sampleNr)+".jpg",gray[y:y+h,x:x+w])
		cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
		cv2.waitKey(100)
	cv2.imshow("face",frame)
	cv2.waitKey(1)
	if (sampleNr > 20):
		break
vs.stop()
cv2.destroyAllWindows()
