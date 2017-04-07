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

recognizer = cv2.face.createLBPHFaceRecognizer()
recognizer.load("trainingData.yml")

id = 0
font = cv2.FONT_HERSHEY_COMPLEX

while (True):
	frame = vs.read()
	frame = imutils.resize(frame, width=250)	
	gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
	predict_image = np.array(gray, 'uint8')
	faces = faceDetect.detectMultiScale(gray,1.3,5)
	for (x,y,w,h) in faces:
		cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
		id = recognizer.predict(predict_image[y:y+h,x:x+w])   
		cv2.putText(frame,str(id),(x,y), font, 1,(255,255,255),1,1)
	cv2.imshow("face",frame)
	if (cv2.waitKey(1) == ord('q')):
		break
vs.stop()
cv2.destroyAllWindows() 
