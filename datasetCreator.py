from imutils.video import VideoStream
import random
import cv2
import numpy as np
import argparse
import imutils
import time
import subprocess

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--picamera", type=int, default=-1,
        help="whether or not the Raspberry Pi camera should be used")
args = vars(ap.parse_args())

faceDetect = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
faceDetect1 = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
vs = VideoStream(src = 0).start()
web = VideoStream(src = 1).start()
time.sleep(1)

id =1 
#id = input("enter user id= ")
sampleNr = 0
sampleNr = random.random()

def getImagesWithID(path):
        imagePaths=[os.path.join(path,f) for f in os.listdir(path)]
        faces = []
        IDs = []
        for imagePath in imagePaths:
                faceImg = Image.open(imagePath).convert('L')
                faceNp = np.array(faceImg, 'uint8')
                ID = int(os.path.split(imagePath)[-1].split('.')[1])
                faces.append(faceNp)
                IDs.append(ID)
                cv2.imshow("training",faceNp)
                cv2.waitKey(10)
        return np.array(IDs), faces

while (True):
	frame = vs.read()
	#frame = imutils.resize(frame, width=400)
	frame1 = web.read()
	gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
	gray1 = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
        faces1 = faceDetect1.detectMultiScale(gray1,1.3,5)
	faces = faceDetect.detectMultiScale(gray,1.3,5)
	for (x,y,w,h) in faces:
		sampleNr = sampleNr + 1
		cv2.imwrite("dataSet/User."+str(id)+"."+str(sampleNr)+".jpg",gray[y:y+h,x:x+w])
		cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
		cv2.waitKey(100)
	cv2.imshow("face",frame)
	for (x,y,w,h) in faces1:
                sampleNr = sampleNr + 1
                cv2.imwrite("dataSet/User."+str(id)+"."+str(sampleNr)+".jpg",gray[y:y+h,x:x+w])
                cv2.rectangle(frame1,(x,y),(x+w,y+h),(255,0,0),2)
                cv2.waitKey(100)
        cv2.imshow("face",frame1)
	cv2.waitKey(1)
	if (sampleNr > 100):
		break
vs.stop()
cv2.destroyAllWindows()
