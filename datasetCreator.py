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
#faceDetect1 = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
vs = VideoStream(src = 0).start()
#web = VideoStream(src = 1).start()
time.sleep(1)

id = input("enter user id= ")
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
	gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
	faces = faceDetect.detectMultiScale(gray,1.2,5)
	for (x,y,w,h) in faces:
		x = x + int(0.1*w)
		w = w - int(0.2*w)
		sampleNr = sampleNr + 1
		face = gray[y:y+h,x:x+w]
		cv2.imwrite("dataSet/User."+str(id)+"."+str(sampleNr)+".jpg",cv2.resize(face,(250,300)))
		cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
		cv2.waitKey(1)
		cv2.imshow("face",frame)
	if (sampleNr > 50):
		break
vs.stop()
cv2.destroyAllWindows()
