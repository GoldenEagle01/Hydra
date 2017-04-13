from imutils.video import VideoStream
import datetime
import argparse
import imutils
import time
import cv2
import os 

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
face_cascade_p = cv2.CascadeClassifier('haarcascade_profileface.xml')

vs = VideoStream(src=0).start()
time.sleep(1.0)

while True:
	frame = vs.read()
        #frame = imutils.resize(frame, width=250)
 
        timestamp = datetime.datetime.now()
        ts = timestamp.strftime("%A %d %B %Y %I:%M:%S%p")
        cv2.putText(frame, ts, (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX,
              0.35, (0, 0, 255), 1)
 	ok = 0
	k = 1
        image_grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(image_grey, 1.2, 3)
        faces_p = face_cascade_p.detectMultiScale(image_grey,1.2, 3)
	for (x,y,w,h) in faces:
	 	mid_x = int(x + (w/2))
               	mid_y = int(y + (h/2))
                new_x = mid_x - 50
                new_y = mid_y - 50
		ok = 1
                if (ok == k):
			crop_img = image_grey[new_y:new_y+100, new_x:new_x+100]
	      		cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        for (x,y,w,h) in faces_p:
                 mid_x = int(x + (w/2))
                 mid_y = int(y + (h/2))
                 new_x = mid_x - 50
                 new_y = mid_y - 50 
		 k = 1
                 if (ok != k):	
			crop_img = image_grey[new_y:new_y+100, new_x:new_x+100]
                 	cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
 
        if key == ord("q"):
                 break
 
cv2.destroyAllWindows()
vs.stop()

