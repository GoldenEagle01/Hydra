# import the necessary packages
from imutils.video import VideoStream
import datetime
import argparse
import imutils
import time
import cv2
import os 
from PIL import Image

def get_images_and_labels(path):
        image_paths = [os.path.join(path, f) for f in os.listdir(path)]
        images = []
        labels = []
        for image_path in image_paths:
                image_pil = Image.open(image_path).convert('L')
                image = np.array(image_pil, 'uint8')
                nbr = int(os.path.split(image_path)[1].split(".")[0].replace("subject", ""))
                faces = faceCascade.detectMultiScale(image)
                for (x, y, w, h) in faces:
                        images.append(image[y: y + h, x: x + w])
                        labels.append(nbr)
                        cv2.imshow("Adding faces to traning set...", image[y: y + h, x: x + w])
                        cv2.waitKey(50)
        return images, labels

cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)
recognizer = cv2.face.createLBPHFaceRecognizer()

path = '/home/pi/Desktop/motion' #absolute path

images, labels = get_images_and_labels(path)
cv2.destroyAllWindows()

recognizer.train(images, np.array(labels))

detected = 0
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--picamera", type=int, default=-1,
        help="whether or not the Raspberry Pi camera should be used")
args = vars(ap.parse_args())
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#face_cascade_p = cv2.CascadeClassifier('haarcascade_profileface.xml')

# initialize the video stream and allow the cammera sensor to warmup
vs = VideoStream(usePiCamera=args["picamera"] > 0).start()
time.sleep(2.0)

# loop over the frames from the video stream
while True:
	# grab the frame from the threaded video stream and resize it
	# to have a maximum width of 300 pixels
        frame = vs.read()
        frame = imutils.resize(frame, width=250)
 
	# draw the timestamp on the frame
        timestamp = datetime.datetime.now()
        ts = timestamp.strftime("%A %d %B %Y %I:%M:%S%p")
        cv2.putText(frame, ts, (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX,
              0.35, (0, 0, 255), 1)
 
	# show the frame
        image_grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        predict_image = np.array(image_grey, 'uint8')
        faces = face_cascade.detectMultiScale(predict_image, 1.1, 3) 
        for (x, y, w, h) in faces:
             nbr_predicted, conf = recognizer.predict(predict_image[y: y + h, x: x + w])

       # faces_p = face_cascade_p.detectMultiScale(image_grey,1.1, 3)
       # for (x,y,w,h) in faces:
       #          mid_x = int(x + (w/2))
       #          mid_y = int(y + (h/2))
       #          new_x = mid_x - 50
       #          new_y = mid_y - 50
       #          crop_img = image_grey[new_y:new_y+100, new_x:new_x+100]
       #          cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
                 #cv2.imshow("Frame", image_greycrop_img)

        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
 
	# if the `q` key was pressed, break from the loop
        if key == ord("q"):
                 break

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
