#!/usr/bin/env python
# -*- coding: utf-8 -*-

from picamera.array import PiRGBArray
from picamera import PiCamera
from threading import Thread
from datetime import timedelta
from imutils.video import VideoStream
import numpy as np
import argparse
import warnings
import datetime
import logging
import imutils
import json
import time
import cv2

import smtplib
import imaplib
import email
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders

#send_mail sends and email and attaches a file when required
def send_email(subj="Motion Detected!", filename="detection.jpg", body="Motion detected! @"):
    #prepare the email
    msg = MIMEMultipart()
    msg["From"]=FROMADDR
    msg["To"]=TOADDR
    msg["Subject"]=subj
    
    t_stamp = datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p")
    body = body +" "+ t_stamp
    msg.attach(MIMEText(body, "plain"))
    
    #attach the file
    if filename != None:
        attachment = open(filename, "rb")
        part = MIMEBase("application", "octet-stream")
        part.set_payload((attachment).read())
        encoders.encode_base64(part)
        part.add_header("Content-Disposition", "attachment; filename= %s" % filename)
        msg.attach(part)
    
    #send the email    
    server = smtplib.SMTP("smtp.gmail.com", 587)
    server.starttls()
    server.login(FROMADDR, SMTPPASS)
    text = msg.as_string()
    server.sendmail(FROMADDR, TOADDR, text)
    server.quit()

# log_setup create the logging file and adds the column titles
def log_setup(filename):
    logger.setLevel(logging.INFO)
    log_file = logging.FileHandler(filename, mode="w")
    log_format = logging.Formatter("%(asctime)s,[%(levelname)s],%(message)s", "%Y-%m-%d,%H:%M:%S")
    log_file.setFormatter(log_format)
    logger.addHandler(log_file)

# msg_out prints a formated message to the console    
def msg_out(typ = "I", msg = "null"):
    msg_time = datetime.datetime.now().strftime("%I:%M:%S%p")
    
    if typ == "I": mtype = "[INFO - "
    elif typ == "W": mtype = "[WARNING - "
    elif typ == "A": mtype = "[ALARM - "
    elif typ == "E": mtype = "[ERROR - "
    elif typ == "C": mtype = "[CMD - "
    else: mtype = "[UNKNOWN - "
        
# sys_check scan the system mailbox for remote commands
def sys_check():
    cmd = "NC"
    try:
        mbox = imaplib.IMAP4_SSL("imap.gmail.com", 993) #Mail box reader
        result = mbox.login(FROMADDR, SMTPPASS)
        mbox.select()
        
        emfilter= "(FROM \"{}\")".format(TOADDR)
        result, data = mbox.uid("search", None,emfilter)
        latest = data[0].split()[-1]
       
        result, data = mbox.uid("fetch", latest, "(RFC822)")
        raw_msg = data[0][1]
        
        emsg = email.message_from_bytes(raw_msg)
        
        mbox.uid("STORE", latest, "+FLAGS", "(\Deleted)")
        mbox.expunge()
        
        #Analyze command
        if emsg["subject"]=="reset" or emsg["subject"]=="Reset": cmd = "C1"
        elif emsg["subject"]=="home" or emsg["subject"]=="Home": cmd = "C2"
        elif emsg["subject"]=="away" or emsg["subject"]=="Away": cmd = "C3"

        else: cmd = "UC"   
    except:
        cmd = "NC"
    finally:
        try:
            mbox.close()
            mbox.logout()
        finally:
            return cmd

# get_cpu_time obtains the system uptime by reading/parsing the /proc/uptime file
def  get_cpu_uptime():
    uptime = ""
    with open("/proc/uptime", "r") as f:
        uptime_seconds = float(f.readline().split()[0])
        uptime = str(timedelta(seconds = uptime_seconds))
    return uptime
    
# get_cpu_load obtains the processor load by reading/parsing /proc/loadavg
def get_cpu_load():
    cpuload= open("/proc/loadavg","r").readline().split(" ")[:3]
    return cpuload 

# get_memory obtains the system available memory by reading/parsing the /proc/meminfo file    
def get_memory():
    meminfo = dict((i.split()[0].rstrip(":"), int(i.split()[1])) for i in open ("/proc/meminfo").readlines())
    return meminfo["MemAvailable"]

#accumulate the weighted average between the current frame and
    #previous frames, then compute the difference between the current
    #fram and running average

class MotionDetector:
        def __init__(self, accumWeight=0.5, deltaThresh=5, minArea=5000):
		# determine the OpenCV version, followed by storing the
		# the frame accumulation weight, the fixed threshold for
		# the delta image, and finally the minimum area required
		# for "motion" to be reported
                self.isv2 = imutils.is_cv2()
                self.accumWeight = accumWeight
                self.deltaThresh = deltaThresh
                self.minArea = minArea
 
		# initialize the average image for motion detection
                self.avg = None
        def update(self,image):
             locs = []
	# if the average image is None, initialize it
             if self.avg is None:
                 self.avg = image.astype("float")
                 return locs
             cv2.accumulateWeighted(gray, self.avg, self.accumWeight)
             frameDelta = cv2.absdiff(gray, cv2.convertScaleAbs(self.avg))

    #threshold the delta image, dilate the thresholded image to fill in holes, then find countours
    #on thresholded image
             thresh = cv2.threshold(frameDelta, self.deltaThresh, 255, cv2.THRESH_BINARY)[1]
             thresh = cv2.dilate(thresh, None, iterations=2)
# find contours in the thresholded image, taking care to
		# use the appropriate version of OpenCV
             cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
			cv2.CHAIN_APPROX_SIMPLE)
             cnts = cnts[0] if self.isv2 else cnts[1]
 
		# loop over the contours
             for c in cnts:
			# only add the contour to the locations list if it
			# exceeds the minimum area
                   if cv2.contourArea(c) > self.minArea:
                          locs.append(c)
		
		# return the set of locations
             return locs
    #clear stream in preparation for the next frame

#construct the command line argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--conf", required=True,
    help="usage: python3 pi_surveillance.py --conf [file.json]")
args = vars(ap.parse_args())
warnings.filterwarnings("ignore")
conf = json.load(open(args["conf"]))

#emailing parameters settings
FROMADDR = conf["fromaddr"]  #email account
SMTPPASS = conf["smtppass"]  #email password
TOADDR = conf["toaddr"]      #email recipient

#initialize CAMERA
msg_out("I","Initialize camera...")
lastUploaded = datetime.datetime.now()
lastsyscheck = datetime.datetime.now()
motionCounter = 0 
webcam = VideoStream(src=0).start()
camera = VideoStream(usePiCamera=True).start()
time.sleep(conf["camera_warmup_time"])

#if required, display delta and thresholded video    
if conf["ghost_video"]:
    cv2.namedWindow("Thresh", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Frame Delta", cv2.WINDOW_NORMAL)

#MAIN
camMotion = MotionDetector()
piMotion = MotionDetector()
total = 0

while True:
    frames = []
    text = "Empty"
    timestamp = datetime.datetime.now()

# loop over the frames and their respective motion detectors
    for (stream, motion) in zip((webcam, camera), (camMotion, piMotion)):
		# read the next frame from the video stream and resize
		# it to have a maximum width of 500 pixels
                frame = stream.read()
                frame = imutils.resize(frame, width=250)
 
		# convert the frame to grayscale, blur it slightly, update
		# the motion detector
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray = cv2.GaussianBlur(gray, (21, 21), 0)
                locs = motion.update(gray)
 
		# we should allow the motion detector to "run" for a bit
		# and accumulate a set of frames to form a nice average
                if total < 16:
                        frames.append(frame)
                        continue
# otherwise, check to see if motion was detected
                if len(locs) > 0:
			# initialize the minimum and maximum (x, y)-coordinates,
			# respectively
                        (minX, minY) = (np.inf, np.inf)
                        (maxX, maxY) = (-np.inf, -np.inf)
 
			# loop over the locations of motion and accumulate the
			# minimum and maximum locations of the bounding boxes
                        for l in locs:
                                (x, y, w, h) = cv2.boundingRect(l)
                                (minX, maxX) = (min(minX, x), max(maxX, x + w))
                                (minY, maxY) = (min(minY, y), max(maxY, y + h))
                  
			# draw the bounding box
			# occupied
                        cv2.rectangle(frame, (minX, minY), (maxX, maxY),(0, 0, 255), 2)
                        text = "Occupied"
                        if (datetime.datetime.now() - lastUploaded).seconds >= conf["min_upload_seconds"]:
                                     print ("send email")
                                     if conf["send_email"]:
                                        cv2.imwrite("detection.jpg", frame)
                                        msg_out("I", "Sending email...")
                                        Thread(target=send_email).start()
                                     lastUploaded = datetime.datetime.now()
                                #motionCounter = 0
     			#write status of screen   
                        cv2.putText(frame, "Status: {}".format(text),(10,20),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),1)
                frames.append(frame)	
# increment the total number of frames read and grab the 
    total += 1	
#    motionCounter = 0
    ts = timestamp.strftime("%A %d %B %Y %I:%M:%S%p")

 # loop over the frames a second time
    for (frame, name) in zip(frames, ("Webcam", "Picamera")):
                # draw the timestamp on the frame and display it
                cv2.putText(frame, ts, (10, frame.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
                if text != "Occupied":cv2.putText(frame, "Status: Empty",(10,20),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),1)
                cv2.imshow(name, frame)

 #validate if system check is required
    if conf["sys_check_seconds"]>0:
        if (timestamp - lastsyscheck).seconds >= conf["sys_check_seconds"]:
            msg_out("I", "System check...")
            lastsyscheck = timestamp
            cmd = sys_check()

            if cmd == "C1":     #reset log
                msg_out("C", "Reset log command received!")
                if conf["keep_log"]:
                    msg_out("I", "Deleting old log file")
                    os.remove(LOGNAME)
            elif cmd == "C10":      #home
                msg_out("C", "home command received!")
                conf["send_email"] = False
                conf["keep_log"] = False
                conf["alarm"] = False
                send_email("Home mode on", None, "Home mode activated")
            elif cmd == "C11":      #away
                msg_out("C", "away command received!")
                conf["send_email"] = True
                conf["alarm"] = True
                send_email("Away mode on", None, "Away mode activated")
	# check to see if a key was pressed
    key = cv2.waitKey(1) & 0xFF
 
	# if the `q` key was pressed, break from the loop
    if key == ord("q"):
          break
 
# do a bit of cleanup
print("[INFO] cleaning up...")
cv2.destroyAllWindows()
webcam.stop()
camera.stop()
