from imutils.video import VideoStream
import numpy as np
import argparse
import cv2
import imutils
import time

from directkeys import  W, A, S, D ,P
from directkeys import PressKey, ReleaseKey 

def nothing(x):
    pass
cv2.namedWindow('test')
cv2.resizeWindow('test',(560,560))
cv2.createTrackbar('HUE LOW','test',0,179,nothing)  # max value and initial pos 
cv2.createTrackbar('HUE HIGH','test',179,179,nothing)
cv2.createTrackbar('SATURATION LOW','test',0,255,nothing)
cv2.createTrackbar('SATURATION HIGH','test',255,255,nothing)
cv2.createTrackbar('VALUE LOW','test',0,255,nothing)
cv2.createTrackbar('VALUE HIGH','test',255,255,nothing)       
cv2.createTrackbar('KernalOpenSize','test',5,50,nothing)    
cv2.createTrackbar('KernalCloseSize','test',5,50,nothing)

prev = None
now  = None
vs = VideoStream(src=0).start()
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

time.sleep(2.0)
initial = True
flag = False
current_key_pressed = set()
circle_radius = 30
windowSize = 160
lr_counter = 0


while True:
    keyPressed = False
    keyPressed_lr = False
    # grab the current frame
    frame = vs.read()
    frame= cv2.flip(frame,1)
    show = frame.copy()
    half = cv2.resize(show, (0, 0), fx = 0.5, fy = 0.5) 
    cv2.imshow('testtt',half)
    height,width = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,     
        scaleFactor=1.2,
        minNeighbors=5,     
        minSize=(20, 20)
    )


    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x-50, y-50), (x+w+50, y+h+50), (0, 2),-1)



    frame = imutils.resize(frame, width=600)
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    kernalOpenSize= cv2.getTrackbarPos('KernalOpenSize','test')
    kernalCloseSize= cv2.getTrackbarPos('KernalCloseSize','test')
    kernalOpen =  np.ones((kernalOpenSize,kernalOpenSize))
    kernalClose =  np.ones((kernalCloseSize,kernalCloseSize))
        
    hl= cv2.getTrackbarPos('HUE LOW','test')
    sl= cv2.getTrackbarPos('SATURATION LOW','test')
    vl= cv2.getTrackbarPos('VALUE LOW','test')

    hh= cv2.getTrackbarPos('HUE HIGH','test')
    sh= cv2.getTrackbarPos('SATURATION HIGH','test')
    vh =cv2.getTrackbarPos('VALUE HIGH','test')
    threshold = cv2.getTrackbarPos('Threshold_below','test')
    tcovert = cv2.getTrackbarPos('threshold_To','test')
        
    lowerbound = np.array([hl,sl,vl])
    upperbound = np.array([hh,sh,vh])
    mask = cv2.inRange(hsv, lowerbound, upperbound)

    mask = cv2.erode(mask, None, iterations=2)

    mask = cv2.dilate(mask, None, iterations=2)
    cv2.imshow('mask3',mask)

    left_mask = mask[:,0:width//2,]
    right_mask = mask[:,width//2:,]

    cnts_left = cv2.findContours(left_mask.copy(), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)
    cnts_left = imutils.grab_contours(cnts_left)
    center_left = None

    cnts_right = cv2.findContours(right_mask.copy(), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)
    cnts_right = imutils.grab_contours(cnts_right)
    center_right = None
 

    if len(cnts_left) > 0:        

        c = max(cnts_left, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)

        center_left = (int(M["m10"] / (M["m00"]+0.000001)), int(M["m01"] / (M["m00"]+0.000001)))
    

        if radius > circle_radius:

            cv2.circle(frame, (int(x), int(y)), int(radius),
                (0, 255, 255), 2)
            cv2.circle(frame, center_left, 5, (0, 0, 255), -1)

           
            if center_left[1] < (height/2 - windowSize//2):
                if prev ==None:
                   prev = "LEFT" 
                else:
                    prev = now
                now = 'LEFT'
                if prev =='UP' and now =='LEFT':
                    print('swipe  LEFT')
                    PressKey(A)
                    ReleaseKey(A)
                cv2.putText(frame,'LEFT',(20,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255))

            elif center_left[1] > (height/2 + windowSize//2):
                if prev ==None:
                   prev = "RIGHT" 
                else:
                    prev = now
                now = 'RIGHT'
                if prev =='LEFT' and now =='RIGHT':
                    print('swipe down')
                    PressKey(D)
                    PressKey(D)
                    PressKey(D)
                    ReleaseKey(D)

                cv2.putText(frame,'RIGHT',(20,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255))
                
    

    if len(cnts_right) > 0:
        c2 = max(cnts_right, key=cv2.contourArea)
        ((x2, y2), radius2) = cv2.minEnclosingCircle(c2)
        M2 = cv2.moments(c2)
        center_right = (int(M2["m10"] / (M2["m00"]+0.000001)), int(M2["m01"] / (M2["m00"]+0.000001)))
        center_right = (center_right[0]+width//2,center_right[1])
    

        if radius2 > circle_radius:

            cv2.circle(frame, (int(x2)+width//2, int(y2)), int(radius2),
                (0, 255, 255), 2)
            cv2.circle(frame, center_right, 5, (0, 0, 255), -1)
            if center_right[1] < (height//2 - windowSize//2):
                if prev ==None:
                   prev = "UP" 
                else:
                    prev = now
                now = 'UP'
                if prev =='RIGHT' and now =='UP':
                    PressKey(P)
                    ReleaseKey(P)
                if prev =='DOWN' and now =='UP':
                    print('swipe up')
                    PressKey(W)
                    PressKey(W)
                    PressKey(W)
                    ReleaseKey(W)
                cv2.putText(frame,'UP',(200,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255))
                
            elif center_right[1] > (height//2 + windowSize//2):
                if prev ==None:
                   prev = "DOWN" 
                else:
                    prev = now
                now = 'DOWN'
                if prev =='RIGHT' and now =='DOWN':
                    print('swipe right ')
                    PressKey(S)
                    PressKey(S)
                    ReleaseKey(S)
                cv2.putText(frame,'DOWN',(200,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255))
                

    frame_copy = frame.copy()
    frame_copy = cv2.rectangle(frame_copy,(0,height//2 - windowSize//2),(width,height//2 + windowSize//2),(255,0,0),2)
    cv2.imshow("Frame", frame_copy)


    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
 

vs.stop() 
cv2.destroyAllWindows()