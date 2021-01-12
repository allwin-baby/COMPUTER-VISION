import cv2
import numpy as np
from pynput.mouse import Button,Controller
import wx
import math

mouse  = Controller()
app = wx.App(False)
(sx,sy) = wx.DisplaySize() 
(camx,camy) = (640 ,480) 

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(3,camx)
cap.set(4,camy)

mlocOld = np.array([0,0])
mouseLoc = np.array([0,0])
dampening_factor = 2

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
cv2.createTrackbar('clicklen','test',0,20,nothing)  
cv2.createTrackbar('Threshold below  ','test',0,255,nothing)
cv2.createTrackbar('threshold_To','test',255,255,nothing) 

global color_set
global  clicked
clicked =False
color_set = False

while 1:
    global x1,x2,x3,y1,y2,y3
    suc,frame = cap.read()
    frame = cv2.flip(frame,1)
    hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)


    if color_set ==False:
        kernalOpenSize= cv2.  getTrackbarPos('KernalOpenSize','test')
        kernalCloseSize= cv2.getTrackbarPos('KernalCloseSize','test')
        kernalOpen =  np.ones((kernalOpenSize,kernalOpenSize))
        kernalClose =  np.ones((kernalCloseSize,kernalCloseSize))
        clicklen = cv2.getTrackbarPos('clicklen','test')
        
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
     
        mask = cv2.inRange(hsv,lowerbound,upperbound) 
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)

        #grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #value = (35, 35)
        #blurred = cv2.GaussianBlur(grey, value, 0)


        maskOpen = cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernalOpen)
        maskClose = cv2.morphologyEx(mask,cv2.MORPH_CLOSE,kernalClose)
        maskFinal = maskClose

        countours,hierarchy = cv2.findContours(maskFinal.copy(),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(frame,countours,-1,(255,255,255),3)
    else:
        kernalOpenSize= cv2.getTrackbarPos('KernalOpenSize','test')
        kernalCloseSize= cv2.getTrackbarPos('KernalCloseSize','test')
        kernalOpen =  np.ones((kernalOpenSize,kernalOpenSize))
        kernalClose =  np.ones((kernalCloseSize,kernalCloseSize))
        clicklen = cv2.getTrackbarPos('clicklen','test')

        hl= cv2.getTrackbarPos('HUE LOW','test')
        sl= cv2.getTrackbarPos('SATURATION LOW','test')
        vl= cv2.getTrackbarPos('VALUE LOW','test')

        hh= cv2.getTrackbarPos('HUE HIGH','test')
        sh= cv2.getTrackbarPos('SATURATION HIGH','test')
        vh =cv2.getTrackbarPos('VALUE HIGH','test')
        lowerbound = np.array([hl,sl,vl])
        upperbound = np.array([hh,sh,vh])
        lowerbound = np.array([hl,sl,vl])
        upperbound = np.array([hh,sh,vh])
     
        mask = cv2.inRange(hsv,lowerbound,upperbound) 
        mask = cv2.erode(mask, None, iterations=5)
        mask = cv2.dilate(mask, None, iterations=5)
        cv2.imshow('mask3',mask)

        maskOpen = cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernalOpen)
        maskClose = cv2.morphologyEx(mask,cv2.MORPH_CLOSE,kernalClose)
        maskFinal = maskClose

        countours,hierarchy = cv2.findContours(maskFinal.copy(),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(frame,countours,-1,(255,255,255),3)

        if len(countours) ==2:
            if cv2.contourArea(countours[0]) > 50 and cv2.contourArea(countours[1]) > 50 : 
                x1,y1,w1,h1 = cv2.boundingRect(countours[0])
                x2,y2,w2,h2 = cv2.boundingRect(countours[1])
                cv2.rectangle(frame,(x1,y1),(x1+w1,y1+h1),(0,255,0),2)
                cv2.rectangle(frame,(x2,y2),(x2+w2,y2+h2),(0,255,0),2)
                cx1 = x1+w1/2
                cy1 = y1+h1/2
                cx2 = x2+ w2/2
                cy2 = y2+h2/2
                cx = (cx1+cx2)/2
                cy = (cy1+cy2)/2


                mouse.position = (cx*sx/camx,cy*sy/camy)
                frame = cv2.putText(frame, str(str(cx1) +" "+str(cy1)), (int(cx1),int(cy1)), cv2.FONT_HERSHEY_SIMPLEX ,1,(255,255,0),2) 
                frame = cv2.putText(frame, str(str(cx2)+" "+str(cy2)), (int(cx2),int(cy2)), cv2.FONT_HERSHEY_SIMPLEX ,1,(255,255,0),2) 
                cv2.line(frame,(int(cx1),int(cy1)),(int(cx2),int(cy2)),(255,255,255),2)


                cv2.putText(frame,str(math.sqrt(abs(cx1-cx2)+abs( cy1-cy2))), (50,50,),cv2.FONT_HERSHEY_SIMPLEX ,1,(255,255,0),2) 
                if clicked ==True:
                    print('relasesd')
                    mouse.release(Button.left)
                    clicked = False
        elif len(countours) == 1:
            if (cv2.contourArea(countours[0])) >300:
                x1,y1,w1,h1 = cv2.boundingRect(countours[0])
                cv2.rectangle(frame,(x1,y1),(x1+w1,y1+h1),(0,255,0),2)
                cx = x1+w1/2
                cy = y1+h1/2

                mouse.position = (cx*sx/camx,cy*sy/camy)
                cv2.circle(frame,(int(cx),int(cy)),2,(0,255,0),2)

                if clicked ==False:
                    print('clicked')
                    mouse.press(Button.left)
                    clicked = True
                else:
                    pass
    k = cv2.waitKey(2)
    if k==ord('s'):
        color_set =True
    cv2.imshow('frame',frame)
    half = cv2.resize(frame, (0, 0), fx = 0.5, fy = 0.5) 
    cv2.imshow('testtt',half)
    if cv2.waitKey(1) == ord('a'):
        break
cv2.destroyAllWindows() 