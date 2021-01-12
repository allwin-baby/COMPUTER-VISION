import cv2
import numpy as np
import time
from directkeys import  Left,Right,Down,Up
from directkeys import PressKey, ReleaseKey 

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 1)
(camx,camy) = (640,480)
fps = cap.get(cv2.CAP_PROP_FPS)
print(fps)
cap.set(3,camx)
cap.set(4,camy)
print(cap.get(3))
print(cap.get(4))


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
cv2.createTrackbar('Threshold below  ','test',0,255,nothing)
cv2.createTrackbar('threshold_To','test',255,255,nothing)     
cv2.createTrackbar('KernalOpenSize','test',5,50,nothing)    
cv2.createTrackbar('KernalCloseSize','test',5,50,nothing)  
cv2.createTrackbar('waitkey','test',1,1000,nothing) 


global blueclicked 
blueclicked =False
global greenclicked 
greenclicked =False


while 1:
    global bx1,bx2,bx3,by1,by2,by3,gx1,gx2,gx3,gy1,gy2,gy3,bko,bkc,gko,gkc
    suc,frame = cap.read()
    frame = cv2.flip(frame,1)
    hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    wait_key= cv2.getTrackbarPos('waitkey','test') 
    if blueclicked ==False or greenclicked ==False:

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
        lowerbound = np.array([hl,sl,vl])
        upperbound = np.array([hh,sh,vh])
    


        mask = cv2.inRange(hsv,lowerbound,upperbound) 
        bitwise = cv2.bitwise_and(frame,frame ,mask = mask)
        mask2 = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)


        maskOpen = cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernalOpen)  
        maskClose = cv2.morphologyEx(mask,cv2.MORPH_CLOSE,kernalClose)
        maskFinal = maskClose


        countours,hierarchy = cv2.findContours(maskFinal.copy(),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(frame,countours,-1,(255,255,255),3)
    if blueclicked ==True and greenclicked ==True: 
        
        blowerbound = np.array([bx1,bx2,bx3])
        bupperbound = np.array([by1,by2,by3])
        #blowerbound = np.array([97, 122 ,116])
        #bupperbound = np.array([119, 255, 255])


        bmask = cv2.inRange(hsv,blowerbound,bupperbound) 
        bmask2 = cv2.cvtColor(bmask, cv2.COLOR_GRAY2BGR)
        

        bmaskOpen = cv2.morphologyEx(bmask,cv2.MORPH_OPEN,bko)  
        bmaskClose = cv2.morphologyEx(bmask,cv2.MORPH_CLOSE,bkc)
        bmaskFinal = bmaskClose
        #contours
        bcountours,hierarchy = cv2.findContours(bmaskFinal.copy(),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(frame,bcountours,-1,(255,255,255),3)

        glowerbound = np.array([gx1,gx2,gx3])
        gupperbound = np.array([gy1,gy2,gy3])
        #glowerbound = np.array([30, 111 ,48])
        #gupperbound = np.array([81, 158 ,111])

        gmask = cv2.inRange(hsv,glowerbound,gupperbound) 
        gmask2 = cv2.cvtColor(gmask, cv2.COLOR_GRAY2BGR)


        gmaskOpen = cv2.morphologyEx(gmask,cv2.MORPH_OPEN,gko)  
        gmaskClose = cv2.morphologyEx(gmask,cv2.MORPH_CLOSE,gkc)
        gmaskFinal = gmaskClose

        gcountours,hierarchy = cv2.findContours(gmaskFinal.copy(),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(frame,gcountours,-1,(255,255,255),3)
        if len(gcountours) ==1 and len(bcountours) ==1:
            x1,y1,w1,h1 = cv2.boundingRect(gcountours[0])
            x2,y2,w2,h2 = cv2.boundingRect(bcountours[0])
            cv2.rectangle(frame,(x1,y1),(x1+w1,y1+h1),(0,255,0),2)
            cv2.rectangle(frame,(x2,y2),(x2+w2,y2+h2),(0,255,0),2)
            cx1 = int(x1+w1/2)
            cy1 = int(y1+h1/2)
            cx2 = int(x2+ w2/2)
            cy2 = int(y2+h2/2)
            cv2.line(frame,(cx1,cy1),(cx2,cy2),(255,255,255),2)
            print(cx1,cx2)
            import math
            PI =3.14159265
            if cx2 -cx1>=0:
                ReleaseKey(Down)
                PressKey(Up)
            else: 
                ReleaseKey(Up)
                PressKey(Down)
            try:
                m1 = (cy2 - cy1) / abs(cx2 - cx1)  
                A = math.atan(m1) * 180 / PI
            except:
                A= 0
            if A < -10:
                
                ReleaseKey(Right)
                PressKey(Left)

            if 10>A>-10:
                
                ReleaseKey(Right)
                ReleaseKey(Left)

            if A > 10:
                
                ReleaseKey(Left)
                PressKey(Right)

               
            frame = cv2.putText(frame, str(A), (20,20), cv2.FONT_HERSHEY_SIMPLEX ,1,(255,255,0),2) 
        else:
            ReleaseKey(Left)
            ReleaseKey(Right) 

    cv2.line(frame,(0,340),(640,340),(255,255,255),2)

    cv2.imshow('frame',frame)
    if cv2.waitKey(1) == ord('b'):
        print('b pressed')
        bx1,bx2,bx3 = hl,sl,vl
        by1,by2,by3 = hh,sh,vh
        bko= kernalOpen
        bkc = kernalClose
        blueclicked = True
        print(bx1,bx2,bx3)
        print(by1,by2,by3)
        print(bko,bkc)
    if cv2.waitKey(1) == ord('g'):
        print('g pressed')
        gko =kernalOpen
        gkc = kernalClose
        gx1,gx2,gx3 = hl,sl,vl
        gy1,gy2,gy3 = hh,sh,vh
        print(gx1,gx2,gx3)
        print(gy1,gy2,gy3)
        print(gko,gkc)
        greenclicked = True
    if cv2.waitKey(1) == ord('q'):
        break     
       
cv2.destroyAllWindows()  
print(bx1,bx2,bx3)
print(by1,by2,by3) 
print(gx1,gx2,gx3)
print(gy1,gy2,gy3)