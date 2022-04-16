import cvzone
import cv2
from cvzone.ColorModule import ColorFinder
import socket

cap = cv2.VideoCapture(0)

success, img = cap.read()
h,w,_=img.shape

cap.set(3,1280)
cap.set(4,720)

myColorFinder = ColorFinder(False)
hsvVals = {'hmin': 171, 'smin': 97, 'vmin': 89, 'hmax': 179, 'smax': 255, 'vmax': 172}

sock = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
serverAddressPort=("127.0.0.1",5051)


while True:
    success, img = cap.read()

    imgColor, mask = myColorFinder.update(img,hsvVals)

    imgContours,contours = cvzone.findContours(img,mask)

    if contours:
        data = contours[0]['center'][0], \
               h-contours[0]['center'][1], \
               int(contours[0]['area'])
        print(data)
        data = str.encode(str(data))
        sock.sendto(data,serverAddressPort)


    imgStack = cvzone.stackImages([img,imgColor,mask,imgContours],2,0.5)

    imgContours=cv2.resize(imgContours,(0,0),None,0.3,0.3)
    cv2.imshow("Image Countours",imgContours)


    cv2.waitKey(1)
