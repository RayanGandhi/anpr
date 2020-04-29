import numpy as np
import numpy
import cv2
import  imutils
import sys
import pandas as pd
import time

image = cv2.imread("1.png")

image = imutils.resize(image, width=500)
cv2.imshow("Original Image", image)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

gray = cv2.bilateralFilter(gray, 11, 17, 17)

edged = cv2.Canny(gray, 170, 200)


( cnts,new) = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cnts=sorted(cnts, key = cv2.contourArea, reverse = True)[:30] 
NumberPlateCnt = None

count = 0
for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        print(approx)
        if len(approx) == 4:  
            NumberPlateCnt = approx 
            print(approx)
            count=+1
            break

mask = np.zeros(gray.shape,np.uint8)

new_image = cv2.drawContours(mask,[NumberPlateCnt],0,255,-1)
new_image = cv2.bitwise_and(image,image,mask=mask)
#cv2.namedWindow("Final_image",cv2.WINDOW_NORMAL)
#cv2.imshow("Final_image",new_image)
cv2.waitKey(0)
cv2.imwrite("Final_image1.jpg", new_image) 
cv2.waitKey(0)



