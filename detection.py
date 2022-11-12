import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
  
img = cv.imread('images/balls.jpg')
  
(b, g, r) = cv.split(img)

bitwiseGR = cv.bitwise_xor(g, r) # gets where there it is both red and green and marks it false
bitwiseR =  cv.bitwise_and(bitwiseGR, r) # removes where it is both red and green and keeps red only

bitwiseGB = cv.bitwise_xor(g, b) # gets where there it is both blue and green and marks it false
bitwiseB =  cv.bitwise_and(bitwiseGB, b) # removes where it is both blue and green and keeps blue only

thres, bitwiseR = cv.threshold(bitwiseR, 70, 255, cv.THRESH_BINARY) #use threshold so the area with balls are very clearly white
thres, bitwiseB = cv.threshold(bitwiseB, 70, 255, cv.THRESH_BINARY)

#could still detect purple at this point
bitwiseBR = cv.bitwise_xor(r, b) # gets where there it is both red and blue and marks it false
bitwiseR = cv.bitwise_and(bitwiseBR, bitwiseR) # removes where it is both red and blue and keeps red or blue respectively
bitwiseB = cv.bitwise_and(bitwiseBR, bitwiseB)

thres, thresholdRed = cv.threshold(bitwiseR, 70, 255, cv.THRESH_BINARY) #use threshold so the area with balls are very clearly white
thres, thresholdBlue = cv.threshold(bitwiseB, 70, 255, cv.THRESH_BINARY)

blurRed = cv.GaussianBlur(thresholdRed, (51, 51), 0) #use a lot of blur so any areas with even a little bit of gray go black
thres, blurRed = cv.threshold(blurRed, 230, 255, cv.THRESH_BINARY)

blurBlue = cv.GaussianBlur(thresholdBlue, (51, 51), 0)
thres, blurBlue = cv.threshold(blurBlue, 230, 255, cv.THRESH_BINARY)

contoursRed, hierarchy = cv.findContours(blurRed, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE) #find all the contours

contoursBlue, hierarchy = cv.findContours(blurBlue, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

for index, contour in enumerate(contoursRed): #some small contours may still be found, so it filters using area
    if cv.contourArea(contour) > 3000:
        x, y, w, h = cv.boundingRect(contour)
        cv.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2) #if it passes area check, it draws it
        M = cv.moments(contour)
        x = int(M['m10']/M['m00'])
        y = int(M['m01']/M['m00'])
        cv.putText(img, 'Red Ball', (x, y), cv.FONT_HERSHEY_TRIPLEX, 1, (255, 255, 255), 2)

for index, contour in enumerate(contoursBlue): #same as red
    if cv.contourArea(contour) > 3000:
        x, y, w, h = cv.boundingRect(contour)
        cv.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        M = cv.moments(contour)
        x = int(M['m10']/M['m00'])
        y = int(M['m01']/M['m00'])
        cv.putText(img, 'Blue Ball', (x, y), cv.FONT_HERSHEY_TRIPLEX, 1, (255, 255, 255), 2)

cv.imshow('Final', img)

cv.waitKey(0)