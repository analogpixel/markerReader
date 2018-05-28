#!/usr/bin/env python

# pip install opencv-contrib-python
# http://www.philipzucker.com/aruco-in-opencv/

import numpy as np
import cv2
import cv2.aruco as aruco
 
 
'''
    drawMarker(...)
        drawMarker(dictionary, id, sidePixels[, img[, borderBits]]) -> img
'''
 
aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
print(aruco_dict)
# second parameter is id number
# last parameter is total image size

for i in range(0,4):
	img = aruco.drawMarker(aruco_dict, i, 700)
	cv2.imwrite("test_marker_{}.jpg".format(i) , img)
 
#cv2.imshow('frame',img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()