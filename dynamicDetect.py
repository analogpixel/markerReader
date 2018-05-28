#!/usr/bin./env python

"""
Detect markers in an image, and then warp the image back to a rectangle and
output it without the markers
"""

import numpy as np
import cv2
import cv2.aruco as aruco
import pprint
import munch


# https://docs.opencv.org/3.4.0/da/d6e/tutorial_py_geometric_transformations.html
# https://docs.opencv.org/3.1.0/d5/dae/tutorial_aruco_detection.html


def sortPoints(corners, ids):
    """
    given the corners and ids from aruco.detectMarkers return a munch
    object of the which quaderant each marker is in, the id of the maker, and
    the points of the marker labeled based on their location:

    q1.ul <- this would give you the  upper left point of the marker in quadrant 1
    """

    debug=False
    """
    array positions
    +------------+
    | 0        1 |
    |            |
    |            |
    |            |
    |            |
    | 3        2 |
    +------------+
    """

    points = {'ul': 0, 'll': 3, 'lr': 2, 'ur': 1}

    #
    # Figure out max a min x and y points
    #
    xmin = 9000
    ymin = 9000
    xmax = 0
    ymax = 0
    for c in corners:
        for point in c[0]:
            if point[0] > xmax:
                xmax = point[0]
            if point[0] < xmin:
                xmin = point[0]
            if point[1] > ymax:
                ymax = point[1]
            if point[1] < ymin:
                ymin = point[1]

    xmid = (xmax - xmin) / 2
    ymid = (ymax - ymin) / 2

    if debug:
        print("xmax:", xmax)
        print("xmin:", xmin)
        print("ymax:", ymax)
        print("ymin:", ymin)
        print("xmid:", xmid)
        print("ymid:", ymid)

    xmid = (xmax+xmin) / 2
    ymid = (ymax+ymin) / 2
    #xmid = xmin + ( (xmax - xmin) / 2)
    #ymid = ymin + ( (ymax - ymin) / 2)
    ret = {}

    #
    # figure out what quad each marker is in
    i=0
    for c in corners:
        p1_x = c[0][1][0]
        p1_y = c[0][1][1]

        if (p1_x <= xmid)  and (p1_x >= xmin) and (p1_y <= ymid) and (p1_y >= ymin):
            q = 'q1'
        if (p1_x <= xmax)  and (p1_x > xmid)  and (p1_y <= ymid) and (p1_y >= ymin):
            q = 'q2'
        if (p1_x <= xmid)  and (p1_x >= xmin) and (p1_y <= ymax)  and (p1_y > ymid):
            q = 'q3'
        if (p1_x <= xmax)  and (p1_x > xmid)  and (p1_y <= ymax)  and (p1_y > ymid):
            q = 'q4'

        #
        # create the final dictionary of the marker id, quad, and points
        #
        ret[q]  = {'id': ids[i][0] }
        for p in points:
            ret[q][p] = [c[0][ points[p] ][0], c[0][ points[p] ][1]]

        i +=1

    # return a munch object of the data
    return munch.Munch.fromDict(ret)

cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, img = cap.read()
    #img = cv2.resize(img,(800,800), interpolation = cv2.INTER_CUBIC)
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    cv2.imshow('frame',img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    height, width, depth = img.shape
    aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
    parameters =  aruco.DetectorParameters_create()
    corners, ids, rejectedImgPoints = aruco.detectMarkers(img, aruco_dict, parameters=parameters)

    if len(corners) != 4:
        print("not enough points:", len(corners))
        continue

    sp =  sortPoints( corners, ids) 
    pprint.pprint(corners)
    pprint.pprint(sp)

    pts1 = np.float32(  [sp.q1.lr, sp.q2.ll, sp.q3.ur, sp.q4.ul])
    pts2 = np.float32([[0,0],[width,0],[0,height],[width,height]])

    M = cv2.getPerspectiveTransform(pts1,pts2)
    dst = cv2.warpPerspective(img, M, (width,height) )
    cv2.imwrite("input.jpg", img)
    cv2.imwrite("output.jpg", dst)
    break


cv2.destroyAllWindows()