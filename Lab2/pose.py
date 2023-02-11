import numpy as np
import cv2 as cv
import glob

mtx = np.array([[1.32548040e+03, 0, 5.94395896e+02],
               [0, 1.32305552e+03, 3.86845731e+02], [0, 0, 1]])
dist = np.array([[-1.79283762e-02, -1.43973406e+00, -
                1.73491243e-03, -3.35946932e-02, 6.92834210e+00]])

def draw(img, corners, imgpts):
    corners = np.uint8(corners)
    imgpts = np.uint8(imgpts)
    corner = tuple(corners[0].ravel())
    img = cv.line(img, corner, tuple(imgpts[0].ravel()), (255,0,0), 5)
    img = cv.line(img, corner, tuple(imgpts[1].ravel()), (0,255,0), 5)
    img = cv.line(img, corner, tuple(imgpts[2].ravel()), (0,0,255), 5)
    return img

criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
objp = np.zeros((6*9,3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)
axis = np.float32([[3,0,0], [0,3,0], [0,0,-3]]).reshape(-1,3)

cv.namedWindow("webcam")
vc = cv.VideoCapture(0)
rval, frame = vc.read()

while rval:
    # print(1)
    rval, frame = vc.read()
    key = cv.waitKey(1)
    if key == 27:  # Escape key to exit the program
        break
    gray = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    ret, corners = cv.findChessboardCorners(gray, (9,6),None)
    if ret == True:
        corners2 = cv.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        # Find the rotation and translation vectors.
        ret,rvecs, tvecs, _ = cv.solvePnPRansac(objp, corners2, mtx, dist)
        # project 3D points to image plane
        imgpts, jac = cv.projectPoints(axis, rvecs, tvecs, mtx, dist)
        frame = draw(frame,corners2,imgpts)
        cv.imshow('webcam',frame)
    else:
        cv.imshow('webcam',frame)
cv.destroyAllWindows()