import cv2
import numpy as np
from object_module import *

def drawAxes(img, corners, imgpts):
    print(corners)
    corner = tuple(np.uint8(corners[0].ravel()))
    img = cv2.line(img, corner, tuple(np.uint8(imgpts[0].ravel())), (255,0,0), 5)
    img = cv2.line(img, corner, tuple(np.uint8(imgpts[1].ravel())), (0,255,0), 5)
    img = cv2.line(img, corner, tuple(np.uint8(imgpts[2].ravel())), (0,0,255), 5)
    return img



mtx = np.array([[1.32548040e+03, 0, 5.94395896e+02],
               [0, 1.32305552e+03, 3.86845731e+02], [0, 0, 1]])
dist = np.array([[-1.79283762e-02, -1.43973406e+00, -1.73491243e-03, -3.35946932e-02, 6.92834210e+00]])

axis = np.float32([[480,0,0], [0,480,0], [0,0,-480]]).reshape(-1,3)

if __name__ == '__main__':
    obj = three_d_object('data/Hulk/Hulk.obj', 'data/Hulk/hulk.png')

    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_50)
    arucoDetecter = cv2.aruco.ArucoDetector(dictionary)

    cv2.namedWindow("webcam")
    vc = cv2.VideoCapture(0)
    rval, frame = vc.read()
    while rval:
        rval, frame = vc.read()  # fetch frame from webcam
        key = cv2.waitKey(1)
        if key == 27:  # Escape key to exit the program
            break

        corners, ids, rejectedImgPoints = arucoDetecter.detectMarkers(frame)
        ret = False
        rvecs, tvecs, imgp = [], [], []

        if corners != ():
            imgp = corners[0]
            objp = np.array([[0, 0, 0], [480, 0, 0], [480, 480, 0], [0, 480, 0]], dtype="float32")
            ret, rvecs, tvecs, _ = cv2.solvePnPRansac(objp, imgp, mtx, dist)

        if not ret:
            cv2.imshow("webcam", frame)
            continue

        
        imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, mtx, dist)
        augmented = drawAxes(frame,imgp[0],imgpts)

        cv2.imshow("webcam", frame)
