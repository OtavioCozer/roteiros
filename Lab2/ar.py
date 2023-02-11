import cv2
import numpy as np
import math
from object_module import *


def get_extended_RT(A, H):
    R_12_T = np.linalg.inv(A).dot(H)

    r1 = R_12_T[:, 0]
    r2 = R_12_T[:, 1]
    T = R_12_T[:, 2]

    norm = math.sqrt(np.linalg.norm(r1) * np.linalg.norm(r2))

    r3 = np.cross(r2, r1)/(norm)
    R_T = np.zeros((3, 4))
    R_T[:, 0] = r1
    R_T[:, 1] = r2
    R_T[:, 2] = r3
    R_T[:, 3] = T
    return R_T


A = np.array([[1.32548040e+03, 0, 5.94395896e+02],
             [0, 1.32305552e+03, 3.86845731e+02], [0, 0, 1]])

if __name__ == '__main__':
    obj = three_d_object('data/Hulk/Hulk.obj', 'data/Hulk/hulk.png')

    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_50)
    arucoDetecter = cv2.aruco.ArucoDetector(dictionary)

    cv2.namedWindow("webcam")
    vc = cv2.VideoCapture(0)
    rval, frame = vc.read()
    H0, H1, H = [], [], []

    while rval:
        rval, frame = vc.read()  # fetch frame from webcam
        key = cv2.waitKey(1)
        if key == 27:  # Escape key to exit the program
            break

        corners, ids, rejectedImgPoints = arucoDetecter.detectMarkers(frame)
        
        if ids is not None:
            ids = ids.ravel()
            i0 = np.where(ids == 0)[0].ravel()
            i1 = np.where(ids == 2)[0].ravel()

            if len(i0) > 0:
                i = i0[0]
                dst_pts0 = corners[i]
                src_pts0 = np.array(
                    [[0, 0], [480, 0], [480, 480], [0, 480]], dtype="float32")
                H, _ = cv2.findHomography(src_pts0, dst_pts0, cv2.RANSAC, 5.0)
                H0 = H if len(H) > 0 else H0

            if len(i1) > 0:
                i = i1[0]
                dst_pts0 = corners[i]
                src_pts0 = np.array(
                    [[0, 0], [480, 0], [480, 480], [0, 480]], dtype="float32")
                H, _ = cv2.findHomography(src_pts0, dst_pts0, cv2.RANSAC, 5.0)
                H1 = H if len(H) > 0 else H1

        if len(H1) > 0:
            print(H1)
            R_T = get_extended_RT(A, H1)
            transformation = A.dot(R_T)
            frame = augment(frame, obj, transformation, 480, 480, 100)

        if len(H0) > 0:
            R_T = get_extended_RT(A, H0)
            transformation = A.dot(R_T)
            frame = augment(frame, obj, transformation, 480, 480, 100)

        cv2.imshow("webcam", frame)
