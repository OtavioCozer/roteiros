import cv2
import json
import glob

WINDOW_WIDTH = 600
WINDOW_HEIGHT = int(WINDOW_WIDTH * 0.75)
RADIUS = 45

names: list[str] = glob.glob("COCO_labelme_classification/classification/*.json")

for name in names:
    with open(name) as f:
        imageJson = json.load(f)

    imageName = name.split(".")
    imageName[-1] = "jpg"
    imageName = '.'.join(imageName)
    img = cv2.imread(imageName)

    shape = imageJson["shapes"][0]
    ptC = (shape["points"][0][0], shape["points"][0][1])
    pt2 = (int(ptC[0] + RADIUS), int(ptC[1] + RADIUS))
    pt1 = (int(ptC[0] - RADIUS), int(ptC[1] - RADIUS))

    try:
        resized = cv2.resize(img[pt1[1]:pt2[1], pt1[0]:pt2[0]], (90, 90))
        newFileName = "positiveDataset/" + imageName.split("/")[-1]
        ret = cv2.imwrite(newFileName, resized)
    except:
        continue # its a finger

    # cv2.namedWindow("Resized", cv2.WINDOW_KEEPRATIO + cv2.WINDOW_GUI_NORMAL)
    # cv2.resizeWindow("Resized", WINDOW_WIDTH, WINDOW_HEIGHT) 
    # cv2.imshow("Resized", resized)
    # cv2.waitKey(0)
    # cv2.destroyWindow("Resized")

    # cv2.namedWindow("Click", cv2.WINDOW_KEEPRATIO + cv2.WINDOW_GUI_NORMAL)
    # cv2.resizeWindow("Click", WINDOW_WIDTH, WINDOW_HEIGHT) 
    # cv2.rectangle(img, pt1, pt2, (255, 0, 0))
    # cv2.imshow("Click", img)
    # cv2.waitKey(0)
    # cv2.destroyWindow("Click")