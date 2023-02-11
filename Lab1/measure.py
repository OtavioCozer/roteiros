from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import numpy as np
import argparse
import imutils
import cv2
WINDOW_WIDTH = 600
WINDOW_HEIGHT = int(WINDOW_WIDTH * 0.75)
print(WINDOW_WIDTH, WINDOW_HEIGHT)

def midpoint(ptA, ptB):
	return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to the input image")
ap.add_argument("-w", "--width", type=float, required=True,
	help="width of the left-most object in the image (in inches)")
args = vars(ap.parse_args())

image = cv2.imread(args["image"])
click = image.copy()
original = image.copy()
points = []

def click_event(event, x, y, flags, params):
    # checking for left mouse clicks
	if event == cv2.EVENT_LBUTTONDOWN:
		points, image = params

		points.append([x, y])
 
        # displaying the coordinates
        # on the image window
		cv2.circle(image, (int(x), int(y)), 20, (0, 255, 0), -1)
		cv2.namedWindow("Click", cv2.WINDOW_KEEPRATIO + cv2.WINDOW_GUI_NORMAL)
		cv2.resizeWindow("Click", WINDOW_WIDTH, WINDOW_HEIGHT) 
		cv2.imshow("Click", image)
		

#corner detection
cv2.namedWindow("Click", cv2.WINDOW_KEEPRATIO + cv2.WINDOW_GUI_NORMAL)
cv2.resizeWindow("Click", WINDOW_WIDTH, WINDOW_HEIGHT) 
cv2.imshow("Click", image)
cv2.setMouseCallback("Click", click_event, param=(points, click))
while len(points) < 4:
	cv2.waitKey(10)

cv2.destroyWindow("Click")
print(points)


source_points = np.float32(points)
output_points = np.float32([[0,0],[WINDOW_WIDTH,0],[0,WINDOW_HEIGHT],[WINDOW_WIDTH,WINDOW_HEIGHT]])
M = cv2.getPerspectiveTransform(source_points, output_points)
image = cv2.warpPerspective(image,M,(WINDOW_WIDTH,WINDOW_HEIGHT))
cv2.namedWindow("Perspective", cv2.WINDOW_KEEPRATIO + cv2.WINDOW_GUI_NORMAL)
cv2.resizeWindow("Perspective", WINDOW_WIDTH, WINDOW_HEIGHT)    
cv2.imshow("Perspective", image)
cv2.imwrite("perspective.jpg", image)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
# thresh = cv2.erode(thresh, None, iterations=1)
# thresh = cv2.dilate(thresh, None, iterations=1)
cv2.namedWindow("Seg", cv2.WINDOW_KEEPRATIO + cv2.WINDOW_GUI_NORMAL)
cv2.resizeWindow("Seg", WINDOW_WIDTH, WINDOW_HEIGHT)    
cv2.imshow("Seg", thresh)
cv2.imwrite("seg.jpg", thresh)

# noise removal
kernel = np.ones((3,3),np.uint8)
opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)
# sure background area
sure_bg = cv2.dilate(opening,kernel,iterations=3)
# Finding sure foreground area
dist_transform = cv2.distanceTransform(thresh,cv2.DIST_L2,5)
ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)
# Finding unknown region
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg,sure_fg)


cv2.namedWindow("opening", cv2.WINDOW_KEEPRATIO + cv2.WINDOW_GUI_NORMAL)
cv2.resizeWindow("opening", WINDOW_WIDTH, WINDOW_HEIGHT)
cv2.imshow("opening", opening)
cv2.imwrite("opening.jpg", opening)

cv2.namedWindow("distant", cv2.WINDOW_KEEPRATIO + cv2.WINDOW_GUI_NORMAL)
cv2.resizeWindow("distant", WINDOW_WIDTH, WINDOW_HEIGHT)
cv2.imshow("distant", dist_transform)   
cv2.imwrite("distant.jpg", dist_transform)

cv2.namedWindow("fg", cv2.WINDOW_KEEPRATIO + cv2.WINDOW_GUI_NORMAL)
cv2.resizeWindow("fg", WINDOW_WIDTH, WINDOW_HEIGHT)    
cv2.imshow("fg", sure_fg)
cv2.imwrite("fg.jpg", sure_fg)

cv2.namedWindow("bg", cv2.WINDOW_KEEPRATIO + cv2.WINDOW_GUI_NORMAL)
cv2.resizeWindow("bg", WINDOW_WIDTH, WINDOW_HEIGHT)    
cv2.imshow("bg", sure_bg)
cv2.imwrite("bg.jpg", sure_bg)

cv2.namedWindow("unknown", cv2.WINDOW_KEEPRATIO + cv2.WINDOW_GUI_NORMAL)
cv2.resizeWindow("unknown", WINDOW_WIDTH, WINDOW_HEIGHT)    
cv2.imshow("unknown", unknown)
cv2.imwrite("unknown.jpg", unknown)


# Marker labelling
ret, markers = cv2.connectedComponents(sure_fg)
# Add one to all labels so that sure background is not 0, but 1
markers = markers+1
# Now, mark the region of unknown with zero
markers[unknown==255] = 0

markers = cv2.watershed(image,markers)

nrow = len(markers)
ncol = len(markers[0])
test = gray.copy()
for i in range(nrow):
	for j in range(ncol):
		if markers[i][j] == -1 and 0 < i and i < nrow - 1 and 0 < j and j < ncol - 1 :
			test[i][j] = 255
		else:
			test[i][j] = 0


cv2.namedWindow("Watershed", cv2.WINDOW_KEEPRATIO + cv2.WINDOW_GUI_NORMAL)
cv2.resizeWindow("Watershed", WINDOW_WIDTH, WINDOW_HEIGHT)    
cv2.imshow("Watershed", test)
cv2.imwrite("watershed.jpg", test)


########################

# perform edge detection, then perform a dilation + erosion to
# close gaps in between object edges
edged = cv2.Canny(gray, 50, 100)
edged = cv2.dilate(edged, None, iterations=1)
edged = cv2.erode(edged, None, iterations=1)
cv2.namedWindow("edged", cv2.WINDOW_KEEPRATIO + cv2.WINDOW_GUI_NORMAL)
cv2.resizeWindow("edged", WINDOW_WIDTH, WINDOW_HEIGHT)    
cv2.imshow("edged", edged)
cv2.imwrite("edged.jpg", edged)



# find contours in the edge map
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)

# sort the contours from left-to-right and initialize the
# 'pixels per metric' calibration variable
(cnts, _) = contours.sort_contours(cnts)
pixelsPerMetric = None

orig = image.copy()
# loop over the contours individually
for c in cnts:
	# if the contour is not sufficiently large, ignore it
	if cv2.contourArea(c) < 100:
		continue
	# compute the rotated bounding box of the contour
	box = cv2.minAreaRect(c)
	box = cv2.boxPoints(box)
	box = np.array(box, dtype="int")
	# order the points in the contour such that they appear
	# in top-left, top-right, bottom-right, and bottom-left
	# order, then draw the outline of the rotated bounding
	# box
	box = perspective.order_points(box)
	cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 2)
	# loop over the original points and draw them
	for (x, y) in box:
		cv2.circle(orig, (int(x), int(y)), 5, (0, 0, 255), -1)

	# unpack the ordered bounding box, then compute the midpoint
	# between the top-left and top-right coordinates, followed by
	# the midpoint between bottom-left and bottom-right coordinates
	(tl, tr, br, bl) = box
	(tltrX, tltrY) = midpoint(tl, tr)
	(blbrX, blbrY) = midpoint(bl, br)
	# compute the midpoint between the top-left and top-right points,
	# followed by the midpoint between the top-righ and bottom-right
	(tlblX, tlblY) = midpoint(tl, bl)
	(trbrX, trbrY) = midpoint(tr, br)
	# draw the midpoints on the image
	cv2.circle(orig, (int(tltrX), int(tltrY)), 5, (255, 0, 0), -1)
	cv2.circle(orig, (int(blbrX), int(blbrY)), 5, (255, 0, 0), -1)
	cv2.circle(orig, (int(tlblX), int(tlblY)), 5, (255, 0, 0), -1)
	cv2.circle(orig, (int(trbrX), int(trbrY)), 5, (255, 0, 0), -1)
	# draw lines between the midpoints
	cv2.line(orig, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)),
		(255, 0, 255), 2)
	cv2.line(orig, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)),
		(255, 0, 255), 2)

	# compute the Euclidean distance between the midpoints
	dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
	dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))
	# if the pixels per metric has not been initialized, then
	# compute it as the ratio of pixels to supplied metric
	# (in this case, inches)
	if pixelsPerMetric is None:
		pixelsPerMetric = dB / args["width"]

	
	# compute the size of the object
	dimA = dA / pixelsPerMetric
	dimB = dB / pixelsPerMetric
	# draw the object sizes on the image
	cv2.putText(orig, "{:.1f}mm".format(dimA),
		(int(tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX,
		0.55, (255, 255, 255), 1)
	cv2.putText(orig, "{:.1f}mm".format(dimB),
		(int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX,
		0.55, (255, 255, 255), 1)
	# show the output image
	cv2.namedWindow("Image", cv2.WINDOW_KEEPRATIO + cv2.WINDOW_GUI_NORMAL)
	cv2.resizeWindow("Image", WINDOW_WIDTH, WINDOW_HEIGHT)    
	cv2.imshow("Image", orig)
	cv2.imwrite("Image.jpg", orig)

cv2.waitKey()
