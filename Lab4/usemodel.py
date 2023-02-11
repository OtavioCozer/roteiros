import numpy as np
import cv2
import os
import imutils

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow import keras

def sliding_window(image, ystep, xstep, windowSize):
    # slide a window across the image
	for y in range(0, image.shape[0] - windowSize[0] + 1, ystep):
		for x in range(0, image.shape[1] - windowSize[1] + 1, xstep):
        	# yield the current window
			yf, xf = y + windowSize[0], x + windowSize[1]
			# print(f"yf: {yf}, xf: {xf}, y: {y}, x: {x}")
			# yf = imH - 1 if yf > imH - 1 else yf
			# xf = imW if xf > imW else xf

			yield (y, yf, x, xf), image[y:yf, x:xf]


height, width = 90, 90
num_classes = 6
class_names = ['0', '10', '100', '25', '5', '50']

model = keras.models.load_model('mymodel')

img = keras.utils.load_img(
    "2023-02-10-143243.jpg"
)

img = tf.keras.utils.img_to_array(img)
img = imutils.resize(img, int(500))
indices, patches = zip(*sliding_window(img, 10, 10, (90, 90)))
# patches_hog = np.array([hog.compute(patch) for patch in patches])

# for im in patches:
	# print(f"shape: {im.shape}")
	# cv2.imshow("Patche", im)
	# cv2.waitKey(0)
	# cv2.destroyWindow("Patche")

correct_indices =[]
for im, index in zip(patches, indices):
    img_array = tf.expand_dims(im, 0) # Create a batch

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    if class_names[np.argmax(score)] != '0':
        correct_indices.append((index, class_names[np.argmax(score)]))

    # print(
    #     "This image most likely belongs to {} with a {:.2f} percent confidence."
    #     .format(class_names[np.argmax(score)], 100 * np.max(score))
    # )



coins = cv2.imread("2023-02-10-143243.jpg")
coins = imutils.resize(coins, int(500))
for index, label in correct_indices:
    y, yf, x, xf = index
    print(index, label)
    
    coins = cv2.rectangle(coins, (x, y), (xf, yf), (255, 0, 0))
    coins = cv2.putText(coins, f"{label}", (x, y), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0))

cv2.imwrite("coins.jpg", coins)