import cv2
import numpy as np
import glob

winSize = (10,10)
blockSize = (10,10)
blockStride = (5,5)
cellSize = (10,10)
nbins = 9
derivAperture = 1
winSigma = -1.
histogramNormType = 0
L2HysThreshold = 0.2
gammaCorrection = 1
nlevels = 64
signedGradients = False
 
hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,
cellSize,nbins,derivAperture,
winSigma,histogramNormType,L2HysThreshold,
gammaCorrection,nlevels, signedGradients)

dataset: list[str] = glob.glob("negativeDataset/*.jpg") + glob.glob("positiveDataset/*.jpg")
trainMat = []
trainLabels = []
for name in dataset:
    img = cv2.imread(name)
    trainMat.append(hog.compute(img))
    trainLabels.append(int(name.split("/")[-1].split("_")[0]))


    
svm = cv2.ml.SVM_create()
svm.setType(cv2.ml.SVM_C_SVC)
svm.setKernel(cv2.ml.SVM_LINEAR)
svm.setTermCriteria((cv2.TermCriteria_MAX_ITER, 100, 1e-6))

svm.trainAuto(np.array(trainMat) , cv2.ml.ROW_SAMPLE, np.array(trainLabels))
svm.save("newModel.yml")
