import cv2
import numpy as np
from matplotlib import pyplot as plt
import os

#test change to learn git actions"
#branch testing

#divergent edit -> no conflict

def symmMetric(image):
	height, width = image.shape[:2]

	leftHalf = image[0:height,0:width/2]
	rightHalf = image[0:height,width/2:width]

	leftHeight, leftWidth = leftHalf.shape[:2]
	rightHeight, rightWidth = rightHalf.shape[:2]

	leftHalf,leftCont,leftHier = cv2.findContours(leftHalf, 1, 2)
	rightHalf,rightCont,rightHier = cv2.findContours(rightHalf, 1, 2)


	#finding difference in area for symmetry calculation
	leftCount = 0.0
	rightCount = 0.0

	for x in range(leftWidth):
		for y in range(leftHeight):
			if leftHalf.item(y,x) == 255:
				leftCount = leftCount + 1

	for x in range(rightWidth):
		for y in range(rightHeight):
			if rightHalf.item(y,x) == 255:
				rightCount = rightCount + 1

	if rightCount == 0.0:
		percentDiffernce = 100.0
	else:
		percentDifference = (abs(leftCount - rightCount)/rightCount)*100.0

	return percentDifference

def borderIrregularity(image):
	im6,contours,hierarchy = cv2.findContours(image, 1, 2)
	#print len(contours)

	#mask2 = np.ones(im6.shape[:2], dtype="uint8") * 255
	#for c in contours:
	#	print cv2.contourArea(c)

	#test = contours[1]
	#cv2.drawContours(mask2, test, -1, 0, -1)
	#cv2.imshow("mask", mask2)
	#cv2.waitKey(0)
	#cv2.destroyAllWindows()

	#use convex irregularity to find border irregulairty metric
	contourLen = len(contours)
	cnt = contours[contourLen-1]
	lesionArea = cv2.contourArea(cnt)
	print len(contours)

	hull = cv2.convexHull(cnt,returnPoints = False)
	defects = cv2.convexityDefects(cnt,hull)

	#print len(defects)

	farSum = 0.0
	count = 0

	for i in range(defects.shape[0]):
	    s,e,f,d = defects[i,0]
	    start = tuple(cnt[s][0])
	    end = tuple(cnt[e][0])
	    far = tuple(cnt[f][0])
	    cv2.line(im6,start,end,[0,255,0],2)
	    #cv2.circle(im6,far,5,[0,0,255],-1)
	    farSum = farSum + d/256.0
	    count = count + 1

	im6 = 255 - im6
	cv2.imshow('convex hall', im6)
	farAvg = farSum / count
	borderIrreg = farAvg/lesionArea
	return borderIrreg

def compactIndex(image):
	im6,contours,hierarchy = cv2.findContours(image, 1, 2)
	contourLen = len(contours)
	cnt = contours[contourLen-1]
	#use compact index to find border irregularity metric
	lesionArea = cv2.contourArea(cnt)
	lesionPerim  = cv2.arcLength(cnt,True)

	contIndex = lesionPerim * (lesionPerim) / (4*np.pi*lesionArea)
	return contIndex


def colorVariance(image):
	(means, stds) = cv2.meanStdDev(image)
	return stds

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename),1)
        if img is not None:
            images.append(img)
    return images

def main():
	#img = cv2.imread('mole-malignant-2.jpg',0)
	#imgColor = cv2.imread('mole-malignant-2.jpg',1)
	# Otsu's thresholding after Gaussian filtering
	img = cv2.imread('mole-malignant-2.jpg',0)
	imgColor = cv2.imread('mole-malignant-2.jpg',1)

	cv2.imshow('origianl', imgColor)
	cv2.imwrite("originalMoleOG.jpg",imgColor)
	#grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	blur = cv2.GaussianBlur(img,(5,5),0)
	ret3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

	#cv2.imshow('test', th3)
	th3 = 255 - th3

	im2,contours,hierarchy = cv2.findContours(th3, 1, 2)
	mask = np.ones(im2.shape[:2], dtype="uint8") * 255

	maxContourArea = 0;

	for c in contours:
		area = cv2.contourArea(c)
		if area > maxContourArea:
			maxContourArea = area

	# loop over the contours
	for c in contours:
		# if the contour is bad, draw it on the mask
		area = cv2.contourArea(c)
		#print area
		if area < maxContourArea:
			cv2.drawContours(mask, [c], -1, 0, -1)

	# remove the contours from the image and show the resulting images

	cv2.imshow("unprocessed",im2)
	cv2.imwrite("unprocessedMal.jpg",im2)

	image = cv2.bitwise_and(im2, im2, mask=mask)


	im3,contours,hierarchy = cv2.findContours(image, 1, 2)

	for cnt in contours:
	    cv2.drawContours(im3,[cnt],0,255,-1)

	gray = cv2.bitwise_not(im3)

	cropX = 0
	cropY = 0
	cropW = 0
	cropH = 0

	for c in contours:
		x,y,w,h = cv2.boundingRect(c)
		cv2.rectangle(gray,(x,y),(x+w,y+h),(0,255,0),1)
		cropX = x
		cropY = y
		cropW = w
		cropH = h

	gray = 255 - gray

	boundedImage = gray[cropY+1:cropY+cropH,cropX+1:cropX+cropW]
	originalCrop = imgColor[cropY+1:cropY+cropH,cropX+1:cropX+cropW]
	maskedColor = cv2.bitwise_and(originalCrop,originalCrop, mask=boundedImage)

	symmImg = maskedColor.copy()
	height, width = image.shape[:2]
	cv2.line(symmImg,(0,width/2),(height,width/2),(0,255,0),5)

	symmResult = symmMetric(boundedImage)
	borderIrrResult = borderIrregularity(im3)
	compactIndResult = compactIndex(im3)
	colorVarResult = colorVariance(originalCrop)

	im8,contours,hierarchy = cv2.findContours(boundedImage, 1, 2)

	contour_id = 0
	border_thickness = 5
	border_color = (255, 0, 0)
	cv2.drawContours(im8, contours, -1, 0, border_thickness)
	#cv2.imshow("bORDERS", im8)

	print symmResult
	print borderIrrResult
	print compactIndResult
	print colorVarResult

	#cv2.imshow("bounded image", boundedImage)
	#cv2.imshow('original bounded', maskedColor)
	#cv2.imshow('line symm',symmImg)
	#cv2.imwrite("maskedImageMalOG.jpg", maskedColor)
	#cv2.imwrite("segmentedImageMal.jpg",maskedColor)
	#cv2.imshow('unbounded',gray)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	'''
	stdR = np.asscalar(colorVarResult[0])
	stdG = np.asscalar(colorVarResult[1])
	stdB = np.asscalar(colorVarResult[2])

	featureVector = {}
	featureVector['symmResult'] = symmResult
	featureVector['borderIrrResult'] = borderIrrResult
	featureVector['commpactIndResult'] = compactIndResult
	featureVector['stdR'] = stdR
	featureVector['stdG'] = stdG
	featureVector['stdB'] = stdB

	#featureVector =  np.array([symmResult,borderIrrResult,compactIndResult,stdR,stdG,stdB])
	#print featureVector

	cv2.imshow("bounded image", boundedImage)
	cv2.imshow('original bounded', maskedColor)
	#cv2.imshow("ellipse image", im6)
	#cv2.imshow("left", leftHalf)
	#cv2.imshow("right", rightHalf)
	#cv2.waitKey(0)
	#cv2.destroyAllWindows()
	'''

if __name__ == "__main__": main()
