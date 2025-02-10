import cv2
import numpy as np
import Utilities

# Task 1
# function to stretch an image
def stretchHistogram(img):
    result = img.copy()
    histogram = calculateHistogram(result,256)
    minPos, maxPos = findMinMaxPos(histogram)
    stretchedImg = ((result -minPos)/(maxPos-minPos) *(256-1))
    return  stretchedImg.astype(np.uint8)

# Task 2
# function to equalize an image
def equalizeHistogram(img):
    result = img.copy()
    return result

#Hilfsfunktion
# function to apply a look-up table onto an image
def applyLUT(img, LUT):
    result = img.copy()
    return cv2.LUT(result, LUT)

# Hilfsfunktion
# function to find the minimum an maximum in a histogram
def findMinMaxPos(histogram):
    minPos = 0
    maxPos = 255
    return minPos, maxPos

# Hilfsfunktion
# function to create a vector containing the histogram
def calculateHistogram(img, nrBins):
    # create histogram vector
    histogram = np.zeros([nrBins], dtype=int)
    return histogram

def apply_log(img):
    result = img.copy()
    return result

def apply_exp(img):
    result = img.copy()
    return result

def apply_inverse(img):
    result = img.copy()
    return result

def apply_threshold(img, threshold):
    result = img.copy()
    return result
