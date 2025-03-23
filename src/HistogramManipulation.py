import cv2
import numpy as np
import matplotlib.pyplot as plt

# Task 1
# function to stretch an image
def stretchHistogram(img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # In Graustufen umwandeln
    histogram = calculateHistogram(gray_img)
    minVal, maxVal = findMinMaxPos(histogram)
    LUT = np.interp(np.arange(256), [minVal, maxVal], [0, 255]).astype(np.uint8)
    return applyLUT(img, LUT)

# Task 2
# function to equalize an image
def equalizeHistogram(img):
    histogram = calculateHistogram(img)
    cdf = np.cumsum(histogram)  # Kumulative Verteilung
    cdf_normalized = ((cdf - cdf.min()) / (cdf.max() - cdf.min()) * 255).astype(np.uint8)
    return applyLUT(img, cdf_normalized)

#Hilfsfunktion
# function to apply a look-up table onto an image
def applyLUT(img, LUT):
    return cv2.LUT(img, LUT)

def convertToGrayScale(img):
    print("Convert To grayScale")
    result = img.copy()
    return result

# Hilfsfunktion
# function to find the minimum an maximum in a histogram
def findMinMaxPos(histogram):
    minPos = np.min(np.where(histogram > 0))
    maxPos = np.max(np.where(histogram > 0))
    return minPos, maxPos

# Hilfsfunktion
# function to create a vector containing the histogram
def calculateHistogram(img, nrBins=256):
    histogram = np.zeros(nrBins, dtype=int)
    for pixel in img.flatten():
        histogram[pixel] += 1
    return histogram


def apply_log(img):
    LUT = (np.log1p(np.arange(256)) / np.log1p(255) * 255).astype(np.uint8)
    return applyLUT(img, LUT)

def apply_exp(img):
    LUT = ((np.exp(np.arange(256) / 255.0) - 1) / (np.exp(1) - 1) * 255).astype(np.uint8)
    return applyLUT(img, LUT)

def apply_inverse(img):
    LUT = np.array([255 - i for i in range(256)], dtype=np.uint8)
    return applyLUT(img, LUT)

def apply_threshold(img, threshold):
    return np.where(img >= threshold, 255, 0).astype(np.uint8)