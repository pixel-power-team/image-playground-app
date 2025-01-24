import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import cv2
import Utilities


# apply median filter
def applyMedianFilter(img, kSize):
    filtered_img = img.copy()
    return filtered_img


# create a moving average kernel of arbitrary size
def createMovingAverageKernel(kSize):
    kernel = np.zeros((kSize, kSize))
    return kernel


def gaussian( x, y, sigmaX, sigmaY, meanX, meanY):
    result = 1
    return result


# create a gaussian kernel of arbitrary size
def createGaussianKernel(kSize, sigma=None):
    kernel = np.zeros((kSize, kSize))
    return kernel


# create a sobel kernel in x direction of size 3x3
def createSobelXKernel():
    kernel = np.zeros((3, 3))
    return kernel

# create a sobel kernel in y direction of size 3x3
def createSobelYKernel():
    kernel = np.zeros((3, 3))
    return kernel


def applyKernelInSpatialDomain(img, kernel):
    filtered_img = img.copy()
    return filtered_img


# Extra: create an integral image of the given image
def createIntegralImage(img):
    integral_image = img.copy()
    return integral_image


# Extra: apply the moving average filter by using an integral image
def applyMovingAverageFilterWithIntegralImage(img, kSize):
    filtered_img = img.copy()
    return filtered_img


# Extra:
def applyMovingAverageFilterWithSeperatedKernels(img, kSize):
    filtered_img = img.copy()
    return filtered_img

def run_runtime_evaluation(img):
    pass