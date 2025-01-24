import cv2
import numpy as np

# Example for basic pixel based image manipulation:
# https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_core/py_basic_ops/py_basic_ops.html

# Task 1:   Implement some kind of noticeable image manipulation in this function
#           e.g. channel manipulation, filter you already know, drawings on the image etc.
def myFirstImageManipulation(img):
    result = img.copy()
    return result



# Task 2:   Return the basic image properties to the console:
#           width, height,
#           the color of the first pixel of the image,
#           Color of the first pixel in the second row
#           Color of the first pixel in the second column
#           This function should work for images with three channels

def imageSize(img):
    return [1,1]

def getPixelColor(img):
    return [[0,0], [255,255]]

# Task 3:   Separate the given channels of a colour image in this function and return it as separate image
#           the separate image need three channels
#
def returnChannel(img, channel):
    result = img.copy()
    return result
