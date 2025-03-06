import cv2
import numpy as np

# Example for basic pixel based image manipulation:
# https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_core/py_basic_ops/py_basic_ops.html

# Task 1:   Implement some kind of noticeable image manipulation in this function
#           e.g. channel manipulation, filter you already know, drawings on the image etc.
def myFirstImageManipulation(img):
    # Copy the image to avoid changing the original image
    # This could be removed, depending on the use case
    result = img.copy() 
    
    # Draw a white rectangle and a black circle on the image
    #                      start point, end point, color, thickness
    cv2.rectangle(result, (10, 10), (100, 100), (255, 255, 255), 3)
    cv2.circle(result, (200, 200), 20, (0, 0, 0), 2)
    
    # Add text to the image
    #                 text, start point, font, font size, color, thickness
    cv2.putText(result, "Hello Pixel-Power!", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Apply a blurry filter to the image
    #                                   kernel size, sigma
    result = cv2.GaussianBlur(result, (15, 15), 0)
    
    return result



# Task 2:   Return the basic image properties to the console:
#           width, height,
#           the color of the first pixel of the image,
#           Color of the first pixel in the second row
#           Color of the first pixel in the second column
#           This function should work for images with three channels

def imageSize(img):
    width, height = img.shape[:2]
    print(f"Width: {width}, Height: {height}")
    print(f"Color of the first pixel: {img[0][0]}")
    print(f"Color of the first pixel in the second row: {img[1][0]}")
    print(f"Color of the first pixel in the second column: {img[0][1]}")
    return [width, height]

def getPixelColor(img):
    return [[0,0], [255,255]]

# Task 3:   Separate the given channels of a colour image in this function and return it as separate image
#           the separate image need three channels
#
def returnChannel(img, channel):
    # split channels
    r, g, b = cv2.split(img)
    match channel:
        case 0:
            return cv2.merge((b, np.zeros_like(g), np.zeros_like(r)))
        case 1:
            return cv2.merge((np.zeros_like(b), g, np.zeros_like(r)))
        case 2:
            return cv2.merge((np.zeros_like(b), np.zeros_like(g), r))
        case _:
            return img
    
