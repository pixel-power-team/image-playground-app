import cv2
import numpy as np

# Task 1
# function to stretch an image
def stretchHistogram(img):
    gray_img = convertToGrayScale(img) # In Graustufen umwandeln
    histogram = calculateHistogram(gray_img)
    minVal, maxVal = findMinMaxPos(histogram)
    LUT = np.interp(np.arange(256), [minVal, maxVal], [0, 255]).astype(np.uint8)
    return applyLUT(gray_img, LUT)

# Task 2
# function to equalize an image
def equalizeHistogram(img):
    gray_img = convertToGrayScale(img)
    histogram = calculateHistogram(gray_img)
    cdf = np.cumsum(histogram)  # Kumulative Verteilung
    cdf_normalized = ((cdf - cdf.min()) / (cdf.max() - cdf.min()) * 255).astype(np.uint8)
    return applyLUT(gray_img , cdf_normalized)

#Hilfsfunktion
# function to apply a look-up table onto an image
def applyLUT(img, LUT):
    return cv2.LUT(img, LUT)


def convertToGrayScale(img):
    """
   Wandelt ein Farbbild manuell in ein Graustufenbild um, basierend auf der menschlichen Helligkeitswahrnehmung:
       Grauwert = 0.299 * R + 0.587 * G + 0.114 * B

   Zusätzlich wird das erzeugte Graubild am Ende in ein 3-kanaliges BGR-Bild umgewandelt, bei dem alle Farbkanäle
   denselben Grauwert tragen.
   Denn bei der Darstellung wird standardmäßig ein Bild mit 3 Farbkanälen (shape = H x W x 3) erwartet.
   Wenn nur ein Graustufenbild (shape = H x W) geliefert wird, kann es dort zu Fehlern kommen, weil
   erwartet wird, dass man „Höhe, Breite und Kanäle“ entpacken kann. 
   Deshalb erzeugen wir aus dem Graubild ein 3-kanaliges Bild mit identischen Grauwerten in jedem Kanal.
   """
    print("Convert to grayscale (manual)")

    # Wenn Bild schon Graustufenbild ist → Umwandlung in 3-Kanal
    if len(img.shape) == 2:
        print("Image is already grayscale. Expanding to 3-channel.")
        height, width = img.shape
        gray_image_3ch = np.zeros((height, width, 3), dtype=np.uint8)
        for y in range(height):
            for x in range(width):
                gray = img[y, x]
                gray_image_3ch[y, x] = (gray, gray, gray)
        return gray_image_3ch

    # Falls Farb-Bild: manuelle Grauwertberechnung
    height, width = img.shape[:2]
    gray_image = np.zeros((height, width), dtype=np.uint8)

    for y in range(height):
        for x in range(width):
            blue, green, red = img[y, x]
            gray = int(0.114 * blue + 0.587 * green + 0.299 * red)
            gray_image[y, x] = gray

    # Graubild manuell auf 3 Kanäle erweitern
    gray_image_3ch = np.zeros((height, width, 3), dtype=np.uint8)
    for y in range(height):
        for x in range(width):
            g = gray_image[y, x]
            gray_image_3ch[y, x] = (g, g, g)

    return gray_image_3ch


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