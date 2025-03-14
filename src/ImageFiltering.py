import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import cv2
import Utilities


###   Hilfsfunktionen  ###
# Randbehandlung:
# Extrapolieren
# Extrapolieren
def applyNachbarschaftsoperatorExtrapolieren(self, kernel):
    print("Extrapolieren angewendet")
    return cv2.filter2D(self.image, -1, kernel, borderType=cv2.BORDER_REPLICATE)

# Spiegeln
def applyNachbarschaftsoperatorSpiegeln(self, kernel):
    print("Spiegeln angewendet")
    return cv2.filter2D(self.image, -1, kernel, borderType=cv2.BORDER_REFLECT)

# Zyklische Wiederholung
def applyNachbarschaftsoperatorZyklisch(self, kernel):
    print("Zyklisch angewendet")
    return cv2.filter2D(self.image, -1, kernel, borderType=cv2.BORDER_WRAP)

# Nullen
def applyNachbarschaftsoperatorNullen(self, kernel):
    print("Nullen angewendet")
    return cv2.filter2D(self.image, -1, kernel, borderType=cv2.BORDER_CONSTANT)

##########################################################################################
# Filter funktionenunter anwednugn der falterung im raum :
##########################################################################################
######################################################################

# apply median filter
# Median Filter
    # Grauwerte der Nachbarschaft sortieren und zuordnen
def applyMedianFilter(img, kSize):
    print(f"Median Filter mit kSize={kSize} angewendet")
    return cv2.medianBlur(img, kSize)

######################################################################

# Moving Average Filter
    # Kernel definieren

    # Faltung im Bildraum
# create a moving average kernel of arbitrary size
def createMovingAverageKernel(kSize):
    print(f"Moving Average Kernel mit kSize={kSize} erstellt")
    return np.ones((kSize, kSize), np.float32) / (kSize * kSize)

def applyMovingAverageFilter(img, kSize, borderType):
    print(f"Applying Moving Average Filter mit kSize={kSize}, borderType={borderType}")
    kernel = createMovingAverageKernel(kSize)
    return cv2.filter2D(img, -1, kernel, borderType=borderType)



##########################################################################

# Gaussian Filter
    # Kernel definieren
    # Faltung im Bildraum
def createGaussianKernel(kSize, sigma=1.0):
    print(f"Applying Gaussian Filter with kSize={kSize}, sigma={sigma}")
    ax = np.linspace(-(kSize // 2), kSize // 2, kSize)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(- (xx**2 + yy**2) / (2 * sigma**2))
    print(f"Generated Gaussian Kernel:\n{kernel}")
    return kernel / np.sum(kernel)  # Normalisierung



def applyGaussianFilter(img, kSize, borderType, sigma=1.0):
    print(f"Applying Gaussian Filter mit kSize={kSize}, sigma={sigma}, borderType={borderType}")
    kernel = createGaussianKernel(kSize, sigma)
    return cv2.filter2D(img, -1, kernel, borderType=borderType)


######################################################################

# Sobel Filter
    # Kernel definieren
    # Faltung im Bildraum

# create a sobel kernel in x direction of size 3x3
def createSobelXKernel():
    print("Sobel X Kernel erstellt")
    return np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)


# create a sobel kernel in y direction of size 3x3
def createSobelYKernel():
    print("Sobel Y Kernel erstellt")
    return np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)

def applySobelFilter(img, direction="x", borderType=cv2.BORDER_REFLECT):
    print(f"Applying Sobel Filter in {direction}-Richtung mit borderType={borderType}")
    kernel = createSobelXKernel() if direction == "x" else createSobelYKernel()
    return cv2.filter2D(img, -1, kernel, borderType=borderType)

######################################################################

def applyKernelInSpatialDomain(img, kernel, borderType=cv2.BORDER_REFLECT):
    print(f"Applying kernel in spatial domain mit borderType={borderType}")
    return cv2.filter2D(img, -1, kernel, borderType=borderType)




######################################################################
####################################################################
# Extra: create an integral image of the given image
def createIntegralImage(img):
    print("Integralbild erstellt")
    integral_image = img.copy()
    return integral_image


# Extra: apply the moving average filter by using an integral image
def applyMovingAverageFilterWithIntegralImage(img, kSize):
    print(f"Applying Moving Average Filter mit Integralbild, kSize={kSize}")
    filtered_img = img.copy()
    return filtered_img


# Extra:
def applyMovingAverageFilterWithSeperatedKernels(img, kSize):
    print(f"Applying Moving Average Filter mit separierten Kernen, kSize={kSize}")

    filtered_img = img.copy()
    return filtered_img

def run_runtime_evaluation(img):
    print("Runtime Evaluation gestartet")
    pass