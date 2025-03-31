import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import cv2
import Utilities


###   Hilfsfunktionen  ###
# Randbehandlung:

### Randbehandlung ###
# Randbehandlung-Funktion
#  Mapping von UI-Wert zu OpenCV-Border-Constant
def map_border_type(border_type_ui):
    mapping = {
        "Spiegeln": cv2.BORDER_REFLECT,
        "Extrapolieren": cv2.BORDER_REPLICATE,
        "Zyklisch": cv2.BORDER_WRAP,
        "Nullen": cv2.BORDER_CONSTANT
    }
    return mapping.get(border_type_ui, cv2.BORDER_REFLECT)  # Fallback: Spiegeln

# Anwendung der Border-Behandlung
def apply_border_handling(img, border_type_ui, border_size=1, constant_value=0):
    """Applies the specified border handling to the image."""
    border_type_cv = map_border_type(border_type_ui)  # Map UI value to OpenCV constant
    print(f"[DEBUG] Applying border handling: {border_type_ui} ({border_type_cv})")
    return cv2.copyMakeBorder(img, border_size, border_size, border_size, border_size,
                              border_type_cv, value=constant_value)

##########################################################################################
# Filter funktionenunter anwednugn der falterung im raum :
##########################################################################################
######################################################################

# apply median filter
# Median Filter
def applyMedianFilter(img, kSize, borderType="spiegeln"):
    print("Applying Median Filter")
    pass
######################################################################

# Moving Average Filter

# 🔹 Kernel erzeugen
def create_moving_average_kernel(kSize):
    if kSize % 2 == 0:
        raise ValueError("Kernel size must be odd.")
    return np.ones((kSize, kSize), dtype=np.float32) / (kSize * kSize)

# 🔹 Anwendung in Spatial Domain
def apply_moving_average_filter(img, kSize=3, border_type_ui="Spiegeln"):
    print(f"[INFO] Moving Average Filter: kSize={kSize}, border='{border_type_ui}'")

    margin = kSize // 2
    img_padded = apply_border_handling(img, border_type_ui, border_size=margin)
    kernel = create_moving_average_kernel(kSize)

    filtered = cv2.filter2D(img_padded, -1, kernel)

    return filtered[margin:-margin, margin:-margin]

##########################################################################

# Gaussian Filter
def createGaussianKernel(kSize, sigma=1.0):
    """Erzeugt einen Gauß-Kernel der Größe kSize x kSize mit Standardabweichung sigma"""
    print(f"Generiere Gauß-Kernel mit kSize={kSize}, sigma={sigma}")
    ax = np.linspace(-(kSize // 2), kSize // 2, kSize)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(- (xx**2 + yy**2) / (2 * sigma**2))
    kernel /= np.sum(kernel)  # Normalisierung
    print("Gauß-Kernel erzeugt:\n", kernel)
    return kernel

def applyGaussianFilter(img, kSize=5, sigma=1.0, border_type="Reflect"):
    """Applies a Gaussian filter with the specified edge handling."""
    print(f"[DEBUG] Applying Gaussian Filter with edge handling: {border_type}")

    # Apply border handling
    img_with_border = apply_border_handling(img, border_type)  # Corrected function name

    # Create the Gaussian kernel
    kernel = createGaussianKernel(kSize, sigma)

    # Perform the convolution
    filtered_img_with_border = cv2.filter2D(img_with_border, -1, kernel, borderType=cv2.BORDER_DEFAULT)

    # Remove the border to restore the original dimensions
    filtered_img = filtered_img_with_border[1:-1, 1:-1]

    print(f"[DEBUG] Gaussian Filter applied successfully with edge handling: {border_type}")
    return filtered_img

######################################################################
# Sobel Filter
def createSobelXKernel():
    print("Sobel X Kernel erstellt")
    pass

def createSobelYKernel():
    print("Sobel Y Kernel erstellt")
    pass

def applySobelFilter(img, direction="x", borderType=cv2.BORDER_REFLECT):
    print(f"Applying Sobel Filter in {direction}-Richtung")
    pass

def applyKernelInSpatialDomain(img, kernel, borderType=cv2.BORDER_REFLECT):
    print("Applying kernel in spatial domain")
    pass

######################################################################
# Extra: create an integral image of the given image
def createIntegralImage(img):
    print("Integralbild erstellt")
    integral_image = img.copy()
    return integral_image

def applyMovingAverageFilterWithIntegralImage(img, kSize):
    print(f"Applying Moving Average Filter mit Integralbild, kSize={kSize}")
    filtered_img = img.copy()
    return filtered_img

def applyMovingAverageFilterWithSeperatedKernels(img, kSize):
    print(f"Applying Moving Average Filter mit separierten Kernen, kSize={kSize}")
    filtered_img = img.copy()
    return filtered_img

def run_runtime_evaluation(img):
    print("Runtime Evaluation gestartet")
    pass
