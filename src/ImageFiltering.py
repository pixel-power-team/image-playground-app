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
def applyMedianFilter(img, kSize, border_type_ui="Spiegeln"):
    """
    Applies a median filter to the image with the specified border handling.
    Args:
        img (numpy.ndarray): Input image (grayscale).
        kSize (int): Kernel size (must be odd).
        border_type_ui (str): Border handling method ("Spiegeln", "Extrapolieren", "Zyklisch", "Nullen").
    Returns:
        numpy.ndarray: Image with the median filter applied.
    """
    print(f"[INFO] Applying Median Filter with kSize={kSize}, border='{border_type_ui}'")

    # Validate kernel size
    if kSize % 2 == 0 or kSize <= 0:
        raise ValueError("Kernel size must be a positive odd integer.")

    # Apply border handling
    margin = kSize // 2  # Calculate the margin based on the kernel size
    img_padded = apply_border_handling(img, border_type_ui, border_size=margin)

    # Apply the median filter
    filtered = cv2.medianBlur(img_padded, kSize)

    # Remove the border to restore the original dimensions
    result = filtered[margin:-margin, margin:-margin]

    print(f"[INFO] Median Filter applied successfully.")
    return result

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
    """Creates the Sobel kernel for the x direction."""
    kernel = np.array([[-1, 0, 1],
                       [-2, 0, 2],
                       [-1, 0, 1]], dtype=np.float32)
    print("Sobel X Kernel erstellt:\n", kernel)
    return kernel

def createSobelYKernel():
    """Creates the Sobel kernel for the y direction."""
    kernel = np.array([[-1, -2, -1],
                       [0,  0,  0],
                       [1,  2,  1]], dtype=np.float32)
    print("Sobel Y Kernel erstellt:\n", kernel)
    return kernel

def applySobelFilter(img, direction="x", border_type_ui="Reflect"):
    """
    Applies the Sobel filter in the specified direction with the given border handling.
    Args:
        img (numpy.ndarray): Input grayscale image.
        direction (str): "x" for horizontal gradients, "y" for vertical gradients.
        border_type_ui (str): Border handling method ("Spiegeln", "Extrapolieren", "Zyklisch", "Nullen").
    Returns:
        numpy.ndarray: Image with Sobel filter applied (raw gradient values).
    """
    print(f"[INFO] Applying Sobel Filter in {direction}-direction with border='{border_type_ui}'")

    # Select the appropriate Sobel kernel
    if direction == "x":
        kernel = createSobelXKernel()
    elif direction == "y":
        kernel = createSobelYKernel()
    else:
        raise ValueError("Invalid direction. Use 'x' or 'y'.")

    # Apply border handling
    margin = kernel.shape[0] // 2  # Calculate the margin based on the kernel size
    img_padded = apply_border_handling(img, border_type_ui, border_size=margin)

    # Apply the Sobel filter using convolution
    filtered = cv2.filter2D(img_padded, cv2.CV_32F, kernel)  # Use CV_32F to preserve gradient values

    # Remove the border to restore the original dimensions
    result = filtered[margin:-margin, margin:-margin]

    print(f"[INFO] Sobel Filter applied successfully in {direction}-direction.")
    return result

def applyKernelInSpatialDomain(img, kernel, border_type_ui="Reflect"):
    """
    Applies a custom kernel to an image in the spatial domain with the specified border handling.
    Args:
        img (numpy.ndarray): Input grayscale image.
        kernel (numpy.ndarray): Kernel to apply.
        border_type_ui (str): Border handling method ("Spiegeln", "Extrapolieren", "Zyklisch", "Nullen").
    Returns:
        numpy.ndarray: Image with the kernel applied.
    """
    print(f"[INFO] Applying custom kernel with border='{border_type_ui}'")

    # Apply border handling
    margin = kernel.shape[0] // 2  # Calculate the margin based on the kernel size
    img_padded = apply_border_handling(img, border_type_ui, border_size=margin)

    # Apply the kernel using convolution
    filtered = cv2.filter2D(img_padded, -1, kernel)

    # Remove the border to restore the original dimensions
    result = filtered[margin:-margin, margin:-margin]

    print(f"[INFO] Custom kernel applied successfully.")
    return result

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
