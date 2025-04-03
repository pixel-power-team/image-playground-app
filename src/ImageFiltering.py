import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import cv2
import Utilities

####################################################################################################
# zusätzliche Funktionen
####################################################################################################

def map_border_type(border_type_ui):
    mapping = {
        "Spiegeln": cv2.BORDER_REFLECT,
        "Extrapolieren": cv2.BORDER_REPLICATE,
        "Zyklisch": cv2.BORDER_WRAP,
        "Nullen": cv2.BORDER_CONSTANT
    }
    return mapping.get(border_type_ui, cv2.BORDER_REFLECT)

def apply_border_handling(img, border_type_ui, border_size=1, constant_value=0):
    border_type_cv = map_border_type(border_type_ui)
    print(f"[DEBUG] Applying border handling: {border_type_ui} ({border_type_cv})")
    return cv2.copyMakeBorder(img, border_size, border_size, border_size, border_size,
                              border_type_cv, value=constant_value)

def custom_convolution(img, kernel):
    """
    Custom convolution function to replace OpenCV's cv2.filter2D.
    This manually applies the kernel to the image by iterating over each pixel.
    """
    img_h, img_w = img.shape
    k_h, k_w = kernel.shape
    result = np.zeros_like(img, dtype=np.float32)
    offset_h = k_h // 2
    offset_w = k_w // 2
    for i in range(offset_h, img_h - offset_h):
        for j in range(offset_w, img_w - offset_w):
            region = img[i - offset_h:i + offset_h + 1, j - offset_w:j + offset_w + 1]
            result[i, j] = np.sum(region * kernel)
    return result

def custom_median_filter(img, kSize):
    """
    Custom median filter function to replace OpenCV's cv2.medianBlur.
    This manually calculates the median value for each pixel's neighborhood.
    """
    img_h, img_w = img.shape
    result = np.zeros_like(img, dtype=img.dtype)
    offset = kSize // 2
    for i in range(offset, img_h - offset):
        for j in range(offset, img_w - offset):
            region = img[i - offset:i + offset + 1, j - offset:j + offset + 1]
            result[i, j] = np.median(region)
    return result

####################################################################################################
# Median Filter
####################################################################################################

def applyMedianFilter(img, kSize, border_type_ui="Spiegeln"):
    """
    Applies a median filter using the custom_median_filter function.
    This avoids using OpenCV's cv2.medianBlur.
    """
    print(f"[INFO] Applying Median Filter with kSize={kSize}, border='{border_type_ui}'")
    if kSize % 2 == 0 or kSize <= 0:
        raise ValueError("Kernel size must be a positive odd integer.")
    margin = kSize // 2
    img_padded = apply_border_handling(img, border_type_ui, border_size=margin)
    filtered = custom_median_filter(img_padded, kSize)
    result = filtered[margin:-margin, margin:-margin]
    print(f"[INFO] Median Filter applied successfully.")
    return result

####################################################################################################
# Moving Average Filter
####################################################################################################

def create_moving_average_kernel(kSize):
    if kSize % 2 == 0:
        raise ValueError("Kernel size must be odd.")
    return np.ones((kSize, kSize), dtype=np.float32) / (kSize * kSize)

def apply_moving_average_filter(img, kSize=3, border_type_ui="Spiegeln"):
    """
    Applies a moving average filter using the custom_convolution function.
    This avoids using OpenCV's cv2.filter2D.
    """
    print(f"[INFO] Moving Average Filter: kSize={kSize}, border='{border_type_ui}'")
    margin = kSize // 2
    img_padded = apply_border_handling(img, border_type_ui, border_size=margin)
    kernel = create_moving_average_kernel(kSize)
    filtered = custom_convolution(img_padded, kernel)
    return filtered[margin:-margin, margin:-margin]

####################################################################################################
# Gaussian Filter
####################################################################################################

def createGaussianKernel(kSize, sigma=1.0):
    print(f"Generiere Gau\u00df-Kernel mit kSize={kSize}, sigma={sigma}")
    ax = np.linspace(-(kSize // 2), kSize // 2, kSize)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(- (xx**2 + yy**2) / (2 * sigma**2))
    kernel /= np.sum(kernel)
    print("Gau\u00df-Kernel erzeugt:\n", kernel)
    return kernel

def applyGaussianFilter(img, kSize=5, sigma=1.0, border_type="Reflect"):
    """
    Applies a Gaussian filter using the custom_convolution function.
    This avoids using OpenCV's cv2.filter2D.
    """
    print(f"[DEBUG] Applying Gaussian Filter with edge handling: {border_type}")
    margin = kSize // 2
    img_with_border = apply_border_handling(img, border_type, border_size=margin)
    kernel = createGaussianKernel(kSize, sigma)
    filtered_img_with_border = custom_convolution(img_with_border, kernel)
    filtered_img = filtered_img_with_border[margin:-margin, margin:-margin]
    print(f"[DEBUG] Gaussian Filter applied successfully with edge handling: {border_type}")
    return filtered_img

####################################################################################################
# Sobel Filter
####################################################################################################

def createSobelXKernel():
    kernel = np.array([[-1, 0, 1],
                       [-2, 0, 2],
                       [-1, 0, 1]], dtype=np.float32)
    print("Sobel X Kernel erstellt:\n", kernel)
    return kernel

def createSobelYKernel():
    kernel = np.array([[-1, -2, -1],
                       [0,  0,  0],
                       [1,  2,  1]], dtype=np.float32)
    print("Sobel Y Kernel erstellt:\n", kernel)
    return kernel

def applySobelFilter(img, direction="x", border_type_ui="Reflect"):
    """
    Applies the Sobel filter using the custom_convolution function.
    This avoids using OpenCV's cv2.filter2D.
    """
    print(f"[INFO] Applying Sobel Filter in {direction}-direction with border='{border_type_ui}'")
    if direction == "x":
        kernel = createSobelXKernel()
    elif direction == "y":
        kernel = createSobelYKernel()
    else:
        raise ValueError("Invalid direction. Use 'x' or 'y'.")
    margin = kernel.shape[0] // 2
    img_padded = apply_border_handling(img, border_type_ui, border_size=margin)
    filtered = custom_convolution(img_padded, kernel)
    result = filtered[margin:-margin, margin:-margin]
    print(f"[INFO] Sobel Filter applied successfully in {direction}-direction.")
    return result

####################################################################################################
# Custom Kernel Application
####################################################################################################

def applyKernelInSpatialDomain(img, kernel, border_type_ui="Reflect"):
    """
    Applies a custom kernel to an image in the spatial domain using manual convolution.
    This function avoids using OpenCV's cv2.filter2D and performs the convolution manually.

    Args:
        img (numpy.ndarray): Input grayscale image.
        kernel (numpy.ndarray): The kernel to apply to the image.
        border_type_ui (str): Border handling method ("Spiegeln", "Extrapolieren", "Zyklisch", "Nullen").

    Returns:
        numpy.ndarray: The filtered image after applying the kernel.
    """
    print(f"[INFO] Applying custom kernel with border='{border_type_ui}'")

    # Calculate the margin (padding size) based on the kernel size
    margin = kernel.shape[0] // 2

    # Apply border handling to pad the image
    img_padded = apply_border_handling(img, border_type_ui, border_size=margin)

    # Perform manual convolution using the custom_convolution function
    filtered = custom_convolution(img_padded, kernel)

    # Remove the border to restore the original image dimensions
    result = filtered[margin:-margin, margin:-margin]

    print(f"[INFO] Custom kernel applied successfully.")
    return result

###########################################################################
# runtime evaluation
####################################################################################################

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