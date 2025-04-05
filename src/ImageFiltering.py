import time
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
    # Ensure the image is single-channel grayscale
    if len(img.shape) > 2:
        print("[DEBUG] Converting multi-channel image to grayscale for convolution.")
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

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

def applyGaussianFilter(img, kSize=5, sigma=1.0, border_type="Spiegeln"):
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

def applySobelFilter(img, direction="x", border_type_ui="Spiegeln"):
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

def applyKernelInSpatialDomain(img, kernel_size, border_type_ui):
    """
    Applies a custom kernel to an image in the spatial domain using manual convolution.
    This function avoids using OpenCV's cv2.filter2D and performs the convolution manually.

    Args:
        img (numpy.ndarray): Input grayscale image.
        kernel_size (int): The size of the kernel (must be odd).
        border_type_ui (str): Border handling method ("Spiegeln", "Extrapolieren", "Zyklisch", "Nullen").

    Returns:
        numpy.ndarray: The filtered image after applying the kernel.
    """
    print(f"[INFO] Applying custom kernel with kernel_size={kernel_size}, border='{border_type_ui}'")

    # Ensure kernel_size is odd
    if kernel_size % 2 == 0:
        raise ValueError("Kernel size must be an odd number.")

    # Create a moving average kernel
    kernel = np.ones((kernel_size, kernel_size), dtype=np.float32) / (kernel_size * kernel_size)

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

########################################################################################################
#Zusatzaufgabe:
######################################################################################################
# Extra: create an integral image of the given image
# to speed up further calculations
def createIntegralImage(img: np.ndarray):
    """
    Computes the integral image of the input image.

    Args:
        img (numpy.ndarray): Grayscale or single-channel image.

    Returns:
        numpy.ndarray: Integral image.
    """
    # Create an array of the same size as the image, initialized with zeros
    integral_image = np.zeros_like(img, dtype=np.float32)

    # Compute the integral image
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            integral_image[i, j] = img[i, j]
            if i > 0:
                # Add the value above the current pixel
                integral_image[i, j] += integral_image[i - 1, j]
            if j > 0:
                # Add the value to the left of the current pixel
                integral_image[i, j] += integral_image[i, j - 1]
            if i > 0 and j > 0:
                # Subtract the value of the pixel above left (schräg links darüber)
                integral_image[i, j] -= integral_image[
                    i - 1, j - 1]

    return integral_image

# Extra: apply the moving average filter by using an integral image
def applyMovingAverageFilterWithIntegralImage(img: np.ndarray, kSize: int):
    """
    Applies a moving average filter using an integral image for efficiency.

    Args:
        img (numpy.ndarray): Image to filter.
        kSize (int): Kernel size (w x w).

    Returns:
        numpy.ndarray: Filtered image.
    """
    img = Utilities.ensure_one_channel_grayscale_image(img)

    # Ensure kSize is odd to have a centered kernel
    if kSize % 2 == 0:
        raise ValueError("Kernel size must be odd.")

    # Compute the integral image
    integral_image = createIntegralImage(img)

    # Determine padding size
    # floor division e.g. kSize = 3, pad = 1
    pad = kSize // 2  # Half of kernel size 

    # Get image dimensions
    h, w = img.shape

    # Create output image
    filtered_img = np.zeros_like(img, dtype=np.float32)

    # Compute the moving average using the integral image
    for i in range(h):
        for j in range(w):
            # Define top-left and bottom-right corners of the window
            # Ensure the window is within the image boundaries by using max and min
            x1, y1 = max(0, i - pad), max(0, j - pad) # (x1, y1) = top-left corner
            x2, y2 = min(h - 1, i + pad), min(w - 1, j + pad) # (x2, y2) = bottom-right corner

            # Sum over the rectangular region using the integral image
            # This only requires 4 lookups in the integral image
            region_sum = integral_image[x2, y2]
            # Subtract the values outside the region
            if x1 > 0:
                region_sum -= integral_image[x1 - 1, y2] # Subtract the top region
            if y1 > 0:
                region_sum -= integral_image[x2, y1 - 1] # Subtract the left region
            if x1 > 0 and y1 > 0:
                region_sum += integral_image[
                    x1 - 1, y1 - 1]  # Add the top-left region because it was subtracted twice

            # Compute the mean value in the window
            num_pixels = (x2 - x1 + 1) * (y2 - y1 + 1) # Number of pixels in the window
            filtered_img[i, j] = region_sum / num_pixels # Compute the mean

    return filtered_img.astype(np.uint8)  # Convert back to uint8

# Extra:
# 1D convolution along rows and columns
def applyMovingAverageFilterWithSeperatedKernels(img, kSize):
    """
    Applies a moving average filter using separable kernels.

    Args:
        img (numpy.ndarray): Image to filter.
        kSize (int): Kernel size (w x w).

    Returns:
        numpy.ndarray: Filtered image.
    """
    img = Utilities.ensure_one_channel_grayscale_image(img)

    # Ensure kSize is odd to keep symmetry
    if kSize % 2 == 0:
        raise ValueError("Kernel size must be odd.")

    # Step 1: Apply 1D convolution along rows (horizontal pass)
    img_horiz = np.zeros_like(img, dtype=np.float32)
    pad = kSize // 2
    h, w = img.shape

    # Every pixel in the horizontal pass is the mean of a row slice
    for i in range(h):
        for j in range(w):
            x1, x2 = max(0, j - pad), min(w - 1, j + pad) # Slice boundaries
            img_horiz[i, j] = np.mean(img[i, x1:x2 + 1])  # Mean of row slice

    # Step 2: Apply 1D convolution along columns (vertical pass)
    img_blur = np.zeros_like(img_horiz, dtype=np.float32)

    for j in range(w):
        for i in range(h):
            y1, y2 = max(0, i - pad), min(h - 1, i + pad)
            img_blur[i, j] = np.mean(
                img_horiz[y1:y2 + 1, j])  # Mean of column slice

    return img_blur.astype(np.uint8)


def run_runtime_evaluation(img: np.ndarray, border_type_ui="Spiegeln"):
    """
    Evaluates and compares the runtime of different Moving Average filter implementations.

    Args:
        img (numpy.ndarray): Image to filter.
        border_type_ui (str): Border handling method ("Spiegeln", "Extrapolieren", etc.).

    Plots the execution time for different kernel sizes (w).
    """
    # (start=3, stop=16, step=2) => w = 3, 5, 7, ..., 15
    kernel_sizes = list(range(3, 16, 2))
    methods = {
        "Naive Convolution": lambda img, w: applyKernelInSpatialDomain(img, w, border_type_ui),
        "Separable Kernels": applyMovingAverageFilterWithSeperatedKernels,
        "Integral Image": applyMovingAverageFilterWithIntegralImage
    }

    runtimes = {method: [] for method in methods}

    for w in kernel_sizes:
        for method_name, method in methods.items():  # Iterate over the methods
            start_time = time.perf_counter()  # Start the timer
            method(img, w)  # Apply the filter
            end_time = time.perf_counter()  # Stop the timer

            runtime = end_time - start_time  # Compute the runtime
            runtimes[method_name].append(runtime)

    # Plot the results
    plt.figure(figsize=(8, 6))
    for method_name, times in runtimes.items():
        plt.plot(kernel_sizes, times, marker='o', label=method_name)

    plt.xlabel("Kernel Size (w)")
    plt.ylabel("Execution Time (seconds)")
    plt.title("Runtime Comparison of Moving Average Filters")
    plt.legend()
    plt.grid()
    plt.show()