import time

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
# to speed up further calculations
def createIntegralImage(img: np.ndarray):
    """
    Computes the integral image of the input image.

    Args:
        img (numpy.ndarray): Grayscale or single-channel image.

    Returns:
        numpy.ndarray: Integral image.
    """
    # Ensure the image is in float format to avoid overflow issues
    img = img.astype(np.float32)

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
        img (numpy.ndarray): Grayscale or single-channel image.
        kSize (int): Kernel size (w x w).

    Returns:
        numpy.ndarray: Filtered image.
    """
    # Ensure kSize is odd to have a centered kernel 
    if kSize % 2 == 0:
        raise ValueError("Kernel size must be odd.")

    # Compute the integral image
    integral_image = createIntegralImage(img)

    # Determine padding size
    # e.g. kSize = 3, pad = 1
    pad = kSize // 2  # Half of kernel size (floor division) 

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
def applyMovingAverageFilterWithSeperatedKernels(img, kSize):
    filtered_img = img.copy()
    return filtered_img


def run_runtime_evaluation(img: np.ndarray):
    """
    Evaluates and compares the runtime of different Moving Average filter implementations.

    Args:
        img (numpy.ndarray): Grayscale or single-channel image.

    Plots the execution time for different kernel sizes (w).
    """
    kernel_sizes = list(range(start=3, stop=16, step=2))  # w = 3, 5, 7, ..., 15
    methods = {
        "Naive Convolution": applyKernelInSpatialDomain,
        "Separable Kernels": applyMovingAverageFilterWithSeperatedKernels,
        "Integral Image": applyMovingAverageFilterWithIntegralImage
    }

    runtimes = {method: [] for method in methods}

    for w in kernel_sizes:
        for method_name, method in methods.items():
            start_time = time.perf_counter()
            method(img, w)  # Apply the filter
            end_time = time.perf_counter()

            runtime = end_time - start_time
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
