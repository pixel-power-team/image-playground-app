import logging
from PyQt6.QtCore import pyqtSignal, pyqtSlot
from loggers import LogEmitter

import cv2
import numpy as np
import Utilities
import ColorAnalysis as CA
import HistogramManipulation as HM
import AWSRekognition as AI
import ImageInformation as II
import ImageFiltering as IF

class MainController():
    def __init__(self, model):
        super().__init__()
        self._model = model

        self.logger = logging.getLogger()
        self.log_handler = LogEmitter()
        self.logger.addHandler(self.log_handler)

        # Set the log level to INFO
        self.logger.setLevel(logging.INFO)

    def test_function(self):
        print("Test function")
        pass

    def loadImage(self, str):
        img = cv2.imread(str, 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self._model.input_image = img
        self.logger.info('Image loaded: '+str)

    def saveImage(self, str):
        img = cv2.cvtColor(self._model.image,
                           cv2.COLOR_BGR2RGB)  # Make sure to convert back to RGB before saving
        cv2.imwrite(str, img)
        self.logger.info('Image written to: '+str)


    def changeImage(self):
        image = np.zeros((256, 256, 3), np.uint8)
        image[0:256 // 2, :] = (255, 0, 0)
        image[256 // 2:256, :] = (0, 0, 255)
        Utilities.resize_image(image, 100)
        self._model.input_image = image

    def set_image_as_input_image(self):
        self._model.input_image = self._model.image.copy()

    def reset_output_image(self):
        self._model.image = self._model.input_image

    def resize_image(self, new_width, new_height):
        self._model.image = Utilities.resize_image(self._model.input_image, new_width, new_height)

    def analyseColors(self, ncluster):
        color_analyzer = CA.ColorAnalysis(self._model.image, ncluster)
        return color_analyzer.dominantColors()

    def create_visual_output(self, colors, width, height_max):
        return CA.create_visual_output(colors, width, height_max)

    def calculate_histogram(self, img):
        return Utilities.calculate_histogram(img)



    def label_image(self):
        ai = AI.AWSRekognition()
        self._model.image = ai.label_image(self._model.input_image)
        self.logger.critical('Labeling successfull')


    #####################################
    # Übung 1
    #####################################

    def get_image_information(self):
        return II.imageSize(self._model.image)

    def get_pixel_information(self):
        return II.getPixelColor(self._model.image)

    def show_channel(self, channel):
        self._model.image = II.returnChannel(self._model.input_image, channel)

    def do_first_image_manipulation(self):
        self._model.image = II.myFirstImageManipulation(self._model.image)

    #####################################
    # Übung 2
    #####################################

    def convert_to_grayscale(self):
        self._model.image = HM.convertToGrayScale(self._model.input_image)
    def stretch_image(self):
        self._model.image = HM.stretchHistogram(self._model.input_image)

    def equalize_image(self):
        self._model.image = HM.equalizeHistogram(self._model.input_image)

    def apply_log(self):
        self._model.image = HM.apply_log(self._model.input_image)

    def apply_exp(self):
        self._model.image = HM.apply_exp(self._model.input_image)

    def apply_inv(self):
        self._model.image = HM.apply_inverse(self._model.input_image)

    def apply_threshold(self, threshold):
        self._model.image = HM.apply_threshold(self._model.input_image, threshold)

    def fill_hist(self):
        self._model.image = HM.fill_hist(self._model.input_image)

    #####################################
    # Übung 3
    #####################################
# Spiegeln is the fallback because it works even if the UI choice breaks or something goes wrong.

    def apply_border_handling(self, border_type):
        self._model.image = IF.applyBorderHandling(self._model.input_image, border_type)

    def apply_gaussian_filter(self, kernel_size, border_type_ui="Spiegeln"):
        print(f"[DEBUG] Applying Gaussian Filter with edge handling: {border_type_ui}")
        # Convert the image to grayscale
        grayscale_image = cv2.cvtColor(self._model.image, cv2.COLOR_RGB2GRAY)

        # Apply the Gaussian filter
        filtered_img = IF.applyGaussianFilter(grayscale_image, kernel_size, border_type=border_type_ui)

        # Ensure the filtered image is in the correct format for display
        filtered_img = np.clip(filtered_img, 0, 255).astype(np.uint8)

        # Convert the filtered image back to 3-channel RGB for display
        self._model.image = cv2.cvtColor(filtered_img, cv2.COLOR_GRAY2RGB)

    def apply_moving_avg_filter(self, kernel_size, border_type_ui="Spiegeln"):
        try:
            if kernel_size % 2 == 0:
                print("[WARNING] Kernel size must be an odd number. Please use an odd value.")

            print(f"[DEBUG] Applying Moving Average Filter with edge handling: {border_type_ui}")
            # Convert the image to grayscale
            grayscale_image = cv2.cvtColor(self._model.image, cv2.COLOR_RGB2GRAY)

            # Apply the moving average filter
            filtered_img = IF.apply_moving_average_filter(grayscale_image, kSize=kernel_size, border_type_ui=border_type_ui)

            # Ensure the filtered image is in the correct format for display
            filtered_img = np.clip(filtered_img, 0, 255).astype(np.uint8)

            # Convert the filtered image back to 3-channel RGB for display
            self._model.image = cv2.cvtColor(filtered_img, cv2.COLOR_GRAY2RGB)
        except Exception as e:
            print(f"[ERROR] Error applying Moving Average Filter: {e}")
            logging.error(f"Error applying Moving Average Filter: {e}", exc_info=True)
            raise

    def apply_moving_avg_filter_integral(self, kernel_size):
        try:
            print(f"[DEBUG] Applying Moving Average Filter with Integral Image: kernel_size={kernel_size}")
            # Convert the image to grayscale
            grayscale_image = cv2.cvtColor(self._model.image, cv2.COLOR_RGB2GRAY)

            # Apply the moving average filter using integral image
            img = IF.applyMovingAverageFilterWithIntegralImage(grayscale_image, kernel_size)

            # Update the model with the filtered grayscale image
            self._model.image = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        except Exception as e:
            print(f"[ERROR] Error applying Moving Average Filter with Integral Image: {e}")
            logging.error(f"Error applying Moving Average Filter with Integral Image: {e}", exc_info=True)
            raise

    def apply_moving_avg_filter_separated(self, kernel_size):
        try:
            print(f"[DEBUG] Applying Moving Average Filter with Separated Kernels: kernel_size={kernel_size}")
            # Convert the image to grayscale
            grayscale_image = cv2.cvtColor(self._model.image, cv2.COLOR_RGB2GRAY)

            # Apply the moving average filter using separated kernels
            img = IF.applyMovingAverageFilterWithSeperatedKernels(grayscale_image, kernel_size)

            # Update the model with the filtered grayscale image
            self._model.image = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        except Exception as e:
            print(f"[ERROR] Error applying Moving Average Filter with Separated Kernels: {e}")
            logging.error(f"Error applying Moving Average Filter with Separated Kernels: {e}", exc_info=True)
            raise

    def apply_median_filter(self, kernel_size, border_type_ui="Spiegeln"):
        # Check if the kernel size is even
        if kernel_size % 2 == 0:
            print("[WARNING] Kernel size must be an odd number. Please use an odd value.")

        print(f"[DEBUG] Applying Median Filter with edge handling: {border_type_ui}")
        # Use the current image (self._model.image) instead of the input image
        grayscale_image = cv2.cvtColor(self._model.image, cv2.COLOR_RGB2GRAY)

        # Apply the median filter
        filtered_img = IF.applyMedianFilter(grayscale_image, kernel_size, border_type_ui=border_type_ui)

        # Convert the filtered image back to 3-channel RGB for display
        self._model.image = cv2.cvtColor(filtered_img, cv2.COLOR_GRAY2RGB)

    def apply_filter_sobelX(self, border_type_ui="Spiegeln"):
        print(f"[DEBUG] Applying Sobel X Filter with edge handling: {border_type_ui}")
        # Convert the image to grayscale
        grayscale_image = cv2.cvtColor(self._model.image, cv2.COLOR_RGB2GRAY)

        # Apply the Sobel X filter
        filtered_img = IF.applySobelFilter(grayscale_image, direction="x", border_type_ui=border_type_ui)

        # Normalize the filtered image to the range 0–255 for display
        normalized_img = cv2.normalize(filtered_img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        # Convert the normalized image back to 3-channel RGB for display
        self._model.image = cv2.cvtColor(normalized_img, cv2.COLOR_GRAY2RGB)

    def apply_filter_sobelY(self, border_type_ui="Spiegeln"):
        print(f"[DEBUG] Applying Sobel Y Filter with edge handling: {border_type_ui}")
        # Convert the image to grayscale
        grayscale_image = cv2.cvtColor(self._model.image, cv2.COLOR_RGB2GRAY)

        # Apply the Sobel Y filter
        filtered_img = IF.applySobelFilter(grayscale_image, direction="y", border_type_ui=border_type_ui)

        #Extra: Normalize the filtered image to the range 0–255 for display
        normalized_img = cv2.normalize(filtered_img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        #Extra: Convert the normalized image back to 3-channel RGB for display
        self._model.image = cv2.cvtColor(normalized_img, cv2.COLOR_GRAY2RGB)

##########################################
#zussatzaufgabe:
#########################################
    
    # def apply_moving_avg_filter_separated(self, kernel_size):
    #     try:
    #         print(f"[DEBUG] Applying Moving Average Filter with Separated Kernels: kernel_size={kernel_size}")
    #         # Convert the image to grayscale
    #         grayscale_image = cv2.cvtColor(self._model.image, cv2.COLOR_RGB2GRAY)

    #         # Apply the moving average filter using separated kernels
    #         img = IF.applyMovingAverageFilterWithSeperatedKernels(grayscale_image, kernel_size)

    #         # Update the model with the filtered grayscale image
    #         self._model.image = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    #     except Exception as e:
    #         print(f"[ERROR] Error applying Moving Average Filter with Separated Kernels: {e}")
    #         logging.error(f"Error applying Moving Average Filter with Separated Kernels: {e}", exc_info=True)
    #         raise

    def apply_moving_avg_filter_convolution(self, kernel_size, border_type_ui):
        """
        Applies the moving average filter using manual convolution.
        Args:
            kernel_size (int): The size of the kernel.
            border_type_ui (str): The border handling method selected in the UI.
        """
        print(f"[DEBUG] Applying Moving Average Filter with Convolution: kernel_size={kernel_size}, border_type={border_type_ui}")
        try:
            # Convert the input image to grayscale
            grayscale_image = cv2.cvtColor(self._model.image, cv2.COLOR_RGB2GRAY)

            # Apply the moving average filter
            filtered_img = IF.applyKernelInSpatialDomain(grayscale_image, kernel_size, border_type_ui)

            # Ensure the filtered image is in the correct format (grayscale)
            filtered_img = np.clip(filtered_img, 0, 255).astype(np.uint8)

            # Convert the filtered image back to 3-channel RGB for display
            self._model.image = cv2.cvtColor(filtered_img, cv2.COLOR_GRAY2RGB)
        except ValueError as e:
            print(f"[ERROR] ValueError in Moving Average Filter with Convolution: {e}")
            logging.error(f"ValueError in Moving Average Filter with Convolution: {e}", exc_info=True)
        except Exception as e:
            print(f"[ERROR] Unexpected error in Moving Average Filter with Convolution: {e}")
            logging.error(f"Unexpected error in Moving Average Filter with Convolution: {e}", exc_info=True)

    def run_runtime_evaluation(self):
        border_type_ui = "Spiegeln"  # Default border type
        IF.run_runtime_evaluation(self._model.input_image, border_type_ui)


