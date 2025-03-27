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
    def apply_boarder_handling(self, border_type):
        self._model.image = IF.applyBorderHandling(self._model.input_image, border_type)

    def apply_gaussian_filter(self, kernel_size, border_type="Reflect"):
        print(f"[DEBUG] Applying Gaussian Filter with edge handling: {border_type}")
        img = IF.applyGaussianFilter(self._model.input_image, kernel_size, border_type=border_type)
        self._model.image = Utilities.ensure_three_channel_grayscale_image(img)

    def apply_moving_avg_filter(self, kernel_size, border_type="Reflect"):
        try:
            if not border_type:
                border_type = "Reflect"  # Default to "Reflect" if no edge handling is provided
            print(f"[DEBUG] Applying Moving Average Filter with edge handling: {border_type}")
            grayscale_image = Utilities.ensure_one_channel_grayscale_image(self._model.image)
            kernel = IF.create_moving_average_kernel(kernel_size)
            filtered_img = IF.apply_filter_in_spatial_domain(grayscale_image, kernel, edge_handling=border_type)
            self._model.image = Utilities.ensure_three_channel_grayscale_image(filtered_img)
        except Exception as e:
            print(f"[ERROR] Error applying Moving Average Filter: {e}")
            logging.error(f"Error applying Moving Average Filter: {e}", exc_info=True)
            raise

    def apply_moving_avg_filter_integral(self, kernel_size):
        img = IF.applyMovingAverageFilterWithIntegralImage(self._model.input_image, kernel_size)
        self._model.image = Utilities.ensure_three_channel_grayscale_image(img)

    def apply_median_filter(self, kernel_size, border_type="Reflect"):
        print(f"[DEBUG] Applying Median Filter with edge handling: {border_type}")
        img = IF.applyMedianFilter(self._model.input_image, kernel_size, borderType=border_type)
        self._model.image = Utilities.ensure_three_channel_grayscale_image(img)

    def apply_filter_sobelX(self):
        img = IF.applySobelXFilter(self._model.input_image)
        self._model.image = Utilities.ensure_three_channel_grayscale_image(img)

    def apply_filter_sobelY(self):
        img = IF.applySobelYFilter(self._model.input_image)
        self._model.image = Utilities.ensure_three_channel_grayscale_image(img)

    def run_runtime_evaluation(self):
        IF.run_runtime_evaluation(self._model.input_image)



