import numpy as np
import Utilities
import cv2

from PyQt6.QtCore import QObject, pyqtSignal

class ImageModel(QObject):
    image_changed = pyqtSignal(np.ndarray)
    input_image_changed = pyqtSignal(np.ndarray)

    _image: np.ndarray = None          #output image
    _input_image: np.ndarray = None    #input image (backup)

    def __init__(self):
        super().__init__()
        self.initialize()

    @property
    def image(self):
        return self._image

    @property
    def input_image(self):
        return self._input_image

    @image.setter
    def image(self, img):
        self._image = img.copy()
        print("M: image changed!")
        # update in model is reflected in view by sending a signal to view
        self.image_changed.emit(img)

    @input_image.setter
    def input_image(self, img):
        self._input_image = img.copy()
        self.image = img
        self.input_image_changed.emit(img)

    def initialize(self):
        image: np.ndarray = np.zeros((256, 256, 3), np.uint8)
        image[:, 0:256//2] = (255, 0, 0)
        image[:, 256//2:256] = (0, 0, 255)
        self.input_image = image

    class ColorAnalysis:

        _clusters = None
        _image = None
        _colors = None
        _labels = None

        def __init__(self, img, clusters=3):
            self.CLUSTERS = clusters
            self.IMAGE = Utilities.resize_image(img, 100)
            cv2.imshow("Input", self.IMAGE)




    def load_rgb_image(self, path):
        image = cv2.imread(path, 1)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.image = image
