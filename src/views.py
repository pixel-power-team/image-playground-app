import numpy as np
import cv2
from PyQt6 import QtGui
from PyQt6.QtCore import Qt, QThread, QTimer, QRegularExpression
from PyQt6.QtWidgets import QMainWindow, QWidget, QPushButton, QVBoxLayout, QApplication, QSlider
from PyQt6.QtGui import QPixmap, QColor, QRegularExpressionValidator
import logging
from loggers import LogEmitter
import models
import ImageFiltering as IF  # Import the ImageFiltering module


import sys
import logging
from PyQt6.QtWidgets import QApplication, QMainWindow, QTextBrowser
from PyQt6.QtCore import pyqtSignal, pyqtSlot

# -*- coding: utf-8 -*-


from PyQt6.QtWidgets import QMainWindow, QFileDialog
from PyQt6.QtWidgets import QComboBox
from PyQt6.QtCore import pyqtSlot, QObject


from Playground_UI import Ui_MainWindow



class MainView(QMainWindow):
    def __init__(self, model, main_controller):
        super().__init__()
        self._model = model
        self._main_controller = main_controller
        self._ui = Ui_MainWindow()
        self._ui.setupUi(self)
        self.update_image()
        self.update_input_image()

        # Clear existing items in the dropdown to avoid duplicates
        # This ensures that the dropdown is reset before adding new items.
        self._ui.comboBox_border_handling.clear()

        # Initialize edge handling dropdown with localized terms
        # These terms ("Extrapolieren", "Spiegeln", etc.) are user-friendly and correspond to the edge handling methods.
        valid_methods = [
            ("Extrapolieren", "Extrapolieren"),
            ("Spiegeln", "Spiegeln"),
            ("Zyklisch", "Zyklisch"),
            ("Nullen", "Nullen")
        ]
        # Add each method to the dropdown menu
        for display_text, method in valid_methods:
            self._ui.comboBox_border_handling.addItem(display_text, method)
        self._ui.comboBox_border_handling.setCurrentIndex(0)  # Default to the first method

        self._ui.widget_histogram.controller = self._main_controller


        ####################################################################
        #   connect widgets to controllers
        ####################################################################
        # open file buttons
        #self._ui.pushButton.clicked.connect(self._main_controller.test_function)
        self._ui.actionBild_laden.triggered.connect(self.on_open_image_from_filesystem)
        self._ui.actionBild_speichern.triggered.connect(self.on_save_image_to_filesystem)
        self._ui.action_save_histogram.triggered.connect(self.on_save_histogram_to_filesystem)
        self._ui.horizontalSlider_color_clusters.sliderReleased.connect(self.on_color_cluster_slider_changed)
        self._ui.pushButton_hist_stretch.clicked.connect(self.on_hist_stretch_button_clicked)
        self._ui.pushButton_hist_convertToGrayScale.clicked.connect(self.on_hist_convertToGrayScale_button_clicked)
        self._ui.pushButton_hist_equalization.clicked.connect(self.on_hist_equal_button_clicked)
        self._ui.pushButton_AWS_Labeling.clicked.connect(self.on_AWS_Rekognition_button_clicked)
        self._ui.pushButton_hist_fill.clicked.connect(self.on_hist_fill_button_clicked)
        self._ui.pushButton_adjust_image_size.clicked.connect(self.on_resize_button_clicked)

        self._ui.pushButton_reset_output_image.clicked.connect(self.on_reset_output_image_button_clicked)
        self._ui.pushButton_overwrite_input_image.clicked.connect(self.on_overwrite_input_image_button_clicked)
        self._ui.lineEdit_image_height.editingFinished.connect(self.on_new_image_height_requested)
        self._ui.lineEdit_image_width.editingFinished.connect(self.on_new_image_width_requested)

        #########
        # Buttons für Übung
        #########
        self._ui.pushButton_show_channel1.clicked.connect(self.on_channel_1_button_clicked)
        self._ui.pushButton_show_channel2.clicked.connect(self.on_channel_2_button_clicked)
        self._ui.pushButton_show_channel3.clicked.connect(self.on_channel_3_button_clicked)
        self._ui.pushButton_do_image_manipulation.clicked.connect(self.on_do_first_image_manipulation_button_clicked)

        self._ui.pushButton_hist_log.clicked.connect(self.on_apply_log_on_hist_button_clicked)
        self._ui.pushButton_hist_exp.clicked.connect(self.on_apply_exp_on_hist_button_clicked)
        self._ui.pushButton_hist_inv.clicked.connect(self.on_apply_inverse_on_hist_button_clicked)
        self._ui.horizontalSlider_hist_threshold.sliderReleased.connect(self.on_apply_threshold_on_hist_button_clicked)

        self._ui.pushButton_filter_sobelX.clicked.connect(self.on_filter_sobelX_button_clicked)
        self._ui.pushButton_filter_sobelY.clicked.connect(self.on_filter_sobelY_button_clicked)
        self._ui.pushButton_filter_gauss.clicked.connect(self.on_filter_gauss_button_clicked)
        self._ui.pushButton_filter_movAvg.clicked.connect(self.on_filter_moving_avg_button_clicked)
        self._ui.pushButton_filter_movAvg_int.clicked.connect(self.on_filter_moving_avg_integral_button_clicked)
        self._ui.pushButton_filter_median.clicked.connect(self.on_filter_median_button_clicked)
        self._ui.pushButton_filter_evaluation.clicked.connect(self.on_runtime_evaluation_button_clicked)
        self._ui.pushButton_filter_movAvg_sep.clicked.connect(self.on_filter_moving_avg_sep_button_clicked)
        self._ui.pushButton_filter_movAvg_conv.clicked.connect(self.on_filter_moving_avg_conv_button_clicked)

        ####################################################################
        #   listen for model event signals
        ####################################################################
        # file name is updated
        self._model.image_changed.connect(self.on_image_changed)
        self._model.input_image_changed.connect(self.on_input_image_changed)

        ###################
        #   Connect Logging
        ###################

        self._main_controller.log_handler.messageEmitted.connect(self.add_log_message)


    def show(self):
        super().show()
        self.on_input_image_changed()

    @pyqtSlot(str)
    def add_log_message(self, msg):
        """Add a log message to the QTextBrowser widget."""
        self._ui.text_output.append(msg)

    def resizeEvent(self, event):
        self.on_input_image_changed()
        QMainWindow.resizeEvent(self, event)

    def on_open_image_from_filesystem(self):
        fname = QFileDialog.getOpenFileName(self, 'Open file', "../","Image Files (*.png *.jpg *.bmp)")
        self._main_controller.loadImage(fname[0])
        print(fname[0])

    def on_save_image_to_filesystem(self):
        fname, _ = QFileDialog.getSaveFileName(self, 'Save file', "../", "Image Files (*.png *.jpg *.bmp)")
        if fname:
            self._main_controller.saveImage(fname)

    def on_save_histogram_to_filesystem(self):
        fname, _ = QFileDialog.getSaveFileName(self, 'Save file', "../", "Image Files (*.png *.jpg *.bmp)")
        if fname:
            self._ui.widget_histogram.save_histogram(fname)

    def on_image_changed(self):
        self.update_image()
        self.update_histogram()
        self.update_image_information()


    def on_input_image_changed(self):
        self.update_input_image()
        self.on_image_changed()

    def on_image_mouse_pressed(self):
        self._main_controller.logger.logger.critical('Mouse pressed')


    def on_overwrite_input_image_button_clicked(self):
        self._main_controller.set_image_as_input_image()

    def on_reset_output_image_button_clicked(self):
        self._main_controller.reset_output_image()

    def on_new_image_height_requested(self):
        if self._ui.checkBox_fix_image_size.isChecked():
            image_size = self._main_controller.get_image_information()
            aspect_ratio = image_size[0] / image_size[1]
            #dim = (height, int(height * aspect_ratio))
            self._ui.lineEdit_image_width.setText(str(int(int(self._ui.lineEdit_image_height.text()) / aspect_ratio)))
        pass

    def on_new_image_width_requested(self):
        if self._ui.checkBox_fix_image_size.isChecked():
            image_size = self._main_controller.get_image_information()
            aspect_ratio = image_size[0] / image_size[1]
            #dim = (height, int(height * aspect_ratio))
            self._ui.lineEdit_image_height.setText(str(int(int(self._ui.lineEdit_image_width.text()) * aspect_ratio)))
        pass

    def on_color_cluster_slider_changed(self):
        dominant_colors = self._main_controller.analyseColors(self._ui.horizontalSlider_color_clusters.sliderPosition())
        self.update_color_cluster_output(dominant_colors)
        print(self._ui.horizontalSlider_color_clusters.sliderPosition())

    def on_hist_stretch_button_clicked(self):
        self._main_controller.stretch_image()
        self.on_image_changed()

    def on_hist_convertToGrayScale_button_clicked(self):
        self._main_controller.convert_to_grayscale()
        self.on_image_changed()

    def on_hist_equal_button_clicked(self):
        self._main_controller.equalize_image()
        self.on_image_changed()
    
    def on_hist_fill_button_clicked(self):
        self._main_controller.fill_hist()
        self.on_image_changed()

    def on_AWS_Rekognition_button_clicked(self):
        self._main_controller.label_image()


    def update_color_cluster_output(self, dominant_colors):
        size = self._ui.label_image_color_analysis_output.size()
        visual_output = self._main_controller.create_visual_output(dominant_colors, size.width(), size.height())
        qt_img = convert_cv_qt(visual_output, size.width(), size.height())
        self._ui.label_image_color_analysis_output.setPixmap(qt_img)

    def on_resize_button_clicked(self):
        self._main_controller.resize_image(int(self._ui.lineEdit_image_width.text()), int(self._ui.lineEdit_image_height.text()))
        #self.on_image_changed()

    def update_image(self):
        frame = self._model.image
        size = self._ui.label_output_image.size()

        # Ensure the image has 3 channels before converting to QPixmap
        if len(frame.shape) == 2:  # Grayscale image
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)

        qt_img = convert_cv2scaledqt(frame, size.width(), size.height())
        self._ui.label_output_image.setPixmap(qt_img)

    def update_input_image(self):
        frame = self._model.input_image
        size = self._ui.label_output_image.size()
        qt_img = convert_cv2scaledqt(frame, size.width(), size.height())
        #qt_img = convert_cv_qt(frame, size.width(), size.height())
        self._ui.label_input_image.setPixmap(qt_img)

    def update_histogram(self):
        self._ui.widget_histogram.drawHistogram(self._model.image)

    def update_image_information(self):
        image_size = self._main_controller.get_image_information()
        self._ui.label_height_image.setText(str(image_size[0]))
        self._ui.label_width_image.setText(str(image_size[1]))

        self._ui.lineEdit_image_height.setText(str(image_size[0]))
        self._ui.lineEdit_image_width.setText(str(image_size[1]))

        pixel_colors = self._main_controller.get_pixel_information()
        self._ui.label_color_pixel1.setText(str(pixel_colors[0]))
        self._ui.label_color_pixel2.setText(str(pixel_colors[1]))


    #####################
    # Übung 1
    #####################

    def on_channel_1_button_clicked(self):
        self._main_controller.show_channel(0)
        self.on_image_changed()

    def on_channel_2_button_clicked(self):
        self._main_controller.show_channel(1)
        self.on_image_changed()

    def on_channel_3_button_clicked(self):
        self._main_controller.show_channel(2)
        self.on_image_changed()

    def on_do_first_image_manipulation_button_clicked(self):
        self._main_controller.do_first_image_manipulation()
        self.on_image_changed()

    #####################
    # Übung 2
    #####################

    def on_apply_log_on_hist_button_clicked(self):
        self._main_controller.apply_log()
        self.on_image_changed()

    def on_apply_exp_on_hist_button_clicked(self):
        self._main_controller.apply_exp()
        self.on_image_changed()

    def on_apply_inverse_on_hist_button_clicked(self):
        self._main_controller.apply_inv()
        self.on_image_changed()

    def on_apply_threshold_on_hist_button_clicked(self):
        self._main_controller.apply_threshold(self._ui.horizontalSlider_hist_threshold.sliderPosition())
        self.on_image_changed()

    #####################
    # Übung 3
    #####################
    def get_selected_border_handling(self):
        """Returns the selected edge handling method from the dropdown."""
        selected_method = self._ui.comboBox_border_handling.currentText()  # Get the selected text
        print(f"[DEBUG] Selected border handling method: {selected_method}")
        return selected_method

    def on_filter_sobelX_button_clicked(self):
        try:
            border_type = self.get_selected_border_handling()
            print(f"Applying Sobel X Filter with border type: {border_type}")
            self._main_controller.apply_filter_sobelX(border_type)
            self.on_image_changed()
        except Exception as e:
            print(f"Error in Sobel X Filter: {e}")
            logging.error(f"Error in Sobel X Filter: {e}", exc_info=True)

    def on_filter_sobelY_button_clicked(self):
        try:
            border_type = self.get_selected_border_handling()
            print(f"Applying Sobel Y Filter with border type: {border_type}")
            self._main_controller.apply_filter_sobelY(border_type)
            self.on_image_changed()
        except Exception as e:
            print(f"Error in Sobel Y Filter: {e}")
            logging.error(f"Error in Sobel Y Filter: {e}", exc_info=True)

    def on_filter_gauss_button_clicked(self):
        try:
            # Retrieve kernel size and edge handling method
            kernel_size = self._ui.spinBox_filter_avg_size.value()
            border_type = self.get_selected_border_handling()

            # Apply the Gaussian filter
            print(f"Applying Gaussian Filter with kernel size: {kernel_size}, border type: {border_type}")
            self._main_controller.apply_gaussian_filter(kernel_size, border_type)
            self.on_image_changed()
        except Exception as e:
            print(f"Error in Gaussian Filter: {e}")
            logging.error(f"Error in Gaussian Filter: {e}", exc_info=True)

    def on_filter_moving_avg_button_clicked(self):
        try:
            # Retrieve kernel size and edge handling method
            kernel_size = self._ui.spinBox_filter_avg_size.value()
            border_type = self.get_selected_border_handling()
            if not border_type:
                border_type = "Spiegeln"  # Default to "Spiegeln" if no edge handling is selected
            # Apply the moving average filter
            print(f"Applying Moving Average Filter with kernel size: {kernel_size}, border type: {border_type}")
            self._main_controller.apply_moving_avg_filter(kernel_size, border_type)
            self.on_image_changed()
        except Exception as e:
            print(f"Error in Moving Average Filter: {e}")
            logging.error(f"Error in Moving Average Filter: {e}", exc_info=True)

    def on_filter_moving_avg_integral_button_clicked(self):
        self._main_controller.apply_moving_avg_filter_integral(self._ui.spinBox_filter_avg_size.value())
        self.on_image_changed()

    def on_filter_median_button_clicked(self):
        border_type = self.get_selected_border_handling()
        self._main_controller.apply_median_filter(self._ui.spinBox_filter_avg_size.value(), border_type)
        self.on_image_changed()

    def on_runtime_evaluation_button_clicked(self):
        self._main_controller.run_runtime_evaluation()
    
    def on_filter_moving_avg_sep_button_clicked(self):
        self._main_controller.apply_moving_avg_filter_separated(self._ui.spinBox_filter_avg_size.value())
        self.on_image_changed()

    def on_filter_moving_avg_conv_button_clicked(self):
        """
        Handles the button click for applying the moving average filter using manual convolution.
        Retrieves the kernel size and border type from the UI and passes them to the controller.
        """
        try:
            # Retrieve kernel size and border type from the UI
            kernel_size = self._ui.spinBox_filter_avg_size.value()
            border_type = self.get_selected_border_handling()

            print(f"[DEBUG] Applying Moving Average Filter with Convolution: kernel_size={kernel_size}, border_type={border_type}")
            self._main_controller.apply_moving_avg_filter_convolution(kernel_size, border_type)
            self.on_image_changed()
        except Exception as e:
            print(f"[ERROR] Error in Moving Average Filter with Convolution: {e}")
            logging.error(f"Error in Moving Average Filter with Convolution: {e}", exc_info=True)
        
def convert_cv_qt(cv_img, display_width, display_height):
    """Convert from an opencv image to QPixmap"""
    h, w, ch = cv_img.shape
    bytes_per_line = ch * w
    convert_to_Qt_format = QtGui.QImage(cv_img.data, w, h, bytes_per_line, QtGui.QImage.Format.Format_RGB888)
    p = convert_to_Qt_format.scaled(display_width, display_height, Qt.AspectRatioMode.KeepAspectRatio)
    return QPixmap.fromImage(p)

def convert_cv2scaledqt(cv_img, display_width, display_height):
    """Convert from an OpenCV image to QPixmap"""
    # Ensure the image has 3 channels before processing
    if len(cv_img.shape) == 2:  # Grayscale image
        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_GRAY2RGB)

    h, w, ch = cv_img.shape
    bytes_per_line = ch * w
    convert_to_Qt_format = QtGui.QImage(cv_img.data, w, h, bytes_per_line, QtGui.QImage.Format.Format_RGB888)
    p = convert_to_Qt_format.scaled(display_width, display_height, Qt.AspectRatioMode.KeepAspectRatio)
    return QPixmap.fromImage(p)