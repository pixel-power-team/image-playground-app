import numpy as np
import cv2
from PyQt6 import QtGui
from PyQt6.QtCore import Qt, QThread, QTimer, QRegularExpression
from PyQt6.QtWidgets import QMainWindow, QWidget, QPushButton, QVBoxLayout, QApplication, QSlider
from PyQt6.QtGui import QPixmap, QColor, QRegularExpressionValidator
import logging
from loggers import LogEmitter
import models


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



        self._ui.widget_histogram.controller = self._main_controller


        ####################
        # Input validators
        ####################
        # Create a regular expression that matches integer values between 1 and infinity
        regex = QRegularExpression("[1-9][0-9]*")
        # Create a validator based on the regular expression
        validator = QRegularExpressionValidator(regex)
        # Set the validator for the line edit widget
        self._ui.lineEdit_image_height.setValidator(validator)
        self._ui.lineEdit_image_width.setValidator(validator)


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
        #
        # # Dropdown-Menü für Randbehandlung hinzufügen
        # self._ui.comboBox_border_handling = QComboBox(self)
        # self._ui.comboBox_border_handling.addItem("Extrapolieren", cv2.BORDER_REPLICATE)
        # self._ui.comboBox_border_handling.addItem("Spiegeln", cv2.BORDER_REFLECT)
        # self._ui.comboBox_border_handling.addItem("Zyklisch", cv2.BORDER_WRAP)
        # self._ui.comboBox_border_handling.addItem("Nullen", cv2.BORDER_CONSTANT)
        # # UI in das Layout integrieren
        # self._ui.layout_filter_options.addWidget(self._ui.comboBox_border_handling)
        # # Standardauswahl
        # self._ui.comboBox_border_handling.setCurrentIndex(1)

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
        #qt_img = convert_cv_qt(frame, size.width(), size.height())
        qt_img = convert_cv2scaledqt(frame, size.width(), size.height())
        #self._ui.label_output_image.loadImage(qt_img)
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
        """Liefert die ausgewählte Randbehandlung zurück"""
        return self._ui.comboBox_border_handling.currentData()

    def on_filter_sobelX_button_clicked(self):
        border_type = self.get_selected_border_handling()
        self._main_controller.apply_filter_sobelX(border_type)
        self.on_image_changed()

    def on_filter_sobelY_button_clicked(self):
        border_type = self.get_selected_border_handling()
        self._main_controller.apply_filter_sobelY(border_type)
        self.on_image_changed()

    def on_filter_gauss_button_clicked(self):
        border_type = self.get_selected_border_handling()
        self._main_controller.apply_gaussian_filter(self._ui.spinBox_filter_avg_size.value(), border_type)
        self.on_image_changed()

    def on_filter_moving_avg_button_clicked(self):
        border_type = self.get_selected_border_handling()
        self._main_controller.apply_moving_avg_filter(self._ui.spinBox_filter_avg_size.value(), border_type)
        self.on_image_changed()

    def on_filter_moving_avg_integral_button_clicked(self):
        self._main_controller.apply_moving_avg_filter_integral(self._ui.spinBox_filter_avg_size.value())
        self.on_image_changed()

    def on_filter_median_button_clicked(self):
        border_type = self.get_selected_border_handling()
        self._main_controller.apply_median_filter(self._ui.spinBox_filter_avg_size.value(), border_type)
        self.on_image_changed()

    def on_runtime_evaluation_button_clicked(self):
        self._main_controller.run_runtime_evaluation()

def convert_cv_qt(cv_img, display_width, display_height):
    """Convert from an opencv image to QPixmap"""
    h, w, ch = cv_img.shape
    bytes_per_line = ch * w
    convert_to_Qt_format = QtGui.QImage(cv_img.data, w, h, bytes_per_line, QtGui.QImage.Format.Format_RGB888)
    p = convert_to_Qt_format.scaled(display_width, display_height, Qt.AspectRatioMode.KeepAspectRatio)
    return QPixmap.fromImage(convert_to_Qt_format)

def convert_cv2scaledqt(cv_img, display_width, display_height):
    """Convert from an opencv image to QPixmap"""
    h, w, ch = cv_img.shape
    bytes_per_line = ch * w
    convert_to_Qt_format = QtGui.QImage(cv_img.data, w, h, bytes_per_line, QtGui.QImage.Format.Format_RGB888)
    p = convert_to_Qt_format.scaled(display_width, display_height, Qt.AspectRatioMode.KeepAspectRatio)
    return QPixmap.fromImage(p)