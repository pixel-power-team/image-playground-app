from PyQt6.QtWidgets import QLabel, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem
from PyQt6.QtGui import QTransform
from PyQt6.QtCore import Qt, QPointF

class ImagePlotter(QGraphicsView):
    def __init__(self, parent=None):
        super(ImagePlotter, self).__init__(parent)

        self.scene = QGraphicsScene()
        #self.setAlignment(Qt.AlignCenter)
        self.zoom_level = 0

    def show_image(self, image):
        self.scene.clear()
        item = QGraphicsPixmapItem(image)
        self.scene.addItem(item)
        self.setScene(self.scene)

        scene_rect = self.scene.itemsBoundingRect()


        self.fitInView(scene_rect, Qt.AspectRatioMode.KeepAspectRatio)
        print(self.scene.sceneRect().width())
        print(self.width())
        self.show()
    def wheelEvent(self, event):
        # Get the current zoom level of the view
        self.zoom_level = self.transform().m11()
        print(self.zoom_level)
        zoom = self.transform().m11()

        # Calculate the new zoom level based on the direction of the mouse wheel
        if event.angleDelta().y() > 0:
            zoom *= 1.2
        else:
            zoom /= 1.2

        # Set the new zoom level of the view
        self.setTransform(QTransform().scale(zoom, zoom))



class ImagePlotter2(QLabel):
    def __init__(self, parent=None):
        super(ImagePlotter2, self).__init__(parent)
        self.setMouseTracking(True)
        self.setCursor(Qt.CrossCursor)

    #def mouseMoveEvent(self, event):
    #    cursor_pos = event.pos()
    #    print("Cursor position:", cursor_pos.x(), cursor_pos.y())

    def mousePressEvent(self, event):
        cursor_pos = event.pos()
        pixmap = self.pixmap()
        label_pos = self.pos()
        pixmap_rect = pixmap.rect()

        pixmap_pos = label_pos + pixmap_rect.topLeft()

        pixel_color = pixmap.toImage().pixelColor(0, 0)
        red = pixel_color.red()
        green = pixel_color.green()
        blue = pixel_color.blue()

        print("RGB value of pixel at position (10, 10):", red, green, blue)

        print("Position of pixmap top left:", pixmap_rect.topLeft().x(), pixmap_rect.topLeft().y())
        print("Position of pixmap in QLabel:", pixmap_pos.x(), pixmap_pos.y())
        print("Label pos:", label_pos.x(), label_pos.y())
        #image_pos = self.mapFrom(self., cursor_pos)

#        print("Mouse clicked at:", event.pos().x(), event.pos().y())
#        print("Mouse clicked at:", image_pos.x(), image_pos.y())


from PyQt6.QtGui import QImage, QPixmap, QPainter
from PyQt6 import QtCore, QtGui, QtWidgets


__author__ = "Atinderpal Singh"
__license__ = "MIT"
__version__ = "1.0"
__email__ = "atinderpalap@gmail.com"

class ImagePlotter3(QLabel):
    ''' Basic image viewer class to show an image with zoom and pan functionaities.
        Requirement: Qt's Qlabel widget name where the image will be drawn/displayed.
    '''
    def __init__(self, parent=None):
        super(ImagePlotter3, self).__init__(parent)
        self.qlabel_image = QLabel()                            # widget/window name where image is displayed (I'm usiing qlabel)
        self.qimage_scaled = QImage()                         # scaled image to fit to the size of qlabel_image
        self.qpixmap = QPixmap()                              # qpixmap to fill the qlabel_image

        self.zoomX = 1              # zoom factor w.r.t size of qlabel_image
        self.position = [0, 0]      # position of top left corner of qimage_label w.r.t. qimage_scaled
        self.panFlag = False        # to enable or disable pan

        self.qlabel_image.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Ignored)
        self.__connectEvents()

    def __connectEvents(self):
        # Mouse events
        self.qlabel_image.mousePressEvent = self.mousePressAction
        self.qlabel_image.mouseMoveEvent = self.mouseMoveAction
        self.qlabel_image.mouseReleaseEvent = self.mouseReleaseAction

    def onResize(self):
        ''' things to do when qlabel_image is resized '''
        self.qpixmap = QPixmap(self.qlabel_image.size())
        self.qpixmap.fill(QtCore.Qt.gray)
        self.qimage_scaled = self.qimage.scaled(self.qlabel_image.width() * self.zoomX, self.qlabel_image.height() * self.zoomX, QtCore.Qt.KeepAspectRatio)
        self.update()

    def loadImage(self, input_image):
        ''' To load and display new image.'''
        self.qimage = input_image.toImage()
        print(self.qimage.width())
        self.qpixmap = QPixmap(self.qlabel_image.size())
        if not self.qimage.isNull():
            # reset Zoom factor and Pan position
            self.zoomX = 1
            self.position = [0, 0]
            self.qimage_scaled = self.qimage.scaled(self.qlabel_image.width(), self.qlabel_image.height(), QtCore.Qt.KeepAspectRatio)
            self.update()
        else:
            self.statusbar.showMessage('Cannot open this image! Try another one.', 5000)

    def update(self):
        ''' This function actually draws the scaled image to the qlabel_image.
            It will be repeatedly called when zooming or panning.
            So, I tried to include only the necessary operations required just for these tasks.
        '''
        if not self.qimage_scaled.isNull():
            # check if position is within limits to prevent unbounded panning.
            px, py = self.position
            px = px if (px <= self.qimage_scaled.width() - self.qlabel_image.width()) else (self.qimage_scaled.width() - self.qlabel_image.width())
            py = py if (py <= self.qimage_scaled.height() - self.qlabel_image.height()) else (self.qimage_scaled.height() - self.qlabel_image.height())
            px = px if (px >= 0) else 0
            py = py if (py >= 0) else 0
            self.position = (px, py)

            if self.zoomX == 1:
                self.qpixmap.fill(QtCore.Qt.white)

            # the act of painting the qpixamp
            painter = QPainter()
            painter.begin(self.qpixmap)
            painter.drawImage(QtCore.QPoint(0, 0), self.qimage_scaled, QtCore.QRect(self.position[0], self.position[1], self.qlabel_image.width(), self.qlabel_image.height()) )
            painter.end()

            self.qlabel_image.setPixmap(self.qpixmap)
        else:
            pass

    def mousePressAction(self, QMouseEvent):
        x, y = QMouseEvent.pos().x(), QMouseEvent.pos().y()
        #print(x,y)
        if self.panFlag:
            self.pressed = QMouseEvent.pos()    # starting point of drag vector
            self.anchor = self.position         # save the pan position when panning starts
        print("Label pos:")

    def mouseMoveAction(self, QMouseEvent):
        x, y = QMouseEvent.pos().x(), QMouseEvent.pos().y()
        if self.pressed:
            dx, dy = x - self.pressed.x(), y - self.pressed.y()         # calculate the drag vector
            self.position = self.anchor[0] - dx, self.anchor[1] - dy    # update pan position using drag vector
            self.update()

            # show the image with udated pan position

        print("Label pos:")

    def mouseReleaseAction(self, QMouseEvent):
        self.pressed = None                                             # clear the starting point of drag vector

    def zoomPlus(self):
        self.zoomX += 1
        px, py = self.position
        px += self.qlabel_image.width()/2
        py += self.qlabel_image.height()/2
        self.position = (px, py)
        self.qimage_scaled = self.qimage.scaled(self.qlabel_image.width() * self.zoomX, self.qlabel_image.height() * self.zoomX, QtCore.Qt.KeepAspectRatio)
        self.update()

    def zoomMinus(self):
        if self.zoomX > 1:
            self.zoomX -= 1
            px, py = self.position
            px -= self.qlabel_image.width()/2
            py -= self.qlabel_image.height()/2
            self.position = (px, py)
            self.qimage_scaled = self.qimage.scaled(self.qlabel_image.width() * self.zoomX, self.qlabel_image.height() * self.zoomX, QtCore.Qt.KeepAspectRatio)
            self.update()

    def resetZoom(self):
        self.zoomX = 1
        self.position = [0, 0]
        self.qimage_scaled = self.qimage.scaled(self.qlabel_image.width() * self.zoomX, self.qlabel_image.height() * self.zoomX, QtCore.Qt.KeepAspectRatio)
        self.update()

    def enablePan(self, value):
        self.panFlag = value