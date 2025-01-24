# Imports
from PyQt6 import QtWidgets
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as Canvas
import matplotlib

# Ensure using PyQt5 backend
matplotlib.use('QT5Agg')

# Matplotlib canvas class to create figure
class MplCanvas(Canvas):
    def __init__(self):
        self.fig = Figure(figsize=(5, 2))
        self.ax = self.fig.add_subplot(111)
        Canvas.__init__(self, self.fig)
        #Canvas.setSizePolicy(self, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        Canvas.updateGeometry(self)

# Matplotlib widget
class MplWidget(QtWidgets.QWidget):
    _controller = None

    def __init__(self, parent=None):
        QtWidgets.QWidget.__init__(self, parent)   # Inherit from QWidget
        self.canvas = MplCanvas()                  # Create canvas object
        self.vbl = QtWidgets.QVBoxLayout()         # Set box for plotting
        self.vbl.addWidget(self.canvas)
        self.setLayout(self.vbl)

    @property
    def controller(self):
        return self._image

    @controller.setter
    def controller(self, controller):
        self._controller = controller

    def drawHistogram(self, img):
        self.canvas.ax.cla()

        histogram = self._controller.calculate_histogram(img)
        color = ('b', 'g', 'r')

        if len(histogram) == 1:
            color = ('b')

        for i, col in enumerate(color):
            self.canvas.ax.plot(histogram[i], color=col)
            self.canvas.ax.set_xlim([0, 256])
            #self.canvas.figure.set_size_inches(2,2)

        self.canvas.fig.tight_layout()
        self.canvas.draw()

    def save_histogram(self, str):
        if self.canvas.fig is not None:
            # Speichere die Figur als Bild
            self.canvas.fig.savefig(str, dpi=300)

