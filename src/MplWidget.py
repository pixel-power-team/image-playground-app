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
        active_channels = [i for i, hist in enumerate(histogram) if
                           any(hist)]  # Prüft, welche Kanäle nicht leer sind

        colors = ('r', 'g', 'b')
        if len(active_channels) == 1:  # Falls nur ein Kanal vorhanden ist
            colors = (colors[active_channels[
                0]],)  # Nur die Farbe des aktiven Kanals behalten

        for i, col in zip(active_channels, colors):
            hist_norm = histogram[i] / max(
                histogram[i])  # Normiere auf [0,1] Bereich
            self.canvas.ax.plot(hist_norm, color=col)
            self.canvas.ax.set_xlim([0, 256])
            self.canvas.ax.set_ylim([0, 1])  # Skalierung auf 0-1 setzen

        self.canvas.fig.tight_layout()
        self.canvas.draw()

    def save_histogram(self, str):
        if self.canvas.fig is not None:
            # Speichere die Figur als Bild
            self.canvas.fig.savefig(str, dpi=300)

