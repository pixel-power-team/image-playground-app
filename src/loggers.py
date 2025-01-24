import logging
from PyQt6.QtCore import pyqtSignal, pyqtSlot, QObject
import html

class LogEmitter(QObject, logging.Handler):
    """A custom logging handler that emits a signal with the log message."""
    messageEmitted = pyqtSignal(str)

    def __init__(self):
        super().__init__()

        # Define color codes
        self.color_codes = {
            logging.DEBUG: 'gray',
            logging.INFO: 'black',
            logging.WARNING: 'orange',
            logging.ERROR: 'red',
            logging.CRITICAL: 'darkred'
        }

        self.level_names = {
            logging.DEBUG: 'DEBUG',
            logging.INFO: 'INFO',
            logging.WARNING: 'WARNING',
            logging.ERROR: 'ERROR',
            logging.CRITICAL: 'CRITICAL'
        }

    def emit(self, record):
        msg = self.format(record)

        # Get color code and level name based on log level
        level_color = self.color_codes.get(record.levelno, 'black')
        level_name = self.level_names.get(record.levelno, 'UNKNOWN')

        # Create HTML string with color code and level name
        html_msg = f'<span style="color: {level_color}">{level_name}: {html.escape(msg)}</span>'

        # Emit signal with HTML message
        self.messageEmitted.emit(html_msg)