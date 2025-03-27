import sys
from models import ImageModel
from views import MainView
from controllers import MainController
from PyQt6.QtWidgets import QApplication



class App(QApplication):
    def __init__(self, sys_argv):
        super(App, self).__init__(sys_argv)
        self.model = ImageModel()
        self.main_controller = MainController(self.model)
        self.main_view = MainView(self.model, self.main_controller)
        self.main_view.show()


if __name__ == '__main__':
    app = App(sys.argv)
    sys.exit(app.exec())
    app.exit(app.exec_())