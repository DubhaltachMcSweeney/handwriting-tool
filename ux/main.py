import sys
from PyQt6.QtWidgets import QApplication

from .gui import MainWindow
from .integration import save_to_pipeline


def main():
    app = QApplication(sys.argv)

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()