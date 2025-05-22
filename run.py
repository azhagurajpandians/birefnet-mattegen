import sys
from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QPalette, QColor
from PySide6.QtCore import Qt
from ui import MatteGeneratorApp

if __name__ == "__main__":
    app = QApplication(sys.argv)

    app.setStyle("Fusion")

    # Enable dark mode
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(30, 30, 40))  # Dark blue background
    palette.setColor(QPalette.WindowText, QColor(220, 220, 220))  # Light gray text
    palette.setColor(QPalette.Base, QColor(40, 40, 50)) 
    palette.setColor(QPalette.AlternateBase, QColor(30, 30, 40)) 
    palette.setColor(QPalette.ToolTipBase, QColor(25, 25, 30))
    palette.setColor(QPalette.ToolTipText, QColor(220, 220, 220))
    palette.setColor(QPalette.Text, QColor(220, 220, 220))
    palette.setColor(QPalette.Button, QColor(40, 40, 50))
    palette.setColor(QPalette.ButtonText, QColor(220, 220, 220))
    palette.setColor(QPalette.Highlight, QColor(0, 120, 215))  # Dark blue highlight
    palette.setColor(QPalette.HighlightedText, Qt.white)
    app.setPalette(palette)

    window = MatteGeneratorApp()
    window.show()
    sys.exit(app.exec())
