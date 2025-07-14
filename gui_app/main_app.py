import sys
from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QStackedWidget, QLabel, QFrame
from PyQt6.QtGui import QFont
from PyQt6.QtCore import Qt, QSize

from gui_app.setup_page import SetupPage

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Hand Gesture Recognition System")
        self.setGeometry(100, 100, 1200, 800)
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1e1e1e;
            }
            QPushButton {
                background-color: #3e3e3e;
                color: #f0f0f0;
                border: 1px solid #555;
                padding: 10px;
                text-align: left;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #4e4e4e;
            }
            QPushButton:checked {
                background-color: #007acc;
                border-left: 3px solid #6ee2ff;
            }
            QLabel {
                color: #f0f0f0;
                font-size: 16px;
            }
            QFrame#sidebar {
                background-color: #252526;
            }
        """)

        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # --- Sidebar --- #
        sidebar = QFrame()
        sidebar.setObjectName("sidebar")
        sidebar.setFixedWidth(200)
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        sidebar_layout.setContentsMargins(0, 20, 0, 20)
        sidebar_layout.setSpacing(5)

        # --- Content Area --- #
        self.content_area = QStackedWidget()
        main_layout.addWidget(sidebar)
        main_layout.addWidget(self.content_area)

        # --- Navigation Buttons and Pages --- #
        self.nav_buttons = {}
        pages = {
            "Setup": SetupPage(),
            "Data Collection": QLabel("Data Collection Page - Coming Soon!"),
            "Training": QLabel("Training Page - Coming Soon!"),
            "Inference": QLabel("Inference Page - Coming Soon!")
        }

        for name, page in pages.items():
            # Center the placeholder text
            if isinstance(page, QLabel):
                page.setAlignment(Qt.AlignmentFlag.AlignCenter)
                page.setFont(QFont("Arial", 24))

            button = QPushButton(name)
            button.setCheckable(True)
            button.clicked.connect(lambda checked, idx=self.content_area.count(): self.content_area.setCurrentIndex(idx))
            sidebar_layout.addWidget(button)
            self.content_area.addWidget(page)
            self.nav_buttons[name] = button

        # Select the first button by default
        self.nav_buttons["Setup"].setChecked(True)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
