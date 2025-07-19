import sys
import os

# Add project root to the Python path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QStackedWidget, QLabel, QFrame, QButtonGroup
)
from PyQt6.QtGui import QFont
from PyQt6.QtCore import Qt

from gui_app.setup_page import SetupPage
from gui_app.data_collection_page import DataCollectionPage
from gui_app.training_page import TrainingPage
from gui_app.quantize_page import QuantizePage
from gui_app.inference_page import InferencePage

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.is_setup_complete = False

        self.setWindowTitle("Hand Gesture Recognition System")
        self.setGeometry(100, 100, 1200, 800)
        self.setStyleSheet("""
            QMainWindow { background-color: #1e1e1e; }
            QFrame#sidebar { background-color: #252526; }
            QLabel { color: #f0f0f0; font-size: 16px; }
            QPushButton {
                color: #ccc;
                background-color: transparent;
                border: none;
                text-align: left;
                padding-left: 20px;
                font: 14px 'Arial';
                min-height: 50px;
            }
            QPushButton:hover {
                background-color: #3a3a3a;
            }
            QPushButton:checked {
                background-color: #0078d4;
                color: white;
                border-left: 4px solid #91c9f7;
            }
        """)

        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        nav_panel = QFrame()
        nav_panel.setObjectName("sidebar")
        nav_panel.setFixedWidth(200)
        nav_layout = QVBoxLayout(nav_panel)
        nav_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        nav_layout.setContentsMargins(0, 20, 0, 20)
        nav_layout.setSpacing(5)

        self.page_container = QStackedWidget()

        self.pages = {
            "Setup": SetupPage(),
            "Data Collection": DataCollectionPage(),
            "Training": TrainingPage(),
            "Quantize": QuantizePage(),
            "Inference": InferencePage()
        }
        for name, page in self.pages.items():
            if isinstance(page, QLabel):
                page.setAlignment(Qt.AlignmentFlag.AlignCenter)
                page.setFont(QFont("Arial", 24))
        
        self.pages["Setup"].setup_completed.connect(self.on_setup_completed)
        self.pages["Data Collection"].set_navigation_enabled.connect(self.set_navigation_enabled)
        self.pages["Training"].set_navigation_enabled.connect(self.set_navigation_enabled)
        self.pages["Inference"].set_navigation_enabled.connect(self.set_navigation_enabled)
        self.pages["Quantize"].set_navigation_enabled.connect(self.set_navigation_enabled)

        self.nav_button_group = QButtonGroup(self)
        self.nav_button_group.setExclusive(True)
        self.nav_button_group.buttonClicked.connect(self.switch_page)

        for name, page in self.pages.items():
            self.page_container.addWidget(page)
            button = QPushButton(name)
            button.setCheckable(True)
            nav_layout.addWidget(button)
            self.nav_button_group.addButton(button)

        main_layout.addWidget(nav_panel)
        main_layout.addWidget(self.page_container)

        initial_button = self.nav_button_group.buttons()[0]
        initial_button.setChecked(True)
        self.page_container.setCurrentWidget(self.pages[initial_button.text()])

    def switch_page(self, button):
        page_name = button.text()
        if page_name in self.pages:
            self.page_container.setCurrentWidget(self.pages[page_name])

    def set_navigation_enabled(self, enabled):
        for button in self.nav_button_group.buttons():
            button.setEnabled(enabled)

    def on_setup_completed(self, success):
        self.is_setup_complete = success
        self.pages["Data Collection"].set_setup_status(success)
        self.pages["Training"].set_setup_status(success)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
