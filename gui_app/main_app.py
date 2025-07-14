import sys
from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QStackedWidget, QLabel, QFrame
from PyQt6.QtGui import QFont
from PyQt6.QtCore import Qt

from gui_app.setup_page import SetupPage
from gui_app.data_collection_page import DataCollectionPage

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
        """)

        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # --- Sidebar --- #
        nav_panel = QFrame()
        nav_panel.setObjectName("sidebar")
        nav_panel.setFixedWidth(200)
        nav_layout = QVBoxLayout(nav_panel)
        nav_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        nav_layout.setContentsMargins(0, 20, 0, 20)
        nav_layout.setSpacing(5)

        # --- Page Container --- #
        self.page_container = QStackedWidget()

        # --- Pages and Navigation --- #
        self.pages = {
            "Setup": SetupPage(),
            "Data Collection": DataCollectionPage(),
            "Training": QLabel("Training Page - Coming Soon!"),
            "Inference": QLabel("Inference Page - Coming Soon!")
        }
        self.pages["Setup"].setup_completed.connect(self.on_setup_completed)

        self.nav_buttons = {}
        for name, page in self.pages.items():
            self.page_container.addWidget(page)
            if isinstance(page, QLabel):
                page.setAlignment(Qt.AlignmentFlag.AlignCenter)
                page.setFont(QFont("Arial", 24))

            button = QPushButton(name)
            button.setFont(QFont("Arial", 14))
            button.setMinimumHeight(50)
            button.setStyleSheet("""
                QPushButton {
                    color: #ccc;
                    background-color: transparent;
                    border: none;
                    text-align: left;
                    padding-left: 20px;
                }
                QPushButton:hover {
                    background-color: #3a3a3a;
                }
                QPushButton[checked="true"] {
                    background-color: #0078d4;
                    color: white;
                    border-left: 4px solid #91c9f7;
                }
            """)
            button.setCheckable(True)
            button.clicked.connect(lambda checked, n=name: self.switch_page(n))
            
            if name != "Setup":
                button.setEnabled(False)

            nav_layout.addWidget(button)
            self.nav_buttons[name] = button

        main_layout.addWidget(nav_panel)
        main_layout.addWidget(self.page_container)

        self.switch_page("Setup")

    def switch_page(self, page_name):
        self.page_container.setCurrentWidget(self.pages[page_name])
        for name, btn in self.nav_buttons.items():
            btn.setChecked(name == page_name)

    def on_setup_completed(self, success):
        self.is_setup_complete = success
        if success:
            self.nav_buttons["Data Collection"].setEnabled(True)
            # self.nav_buttons["Training"].setEnabled(True) # Will be enabled later
            # self.nav_buttons["Inference"].setEnabled(True) # Will be enabled later

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

