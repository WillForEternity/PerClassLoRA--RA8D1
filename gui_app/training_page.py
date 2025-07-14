import sys
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QTextEdit, QFrame, QStackedWidget
from PyQt6.QtGui import QFont
from PyQt6.QtCore import Qt

# This will be expanded later with a real worker thread

class TrainingPage(QWidget):
    def __init__(self):
        super().__init__()
        self.is_setup_complete = False
        self.setup_ui()

    def setup_ui(self):
        main_layout = QVBoxLayout(self)
        self.page_stack = QStackedWidget()
        main_layout.addWidget(self.page_stack)

        # --- Prerequisites View ---
        prereq_view = QWidget()
        prereq_layout = QVBoxLayout(prereq_view)
        prereq_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        prereq_label = QLabel("Please complete the environment setup on the 'Setup' page first.")
        prereq_label.setFont(QFont("Arial", 16))
        prereq_layout.addWidget(prereq_label)
        self.page_stack.addWidget(prereq_view)

        # --- Main Content View ---
        main_view = QWidget()
        layout = QVBoxLayout(main_view)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)

        title = QLabel("Model Training")
        title.setFont(QFont("Arial", 24, QFont.Weight.Bold))
        layout.addWidget(title)

        main_content_layout = QHBoxLayout()
        layout.addLayout(main_content_layout)

        left_panel = QVBoxLayout()
        main_content_layout.addLayout(left_panel, 1)

        self.run_button = QPushButton("Start Training")
        self.run_button.setFont(QFont("Arial", 12))
        self.run_button.setMinimumHeight(40)
        left_panel.addWidget(self.run_button)

        self.log_console = QTextEdit()
        self.log_console.setReadOnly(True)
        self.log_console.setFont(QFont("Courier New", 10))
        self.log_console.setStyleSheet("background-color: #000; color: #0f0; border: 1px solid #444;")
        self.log_console.setText("Training logs will appear here...")
        left_panel.addWidget(self.log_console)

        graph_placeholder = QLabel("Training Graph (Coming Soon)")
        graph_placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        graph_placeholder.setFont(QFont("Arial", 16))
        graph_placeholder.setFrameShape(QFrame.Shape.StyledPanel)
        graph_placeholder.setStyleSheet("background-color: #2d2d2d; border: 1px solid #444;")
        main_content_layout.addWidget(graph_placeholder, 1)
        
        self.page_stack.addWidget(main_view)

    def set_setup_status(self, is_complete):
        self.is_setup_complete = is_complete
        if self.is_setup_complete:
            self.page_stack.setCurrentIndex(1)
        else:
            self.page_stack.setCurrentIndex(0)
