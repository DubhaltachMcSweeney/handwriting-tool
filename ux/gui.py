import sys
import tempfile

from PyQt6.QtWidgets import (
    QWidget, QPushButton,
    QVBoxLayout, QLabel,
    QFileDialog, QComboBox, QLineEdit
)

from .file_handler import convert_to_png
from .canvas import DrawingCanvas
from .integration import save_to_pipeline


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Handwriting Tool - Dataset Builder")
        self.setMinimumWidth(520)

        self.current_file = None

        # ---------------- UI ----------------
        self.label = QLabel("No file selected")
        self.status = QLabel("Ready")

        self.select_btn = QPushButton("Select Image")
        self.select_btn.clicked.connect(self.select_file)

        self.category_box = QComboBox()
        self.category_box.addItems([
            "digit",
            "letter",
            "sentence",
            "multi_digit"
        ])

        self.label_input = QLineEdit()
        self.label_input.setPlaceholderText("Optional label (A, B, word...)")

        self.canvas = DrawingCanvas()

        self.save_drawing_btn = QPushButton("Save Drawing")
        self.save_drawing_btn.clicked.connect(self.save_drawing)

        self.clear_canvas_btn = QPushButton("Clear Canvas")
        self.clear_canvas_btn.clicked.connect(self.canvas.clear)

        # ---------------- LAYOUT ----------------
        layout = QVBoxLayout()

        layout.addWidget(self.label)
        layout.addWidget(self.select_btn)

        layout.addWidget(QLabel("Category"))
        layout.addWidget(self.category_box)

        layout.addWidget(QLabel("Label (optional)"))
        layout.addWidget(self.label_input)

        layout.addWidget(QLabel("Draw"))
        layout.addWidget(self.canvas)

        layout.addWidget(self.save_drawing_btn)
        layout.addWidget(self.clear_canvas_btn)

        layout.addWidget(self.status)

        self.setLayout(layout)

    # ---------------- FILE INPUT ----------------
    def select_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Image",
            "",
            "Images (*.png *.jpg *.jpeg *.bmp)"
        )

        if not file_path:
            return

        try:
            png_path = convert_to_png(file_path)

            saved_path = save_to_pipeline(
                png_path,
                task_type=self.category_box.currentText(),
                label=self.label_input.text().strip() or None
            )

            self.current_file = saved_path

            self.label.setText(f"Saved:\n{saved_path}")
            self.status.setText("Image added ✔")

        except Exception as e:
            self.status.setText(f"Error: {e}")

    # ---------------- DRAWING INPUT ----------------
    def save_drawing(self):
        try:
            temp_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            temp_path = temp_file.name
            temp_file.close()

            self.canvas.export_image(temp_path)

            saved_path = save_to_pipeline(
                temp_path,
                task_type=self.category_box.currentText(),
                label=self.label_input.text().strip() or None
            )

            self.label.setText(f"Saved drawing:\n{saved_path}")
            self.status.setText("Drawing added ✔")

            self.canvas.clear()

        except Exception as e:
            self.status.setText(f"Error: {e}")