import sys
import os
import torch
import numpy as np
import pyttsx3
from PyQt6.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QFileDialog, QVBoxLayout,
    QHBoxLayout, QScrollArea, QFrame, QGridLayout, QSizePolicy, QMessageBox, QPushButton, QDialog
)
from PyQt6.QtGui import QPixmap, QPainter, QPen
from PyQt6.QtCore import Qt, QRect, QThread, pyqtSignal
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import matplotlib as mpl
import screeninfo
from mamba_model import MambaTransformerClassifier  # Ensure this file is present
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import queue
import threading
import csv

# Load trained model securely
num_classes = 43
model = MambaTransformerClassifier(num_classes=num_classes)
model.load_state_dict(torch.load("mamba_transformer_best.pth", map_location="cpu", weights_only=True), strict=False)
model.eval()


# Get screen DPI dynamically
screen = screeninfo.get_monitors()[0]  # Detect primary screen
dpi_scale = screen.width / 1920  # Scale based on 1080p resolution

# Dynamically adjust font sizes
mpl.rcParams.update({
    "axes.titlesize": 14 * dpi_scale,  # Scale Title Size
    "axes.labelsize": 12 * dpi_scale,  # Scale Labels
    "xtick.labelsize": 11 * dpi_scale,  # Scale X-axis Ticks
    "ytick.labelsize": 11 * dpi_scale,  # Scale Y-axis Ticks
    "legend.fontsize": 12 * dpi_scale,  # Scale Legend Size
    "figure.titlesize": 16 * dpi_scale,  # Scale Figure Title
    "figure.dpi": 100 * dpi_scale,  # Adjust DPI dynamically
})

# Initialize text-to-speech engine
tts_engine = pyttsx3.init()

class TTSWorker(QThread):
    """Thread for text-to-speech to prevent UI blocking."""
    finished = pyqtSignal()

    def __init__(self, text):
        super().__init__()
        self.text = text

    def run(self):
        tts_engine = pyttsx3.init()
        tts_engine.say(self.text)
        tts_engine.runAndWait()
        self.finished.emit()  # Signal when done


# Define class labels
class_labels = {
    0: "Speed Limit 20 km/h", 1: "Speed Limit 30 km/h", 2: "Speed Limit 50 km/h", 
    3: "Speed Limit 60 km/h", 4: "Speed Limit 70 km/h", 5: "Speed Limit 80 km/h", 
    6: "End of Speed Limit 80", 7: "Speed Limit 100 km/h", 8: "Speed Limit 120 km/h", 
    9: "No Overtaking", 10: "No Overtaking By Lorry", 11: "Intersection Ahead",
    12: "Priority Road Sign", 13: "Yield", 14: "Stop",
    15: "No Vehicles Permitted", 16: "No Trucks permitted", 17: "No Entry",
    18: "Danger Sign", 19: "Left Hand Curve", 20: "Right Hand Curve",
    21: "Double Curve", 22: "Bumpy Road", 23: "Slippery Road", 
    24: "Road Narrows From Right", 25: "Road Work Under Process", 26: "Traffic Signals Ahead",
    27: "Pedestrian Crossing", 28: "School Zone or Children Crossing", 29: "Cycles Crossing",
    30: "Slipperiness Due To Snow or Ice", 31: "Deer Crossing Area", 32: "End Of All Restrictions",
    33: "Turn Right Ahead", 34: "Turn Left Ahead", 35: "Go Straight",
    36: "Go Straight or Turn Right Ahead", 37: "Go Straight or Turn Left Ahead", 38: "Pass on Right",
    39: "Pass on Left", 40: "RoundAbout", 41: "End of No Overtaking", 42: "End of No Overtaking by Lorries",
}
  # Update with actual labels

# Image transformation pipeline
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
])

class AccuracyBar(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.accuracy = 0
        self.dark_mode = True  # Default is dark mode

    def setAccuracy(self, accuracy):
        self.accuracy = accuracy
        self.update()

    def setDarkMode(self, dark_mode):
        """ Updates color scheme based on dark mode setting """
        self.dark_mode = dark_mode
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        line_color = Qt.GlobalColor.white if self.dark_mode else Qt.GlobalColor.black
        marker_color = Qt.GlobalColor.red if self.dark_mode else Qt.GlobalColor.blue
        text_color = Qt.GlobalColor.white if self.dark_mode else Qt.GlobalColor.black

        pen = QPen(line_color, 4)
        painter.setPen(pen)
        painter.drawLine(10, 20, self.width() - 10, 20)

        marker_x = int(10 + (self.accuracy / 100) * (self.width() - 20))
        pen.setColor(marker_color)
        pen.setWidth(8)
        painter.setPen(pen)
        painter.drawPoint(marker_x, 20)

        painter.setPen(text_color)
        painter.drawText(QRect(0, 30, 50, 20), Qt.AlignmentFlag.AlignLeft, "0%")
        painter.drawText(QRect(self.width() - 50, 30, 50, 20), Qt.AlignmentFlag.AlignRight, "100%")
        painter.drawText(QRect(self.width() // 2 - 25, 30, 50, 20), Qt.AlignmentFlag.AlignCenter, f"{self.accuracy:.1f}%")

class TrafficSignClassifier(QWidget):
    def __init__(self):
        super().__init__()
        self.dark_mode = True
        self.image_paths = []
        self.initUI()
        self.tts_threads = []  # ✅ Store active TTS threads
        self.tts_queue = queue.Queue()
        self.tts_thread = threading.Thread(target=self.process_speech_queue, daemon=True)
        self.tts_thread.start()
        self.predictions = {}

    def save_results_to_csv(self):
        """Saves batch classification results to a CSV file."""
        if not self.predictions:
            QMessageBox.warning(self, "No Data", "No classification results available to save.")
            return

        file_path, _ = QFileDialog.getSaveFileName(self, "Save Report", "", "CSV Files (*.csv)")
        if not file_path:
            return

        with open(file_path, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["Image File", "Predicted Class", "Accuracy (%)"])
            for image_path, (predicted_text, accuracy) in self.predictions.items():
                writer.writerow([os.path.basename(image_path), predicted_text, f"{accuracy:.2f}"])

        QMessageBox.information(self, "Report Saved", f"Results saved to {file_path}")


    def process_speech_queue(self):
        """Continuously processes speech requests from the queue."""
        tts_engine = pyttsx3.init()
        while True:
            text = self.tts_queue.get()
            if text is None:
                break  # Exit loop if None is received
            tts_engine.say(text)
            tts_engine.runAndWait()


    def speak(self, text):
         self.tts_queue.put(text)


    def initUI(self):
        self.setWindowTitle("Traffic Sign Classifier")
        self.setGeometry(100, 100, 900, 600)

        # Define buttons first
        self.upload_button = QPushButton("Upload Images", self)
        self.upload_button.clicked.connect(self.upload_images)

        self.dark_mode_button = QPushButton("Toggle Dark Mode", self)
        self.dark_mode_button.clicked.connect(self.toggle_dark_mode)

        self.set_dark_mode_styles()

        # Scroll area for displaying multiple images
        self.scroll_area = QScrollArea(self)
        self.scroll_area.setWidgetResizable(True)
        self.results_container = QWidget()
        self.results_layout = QGridLayout(self.results_container)  # ✅ Use GridLayout here
        self.results_container.setLayout(self.results_layout)  # ✅ Ensure layout is applied
        self.scroll_area.setWidget(self.results_container)

        layout = QVBoxLayout()
        layout.addWidget(self.upload_button)
        layout.addWidget(self.dark_mode_button)
        layout.addWidget(self.scroll_area)

        #analytics
        # Button to Show Class Distribution
        self.analytics_button = QPushButton("Show Analytics", self)
        self.analytics_button.clicked.connect(self.show_class_distribution)
        layout.addWidget(self.analytics_button)

        # Button to Show Accuracy Trend
        self.accuracy_trend_button = QPushButton("Show Accuracy Trend", self)
        self.accuracy_trend_button.clicked.connect(self.show_accuracy_trend)
        layout.addWidget(self.accuracy_trend_button)



        self.setLayout(layout)
        #to save csv file
        self.save_csv_button = QPushButton("Save Results to CSV", self)
        self.save_csv_button.clicked.connect(self.save_results_to_csv)
        layout.addWidget(self.save_csv_button)



    def set_dark_mode_styles(self):
        """Applies a professional theme for both dark and light mode"""
        if self.dark_mode:
            self.setStyleSheet("""
                background-color: #121212; 
                color: white; 
                font-size: 16px;
                border-radius: 10px;
            """)
            self.upload_button.setStyleSheet("""
                QPushButton { 
                    background-color: #1E88E5; 
                    color: white; 
                    padding: 8px; 
                    border-radius: 5px;
                }
                QPushButton:hover { background-color: #1565C0; }
            """)
            self.dark_mode_button.setStyleSheet("""
                QPushButton { 
                    background-color: #444; 
                    color: white; 
                    padding: 8px; 
                    border-radius: 5px;
                }
                QPushButton:hover { background-color: #666; }
            """)
        else:
            self.setStyleSheet("""
                background-color: white; 
                color: black; 
                font-size: 16px;
                border-radius: 10px;
            """)
            self.upload_button.setStyleSheet("""
                QPushButton { 
                    background-color: #0D47A1; 
                    color: white; 
                    padding: 8px; 
                    border-radius: 5px;
                }
                QPushButton:hover { background-color: #1565C0; }
            """)
            self.dark_mode_button.setStyleSheet("""
                QPushButton { 
                    background-color: #DDD; 
                    color: black; 
                    padding: 8px; 
                    border-radius: 5px;
                }
                QPushButton:hover { background-color: #BBB; }
            """)

    def show_matplotlib_figure(self, figure):
        """Displays a Matplotlib figure inside a PyQt6 dialog with a grey-themed toolbar."""
        dialog = QDialog(self)
        dialog.setWindowTitle("Analytics")
        dialog.setGeometry(200, 200, 900, 650)

        layout = QVBoxLayout()
        canvas = FigureCanvas(figure)
        toolbar = NavigationToolbar(canvas, dialog)

        # ✅ Apply Grey Theme to Matplotlib Toolbar
        toolbar.setStyleSheet("""
            QToolBar {
                background-color: #444;
                border: 1px solid #666;
            }
            QToolButton {
                color: white;
                background-color: #666;
                border: none;
                padding: 5px;
                margin: 2px;
            }
            QToolButton:hover {
                background-color: #888;
            }
            QLabel {
                color: white;
            }
        """)

        layout.addWidget(toolbar)
        layout.addWidget(canvas)

        dialog.setLayout(layout)
        dialog.exec()
        
    def toggle_dark_mode(self):
        """Toggles between dark mode and light mode"""
        self.dark_mode = not self.dark_mode
        self.set_dark_mode_styles()

    def upload_images(self):
        file_paths, _ = QFileDialog.getOpenFileNames(self, "Select Images", "", "Images (*.png *.jpg *.jpeg)")
        if file_paths:
            self.image_paths = file_paths
            self.display_predictions()

    def get_dpi_scaling(self):
        """Dynamically get the DPI scaling factor for UI elements."""
        try:
            import screeninfo
            screen = screeninfo.get_monitors()[0]
            return screen.width / 1920  # Scale based on 1080p reference
        except:
            return 1  # Default scale for unknown screens

    def show_class_distribution(self):
        """Displays a bar chart showing the distribution of detected traffic signs with UI scaling."""
        if not self.predictions:
            QMessageBox.warning(self, "No Data", "No classification results available.")
            return

        # Count occurrences of each traffic sign class
        class_counts = {}
        for _, (label, _) in self.predictions.items():
            class_counts[label] = class_counts.get(label, 0) + 1

        if not class_counts:  # ✅ Prevents division by zero
            QMessageBox.warning(self, "No Data", "No valid traffic sign classifications found.")
            return

        # ✅ Apply Matplotlib UI Scaling
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(class_counts.keys(), class_counts.values(), color="skyblue")
        
        # ✅ Dynamically Adjust Text Sizes
        dpi_scale = self.get_dpi_scaling()
        ax.set_xticklabels(class_counts.keys(), rotation=45, ha="right", fontsize=11 * dpi_scale)
        ax.set_yticklabels(ax.get_yticks(), fontsize=11 * dpi_scale)
        ax.set_title("Detected Traffic Sign Distribution", fontsize=14 * dpi_scale)
        ax.set_xlabel("Traffic Sign Class", fontsize=12 * dpi_scale)
        ax.set_ylabel("Frequency", fontsize=12 * dpi_scale)
        ax.grid(axis="y", linestyle="--", alpha=0.7)

        self.show_matplotlib_figure(fig)  # ✅ Fix plt.show()

    def show_accuracy_trend(self):
        """Displays a bar chart of prediction accuracy trends."""
        if not self.predictions:
            QMessageBox.warning(self, "No Data", "No classification results available.")
            return

        image_files = []
        accuracies = []

        for img_path, (_, accuracy) in self.predictions.items():
            if accuracy is not None and 0 <= accuracy <= 100:  # ✅ Prevent invalid values
                image_files.append(os.path.basename(img_path))
                accuracies.append(accuracy)

        if not accuracies:  # ✅ Prevent empty plots
            QMessageBox.warning(self, "No Data", "No valid accuracy data available.")
            return

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(image_files, accuracies, color="green")
        ax.set_xticklabels(image_files, rotation=45, ha="right")
        ax.set_title("Prediction Accuracy Trend")
        ax.set_xlabel("Image File")
        ax.set_ylabel("Accuracy (%)")
        ax.set_ylim(0, 100)
        ax.grid(axis="y", linestyle="--", alpha=0.7)

        self.show_matplotlib_figure(fig)  # ✅ Fix plt.show()



    from PyQt6.QtWidgets import QSizePolicy, QGridLayout

    def display_predictions(self):
        """Dynamically arranges images in a grid while ensuring previous results are properly cleared."""

        # ✅ Properly Clear Previous Layout
        while self.results_layout.count():
            item = self.results_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()  # ✅ Proper widget deletion

        if not self.image_paths:
            return  # No images, so exit

        num_columns = max(1, self.width() // 300)
        row, col = 0, 0

        for image_path in self.image_paths:
            prediction_widget = self.create_prediction_widget(image_path)
            prediction_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
            self.results_layout.addWidget(prediction_widget, row, col)

            col += 1
            if col >= num_columns:
                col = 0
                row += 1

        self.results_container.adjustSize()  # Ensure container resizes
        self.scroll_area.setWidget(self.results_container)  


    def create_prediction_widget(self, image_path):
        """Creates a clean, professional UI block for each image prediction."""
        frame = QFrame(self)
        frame.setFrameShape(QFrame.Shape.StyledPanel)
        frame.setStyleSheet("""
            QFrame {
                border: 2px solid #1E88E5; 
                border-radius: 10px;
                padding: 10px;
                background-color: #1A1A1A;
            }
        """ if self.dark_mode else """
            QFrame {
                border: 2px solid #0D47A1; 
                border-radius: 10px;
                padding: 10px;
                background-color: white;
            }
        """)

        layout = QHBoxLayout(frame)

        # Image Box (Centered)
        image_label = QLabel(frame)
        pixmap = QPixmap(image_path).scaled(200, 200, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        image_label.setPixmap(pixmap)
        image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Process Image
        image = Image.open(image_path).convert("RGB")
        image = transform(image).unsqueeze(0)
        output = model(image)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        confidence, predicted_class = torch.max(probabilities, dim=1)

        predicted_text = class_labels.get(predicted_class.item(), "Unknown Traffic Sign")
        accuracy_percentage = confidence.item() * 100

        # Prediction & Accuracy Labels (Updated color)
        text_color = "white" if self.dark_mode else "#0D47A1"  # Dark blue for light mode
        prediction_label = QLabel(f"Prediction: {predicted_text}", frame)
        prediction_label.setStyleSheet(f"font-size: 18px; font-weight: bold; color: {text_color};")

        accuracy_label = QLabel(f"Accuracy: {accuracy_percentage:.2f}%", frame)
        accuracy_label.setStyleSheet("font-size: 16px; color: #FBC02D;")

        # Improved Accuracy Bar
        accuracy_bar = AccuracyBar()
        accuracy_bar.setFixedHeight(50)
        accuracy_bar.setAccuracy(accuracy_percentage)
        accuracy_bar.setDarkMode(self.dark_mode)

        # Arrange Widgets
        text_layout = QVBoxLayout()
        text_layout.addWidget(prediction_label)
        text_layout.addWidget(accuracy_label)
        text_layout.addWidget(accuracy_bar)

        layout.addWidget(image_label, alignment=Qt.AlignmentFlag.AlignCenter)
        layout.addLayout(text_layout)

        self.speak(f"Prediction: {predicted_text}. Accuracy: {accuracy_percentage:.2f} percent")

        self.predictions[image_path] = (predicted_text, accuracy_percentage)

        return frame


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = TrafficSignClassifier()
    window.show()
    sys.exit(app.exec())
