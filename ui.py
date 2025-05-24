from PySide6.QtWidgets import QMainWindow, QVBoxLayout, QTabWidget, QWidget, QLabel, QPushButton, QProgressBar, QSlider, QLineEdit, QCheckBox, QComboBox, QFileDialog, QListWidget, QListWidgetItem, QMessageBox,QGraphicsView,QGraphicsScene,QGraphicsPixmapItem,QHBoxLayout
from PySide6.QtGui import QPixmap, QImage, QPainter, QColor, QPalette, QMouseEvent, QPen, QBrush
from PySide6.QtCore import Qt, QTimer, QRectF, QThread, Signal
from workers import VideoProcessor, BatchProcessor
from utils import generate_matte_from_image, save_exr, create_mp4_from_pngs
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "BiRefNet"))
from models.birefnet import BiRefNet
import cv2
from PIL import Image, ImageQt
import logging
import subprocess
import torch
import numpy as np
import sys
import shutil
import tempfile
# Add these imports for SAM
from segment_anything import sam_model_registry, SamPredictor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "BiRefNet"))

class ImageProcessor(QThread):
    finished = Signal(object, str)
    error = Signal(str)

    def __init__(self, input_image, model, device, output_path, output_format):
        super().__init__()
        self.input_image = input_image
        self.model = model
        self.device = device
        self.output_path = output_path
        self.output_format = output_format

    def run(self):
        try:
            matte = generate_matte_from_image(self.input_image, self.model, self.device)
            matte_img = Image.fromarray(matte.astype(np.uint8))
            matte_img.save(self.output_path, format=self.output_format)
            self.finished.emit(matte_img, self.output_path)
        except Exception as e:
            self.error.emit(str(e))

class MatteGeneratorApp(QMainWindow):
    def __init__(self):
        super().__init__()
        # Configure logging to capture errors in a file
        logging.basicConfig(
            filename="mattegen_debug.log",
            level=logging.DEBUG,
            format="%(asctime)s - %(levelname)s - %(message)s"
        )
        logging.info("MatteGeneratorApp initialized.")
        self.setWindowTitle("PyTools MatteGen 0.2")
        self.setGeometry(100, 100, 800, 600)

        # Variables
        self.input_path = None
        self.is_video = False
        self.cap = None
        self.timer = QTimer()
        self.input_image = None
        self.output_image = None
        self.temp_dir = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = None
        self.worker = None
        self.batch_size = 4  # Default batch size
        self.zoom_factor = 1.0  # Initial zoom level
        self.current_frame = 0  # storing the frame index
        self.fps = 30  # set frames per second
        self.filename_prefix = ""
        self.weights_path = os.path.join("weights", "BiRefNet_HR-matting-epoch_135.pth")  # Default model path
        self.jpg_sequence = []  # list to store the jpg
        self.jpg_index = 0  # index to start
        self.user_mask = None  # Store user-selected mask

        # SAM Variables
        self.sam_predictor = None
        self.sam_loaded = False
        self.sam_checkpoint = os.path.join("weights", "sam_vit_h_4b8939.pth")  # Correct path to weights folder
        self.sam_model_type = "vit_h"  # or "vit_b", "vit_l" depending on your checkpoint

        # Create a tab widget
        self.tab_widget = QTabWidget()
        self.main_tab = QWidget()
        self.batch_tab = QWidget()

        self.tab_widget.addTab(self.main_tab, "Single File")
        self.tab_widget.addTab(self.batch_tab, "Batch Processing")

        # Main layout
        main_layout = QVBoxLayout()
        main_layout.addWidget(self.tab_widget)

        # Set the main layout
        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

        # Call all set up methods for UI
        self.setup_ui()

    def setup_ui(self):
        """Setup the layout for the UI"""
        self.setup_main_tab()
        self.setup_batch_tab()
        self.load_model()
        self.load_sam_model()

    def setup_main_tab(self):
        """Setup the layout for the main tab"""
        main_tab_layout = QVBoxLayout()

        # Graphics View for Zoom/Pan
        self.graphics_view = QGraphicsView(self)
        self.graphics_scene = QGraphicsScene(self)
        self.graphics_view.setScene(self.graphics_scene)
        self.graphics_view.setRenderHint(QPainter.Antialiasing)  # Corrected Line
        self.graphics_view.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)  # Zoom around mouse

        main_tab_layout.addWidget(self.graphics_view)

        # Preview window (graphics item to display the image)
        self.image_item = QGraphicsPixmapItem()
        self.graphics_scene.addItem(self.image_item)
        self.preview_image = None  # To store QImage

        # Slider for input/output comparison
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(100)
        self.slider.setValue(50)
        self.slider.valueChanged.connect(self.update_preview)
        main_tab_layout.addWidget(self.slider)

        # Load Model Dropdown
        load_model_layout = QHBoxLayout()
        self.model_dropdown = QComboBox()
        self.populate_model_dropdown()  # Populate dropdown with weights
        self.model_dropdown.currentIndexChanged.connect(self.load_model_from_dropdown)
        load_model_layout.addWidget(QLabel("Select Model:"))
        load_model_layout.addWidget(self.model_dropdown)

        # Add button to set weights folder
        self.set_weights_folder_button = QPushButton("Set Weights Folder")
        self.set_weights_folder_button.clicked.connect(self.set_weights_folder)
        load_model_layout.addWidget(self.set_weights_folder_button)

        main_tab_layout.addLayout(load_model_layout)

        # Input area
        input_layout = QHBoxLayout()

        self.drop_label = QLabel("Drag and drop an image or video file here")
        self.drop_label.setAlignment(Qt.AlignCenter)
        self.drop_label.setStyleSheet("border: 2px dashed #aaa; padding: 10px;")
        self.drop_label.setAcceptDrops(True)
        input_layout.addWidget(self.drop_label)

        self.select_file_button = QPushButton("Browse")
        self.select_file_button.clicked.connect(self.select_file)
        input_layout.addWidget(self.select_file_button)

        main_tab_layout.addLayout(input_layout)

        # Playback controls
        playback_layout = QHBoxLayout()

        self.play_pause_button = QPushButton("Play")
        self.play_pause_button.clicked.connect(self.toggle_play_pause)
        self.play_pause_button.setEnabled(False)
        playback_layout.addWidget(self.play_pause_button)

        self.frame_slider = QSlider(Qt.Horizontal)
        self.frame_slider.setMinimum(0)
        self.frame_slider.setMaximum(0)  # Set dynamically when video is loaded
        self.frame_slider.valueChanged.connect(self.set_frame)
        self.frame_slider.setEnabled(False)
        playback_layout.addWidget(self.frame_slider)

        main_tab_layout.addLayout(playback_layout)

        # Output options
        output_options_layout = QVBoxLayout()
        self.filename_prefix_label = QLabel("Filename Prefix:")
        self.filename_prefix_edit = QLineEdit()
        self.filename_prefix_edit.setPlaceholderText("Enter prefix")
        output_options_layout.addWidget(self.filename_prefix_label)
        output_options_layout.addWidget(self.filename_prefix_edit)
        main_tab_layout.addLayout(output_options_layout)

        # Add output format option
        output_format_layout = QHBoxLayout()
        self.output_format_label = QLabel("Output Format:")
        self.output_format_combo = QComboBox()
        self.output_format_combo.addItems(["PNG", "JPEG", "TIFF"])
        output_format_layout.addWidget(self.output_format_label)
        output_format_layout.addWidget(self.output_format_combo)
        main_tab_layout.addLayout(output_format_layout)

        # Add looping options
        self.loop_checkbox = QCheckBox("Loop Video")
        main_tab_layout.addWidget(self.loop_checkbox)

        # Progress bar
        self.progress_bar = QProgressBar()
        main_tab_layout.addWidget(self.progress_bar)
        self.progress_label = QLabel("Estimated time remaining: N/A")
        main_tab_layout.addWidget(self.progress_label)

        # Generate button
        self.generate_button = QPushButton("Generate Matte")
        self.generate_button.clicked.connect(self.generate_matte)
        main_tab_layout.addWidget(self.generate_button)

        # Stop button
        self.stop_button = QPushButton("Stop Processing")
        self.stop_button.clicked.connect(self.stop_processing)
        self.stop_button.setEnabled(False)  # Initially disabled
        main_tab_layout.addWidget(self.stop_button)

        # View Output Folder Button
        self.view_output_button = QPushButton("View Output Folder")
        self.view_output_button.clicked.connect(self.view_output_folder)
        main_tab_layout.addWidget(self.view_output_button)

        # === Move Advanced Matte Buttons Here ===
        advanced_buttons_layout = QHBoxLayout()
        self.batch_erosion_button = QPushButton("Erosion Effect")
        self.batch_erosion_button.clicked.connect(self.apply_erosion_effect)
        advanced_buttons_layout.addWidget(self.batch_erosion_button)
        self.batch_dilation_button = QPushButton("Dilation Effect")
        self.batch_dilation_button.clicked.connect(self.apply_dilation_effect)
        advanced_buttons_layout.addWidget(self.batch_dilation_button)
        self.batch_feather_button = QPushButton("Feather Effect")
        self.batch_feather_button.clicked.connect(self.apply_feather_effect)
        advanced_buttons_layout.addWidget(self.batch_feather_button)
        self.batch_smoothing_button = QPushButton("Smoothing Effect")
        self.batch_smoothing_button.clicked.connect(self.apply_smoothing_effect)
        advanced_buttons_layout.addWidget(self.batch_smoothing_button)
        main_tab_layout.addLayout(advanced_buttons_layout)
        # === End Advanced Matte Buttons ===

        # Add Save Matte Button
        self.save_matte_button = QPushButton("Save Matte")
        self.save_matte_button.clicked.connect(self.save_current_matte)
        main_tab_layout.addWidget(self.save_matte_button)

        # Add Select Object Button
        self.select_object_button = QPushButton("Select Object/Person for Matte")
        self.select_object_button.clicked.connect(self.start_object_selection)
        main_tab_layout.addWidget(self.select_object_button)

        # Add a variable to store user points for SAM
        self.sam_points = []
        self.sam_point_mode = False

        # Studio Version Text\
        studio_version_label = QLabel("PyTools MatteGen 0.2")
        studio_version_label.setAlignment(Qt.AlignCenter)
        main_tab_layout.addWidget(studio_version_label)

        self.main_tab.setLayout(main_tab_layout)

    def setup_batch_tab(self):
        """Setup the layout for the batch processing tab"""
        batch_tab_layout = QVBoxLayout()

        # Drag-and-Drop Area
        self.drag_drop_label = QLabel("Drag and drop folders here")
        self.drag_drop_label.setAlignment(Qt.AlignCenter)
        self.drag_drop_label.setStyleSheet("border: 2px dashed #aaa; padding: 20px;")
        self.drag_drop_label.setAcceptDrops(True)
        self.drag_drop_label.dragEnterEvent = self.drag_enter_event
        self.drag_drop_label.dropEvent = self.drop_event
        batch_tab_layout.addWidget(self.drag_drop_label)

        # Select files list
        self.file_list_widget = QListWidget()
        batch_tab_layout.addWidget(self.file_list_widget)

        # Remove selected button
        self.remove_selected_button = QPushButton("Remove Selected")
        self.remove_selected_button.clicked.connect(self.remove_selected_items)
        batch_tab_layout.addWidget(self.remove_selected_button)

        # Output options
        output_options_layout = QVBoxLayout()
        self.filename_prefix_label2 = QLabel("Filename Prefix:")
        self.filename_prefix_edit2 = QLineEdit()
        self.filename_prefix_edit2.setPlaceholderText("Enter prefix")
        output_options_layout.addWidget(self.filename_prefix_label2)
        output_options_layout.addWidget(self.filename_prefix_edit2)
        batch_tab_layout.addLayout(output_options_layout)

        # Add output format option
        output_format_layout = QHBoxLayout()
        self.output_format_label2 = QLabel("Output Format:")
        self.output_format_combo2 = QComboBox()
        self.output_format_combo2.addItems(["PNG", "JPEG", "TIFF"])
        output_format_layout.addWidget(self.output_format_label2)
        output_format_layout.addWidget(self.output_format_combo2)
        batch_tab_layout.addLayout(output_format_layout)

        # Batch Process button
        self.batch_process_button = QPushButton("Process Batch")
        self.batch_process_button.clicked.connect(self.process_batch)
        batch_tab_layout.addWidget(self.batch_process_button)

        # Progress bar for batch processing
        self.batch_progress_bar = QProgressBar()
        batch_tab_layout.addWidget(self.batch_progress_bar)
        self.batch_progress_label = QLabel("Estimated time remaining: N/A")
        batch_tab_layout.addWidget(self.batch_progress_label)

        # View Output Folder Button
        self.view_output_button = QPushButton("View Output Folder")
        self.view_output_button.clicked.connect(self.view_output_folder)
        batch_tab_layout.addWidget(self.view_output_button)

        self.batch_tab.setLayout(batch_tab_layout)

    def load_model(self):
        """Load the selected model from the local weights folder."""
        weights_path = self.weights_path  # get the path to load
        logging.info(f"Attempting to load model from: {weights_path}")
        if not os.path.exists(weights_path):
            msg = f"Weights not found at {weights_path}"
            logging.error(msg)
            if hasattr(self, 'drop_label'):
                self.drop_label.setText(msg)
            return

        try:
            self.model = BiRefNet(bb_pretrained=False)
            state_dict = torch.load(weights_path, map_location=self.device)
            self.model.load_state_dict(state_dict, strict=False)

            self.model.to(self.device)
            self.model.eval()

            msg = f"Model successfully loaded from: {weights_path}"
            logging.info(msg)
            if hasattr(self, 'drop_label'):
                self.drop_label.setText(msg)

        except Exception as e:
            msg = f"Error loading model: {e}"
            logging.exception(msg)
            if hasattr(self, 'drop_label'):
                self.drop_label.setText(msg)
            else:
                QMessageBox.critical(self, "Error", msg)

    def load_sam_model(self):
        """Load the Segment Anything Model (SAM) for interactive segmentation."""
        try:
            sam = sam_model_registry[self.sam_model_type](checkpoint=self.sam_checkpoint)
            self.sam_predictor = SamPredictor(sam)
            self.sam_loaded = True
            print("SAM loaded successfully.")
        except Exception as e:
            self.sam_loaded = False
            print(f"Failed to load SAM: {e}")

    def set_weights_folder(self):
        """Open a dialog to set the weights folder path."""
        folder_path = QFileDialog.getExistingDirectory(self, "Select Weights Folder")
        if folder_path:
            self.weights_folder = folder_path
            self.populate_model_dropdown()

    def populate_model_dropdown(self):
        """Populate the model dropdown with files from the weights folder."""
        self.weights_folder = getattr(self, 'weights_folder', "weights")  # Default folder
        if not os.path.exists(self.weights_folder):
            logging.warning(f"Weights folder not found: {self.weights_folder}")
            self.drop_label.setText("Weights folder not found.")
            return

        self.model_dropdown.clear()
        for file_name in sorted(os.listdir(self.weights_folder)):
            if file_name.endswith(".pth") or file_name.endswith(".pt"):
                self.model_dropdown.addItem(file_name)

    def load_model_from_dropdown(self):
        """Load the selected model from the dropdown."""
        selected_model = self.model_dropdown.currentText()
        if not selected_model:
            logging.warning("No model selected.")
            self.drop_label.setText("No model selected.")
            return

        weights_path = os.path.join(self.weights_folder, selected_model)
        if not os.path.exists(weights_path):
            logging.warning(f"Selected model not found: {weights_path}")
            self.drop_label.setText(f"Selected model not found: {weights_path}")
            return

        try:
            self.model = BiRefNet(bb_pretrained=False)
            state_dict = torch.load(weights_path, map_location=self.device)
            self.model.load_state_dict(state_dict, strict=False)

            self.model.to(self.device)
            self.model.eval()

            self.drop_label.setText(f"Loaded model: {selected_model}")
            logging.info(f"Loaded model: {selected_model}")

        except Exception as e:
            msg = f"Error loading model: {e}"
            logging.error(msg)
            self.drop_label.setText(msg)

    def select_model_path(self):
        """Open a file dialog to select a model weights file."""
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Model Weights", "",
                                                   "PyTorch Model Files (*.pth *.pt)")
        if file_path:
            self.weights_path = file_path
            self.drop_label.setText(f"Selected model: {os.path.basename(self.weights_path)}")
            self.load_model()

    def select_file(self):
        """Open a file dialog to select an image or video file."""
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Image/Video", "",
                                                   "Images (*.jpg *.png);;Videos (*.mp4 *.avi *.mov);;JPG Sequence (*.jpg)")
        if file_path:
            self.load_file(file_path)  # call load file and pass file path

    def load_file(self, file_path):
        """Load the selected file."""
        # Check if a directory was selected instead of a file
        if os.path.isdir(file_path):  # Check if it is a directory
            self.load_jpg_sequence(file_path)
            return

        self.input_path = file_path
        self.drop_label.setText(f"Selected: {os.path.basename(self.input_path)}")
        self.is_video = self.input_path.lower().endswith(('.mp4', '.avi', '.mov'))

        # Enable/Disable button
        self.play_pause_button.setEnabled(self.is_video)
        self.is_playing = False
        self.play_pause_button.setText("Play")

        if self.is_video:
            try:
                self.cap = cv2.VideoCapture(self.input_path)
                if not self.cap.isOpened():
                    self.drop_label.setText("Could not open video file.")
                    self.is_video = False
                    self.frame_slider.setEnabled(False)
                    return

                self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
                self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))  # Get frames per second
                self.timer.setInterval(int(1000 / self.fps))  # Setting speed
                self.frame_slider.setMaximum(self.frame_count - 1)  # setting maximum range to slider
                self.frame_slider.setEnabled(True)  # enable frame slider
                self.current_frame = 0  # Start at the beginning
                self.set_frame(0)  # update frame
                self.timer.timeout.connect(self.next_frame)  # Call it so user will see it

            except Exception as e:
                self.drop_label.setText(f"Error opening video: {e}")
                self.is_video = False
                self.frame_slider.setEnabled(False)  # keep slider disabled
                logging.exception("Error opening video:")

        else:
            try:
                self.input_image = Image.open(self.input_path).convert("RGB")
                self.show_image_preview(self.input_image)

            except FileNotFoundError:
                self.drop_label.setText("File not found.")
            except Image.UnidentifiedImageError:
                self.drop_label.setText("Could not open image file.")
            except Exception as e:
                self.drop_label.setText(f"Error opening image: {e}")
                logging.exception("Error opening image:")

    def show_image_preview(self, image):
        """Display the selected image in the preview window and clear overlays."""
        q_img = self.pil_to_qimage(image)
        pixmap = QPixmap.fromImage(q_img)
        self.image_item.setPixmap(pixmap)
        self.graphics_scene.setSceneRect(QRectF(pixmap.rect()))
        self.graphics_view.fitInView(self.graphics_scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
        # Clear all overlays (SAM mask, points, etc.)
        for item in self.graphics_scene.items():
            if item is not self.image_item:
                self.graphics_scene.removeItem(item)
        self.user_mask = None
        self.sam_points = []

    def pil_to_qimage(self, image):
        """Convert a PIL Image to QImage with correct color."""
        if image.mode == "RGB":
            data = image.tobytes("raw", "RGB")
            qimg = QImage(data, image.width, image.height, QImage.Format_RGB888)
        elif image.mode == "L":
            data = image.tobytes("raw", "L")
            qimg = QImage(data, image.width, image.height, QImage.Format_Grayscale8)
        elif image.mode == "RGBA":
            data = image.tobytes("raw", "RGBA")
            qimg = QImage(data, image.width, image.height, QImage.Format_RGBA8888)
        else:
            image = image.convert("RGBA")
            data = image.tobytes("raw", "RGBA")
            qimg = QImage(data, image.width, image.height, QImage.Format_RGBA8888)
        return qimg

    def generate_matte(self):
        """Generate matte for the selected file."""
        logging.info("Starting matte generation.")
        if not self.input_path:
            msg = "No input file selected."
            logging.error(msg)
            self.drop_label.setText(msg)
            return

        if self.model is None:
            msg = "No model loaded. Please select a valid model."
            logging.error(msg)
            self.drop_label.setText(msg)
            return

        try:
            # Create a "mattes" folder in the same directory as the input file
            output_folder = os.path.join(os.path.dirname(self.input_path), "mattes")
            os.makedirs(output_folder, exist_ok=True)
            filename_prefix = self.filename_prefix_edit.text()
            output_format = self.output_format_combo.currentText()

            if self.is_video:
                # passing all
                self.process_video(output_folder, filename_prefix, output_format)
            else:
                self.process_image(output_folder, filename_prefix, output_format)  # passing the variable
            logging.info("Matte generation completed successfully.")
        except Exception as e:
            msg = f"Error during matte generation: {e}"
            logging.exception(msg)
            self.drop_label.setText(msg)

    def process_image(self, output_folder, filename_prefix, output_format):
        """Process a single image and save the matte."""
        filename_prefix = filename_prefix if filename_prefix else "Matte"
        if self.input_image is None:
            self.drop_label.setText("No valid image loaded.")
            return
        base_filename = os.path.splitext(os.path.basename(self.input_path))[0]
        filename_prefix = filename_prefix + "_" if filename_prefix else ""
        output_filename = f"{filename_prefix}{base_filename}.{output_format.lower()}"
        output_path = os.path.join(output_folder, output_filename)

        self.generate_button.setEnabled(False)
        self.worker = ImageProcessor(self.input_image, self.model, self.device, output_path, output_format)
        self.worker.finished.connect(self.on_image_processed)
        self.worker.error.connect(self.on_processing_error)
        self.worker.start()

    def on_image_processed(self, matte_img, output_path):
        """Handle the completion of image processing."""
        self.output_image = matte_img
        self.update_preview()
        self.drop_label.setText(f"Matte saved to: {output_path}")
        self.generate_button.setEnabled(True)

    def process_video(self, output_folder, filename_prefix, output_format):
        """Process a video and save mattes for each frame."""
        logging.info(f"Starting video processing for: {self.input_path}")
        if self.cap is None:
            self.drop_label.setText("No valid video loaded.")
            return

        self.filename_prefix = filename_prefix  # passing the value of file name
        self.output_format = output_format  # pass the value of output format
        self.generate_button.setEnabled(False)
        self.stop_button.setEnabled(True)  # Enable stop button when processing starts
        self.progress_bar.setValue(0)
        self.progress_label.setText("Estimated time remaining: N/A")
        self.play_pause_button.setEnabled(False)
        self.frame_slider.setEnabled(False)  # keep slider disabled

        try:
            # Create a worker thread for video processing
            if hasattr(self, "user_mask") and self.user_mask is not None:
                self.worker = VideoProcessor(self.input_path, output_folder, self.model, self.device, self.batch_size,
                                             filename_prefix, output_format, user_mask=self.user_mask)
            else:
                self.worker = VideoProcessor(self.input_path, output_folder, self.model, self.device, self.batch_size,
                                             filename_prefix, output_format)
            self.worker.progress_updated.connect(self.update_progress)  # Connect to update progress and time
            self.worker.processing_finished.connect(self.on_processing_finished)
            self.worker.error_occurred.connect(self.on_processing_error)  # Connect error signal
            self.worker.start()
            logging.info("Video processing completed successfully.")
        except Exception as e:
            msg = f"Error during video processing: {e}"
            logging.exception(msg)
            self.drop_label.setText(msg)

    def stop_processing(self):
        """Stop the ongoing processing."""
        if self.worker and self.worker.isRunning():
            self.worker.terminate()  # Terminate the worker thread
            self.worker = None
            self.generate_button.setEnabled(True)
            self.stop_button.setEnabled(False)
            self.progress_label.setText("Processing stopped.")
            logging.info("Processing stopped by the user.")

    def update_progress(self, progress, time_remaining):
        """Update progress bar and estimated time remaining label."""
        self.progress_bar.setValue(progress)
        if time_remaining > 0:
            minutes, seconds = divmod(int(time_remaining), 60)
            self.progress_label.setText(f"Estimated time remaining: {minutes:02d}:{seconds:02d}")
        else:
            self.progress_label.setText("Estimated time remaining: N/A")

    def update_preview(self):
        """Update the preview with the matte multiplied on top of the input image."""
        if self.input_image is None or self.output_image is None:
            return

        try:
            # Convert input image and matte to NumPy arrays
            input_array = np.array(self.input_image, dtype=np.float32) / 255.0  # Normalize to [0, 1]
            matte_array = np.array(self.output_image, dtype=np.float32) / 255.0  # Normalize to [0, 1]

            # Ensure the matte is a single-channel grayscale image
            if matte_array.ndim == 3:
                matte_array = matte_array[:, :, 0]

            # Multiply the matte with the input image
            multiplied_array = input_array * matte_array[:, :, None]  # Apply matte to all channels

            # Create a blended image based on the slider value
            slider_value = self.slider.value() / 100.0
            blended_array = (input_array * (1 - slider_value) + multiplied_array * slider_value)

            # Convert the blended array back to an image
            blended_array = (blended_array * 255).astype(np.uint8)  # Convert back to [0, 255]
            blended_image = Image.fromarray(blended_array)

            # Convert the blended image to QImage for display
            q_img = self.pil_to_qimage(blended_image)
            pixmap = QPixmap.fromImage(q_img)
            self.image_item.setPixmap(pixmap)
            self.graphics_scene.setSceneRect(QRectF(pixmap.rect()))
            self.graphics_view.fitInView(self.graphics_scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

        except Exception as e:
            logging.error(f"Error updating preview: {e}")

    def update_graphics_view(self, q_img):
        """Update the graphics view with a QImage."""
        pixmap = QPixmap.fromImage(q_img)
        self.image_item.setPixmap(pixmap)  # Set the pixmap to the image item
        self.graphics_scene.setSceneRect(QRectF(pixmap.rect()))  # Set the scene size
        self.graphics_view.fitInView(self.graphics_scene.sceneRect(),
                                     Qt.AspectRatioMode.KeepAspectRatio)  # Fit the content in the view

    def toggle_play_pause(self):
        """Toggle video playback."""
        if self.is_playing:
            self.timer.stop()
            self.play_pause_button.setText("Play")
        else:
            self.timer.start()  # timer to start as fps setting that load file sets
            self.play_pause_button.setText("Pause")
        self.is_playing = not self.is_playing

    def set_frame(self, frame_number):
        """Set the video to a specific frame."""
        if not self.is_video or self.cap is None:
            return

        self.current_frame = frame_number
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.input_image = Image.fromarray(frame)
            self.show_image_preview(self.input_image)
            self.frame_slider.setValue(frame_number)  # set to current value
        else:
            self.drop_label.setText("Error seeking to frame.")

    def next_frame(self):
        """Advance to the next frame."""
        if not self.is_video or self.cap is None or not self.is_playing:
            return

        next_frame = self.current_frame + 1
        if next_frame >= self.frame_count:
            if self.loop_checkbox.isChecked():
                next_frame = 0
            else:
                self.timer.stop()  # Stop the timer when it reach and loop at 0
                self.is_playing = False  # Update the video is playing
                self.play_pause_button.setText("Play")  # reset to play at 0
                return  # return to the method

        self.set_frame(next_frame)
        self.frame_slider.setValue(next_frame)
        self.current_frame = next_frame

    def view_output_folder(self):
        """Open the output folder in the file explorer."""
        if self.tab_widget.currentIndex() == 1:  # Check if the Batch Processing tab is active
            output_folder = os.path.join(self.input_folder_edit.text(), "mattes_batch")
        else:
            if self.input_path:
                output_folder = os.path.join(os.path.dirname(self.input_path), "mattes")
            else:
                QMessageBox.warning(self, "Warning", "No input file selected.")
                return

        if os.path.exists(output_folder):
            os.startfile(output_folder)  # Open the folder in the file explorer
        else:
            QMessageBox.warning(self, "Warning", "Output folder does not exist.")

    def drag_enter_event(self, event):
        """Handle drag enter event."""
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def drop_event(self, event):
        """Handle drop event."""
        for url in event.mimeData().urls():
            path = url.toLocalFile()
            if os.path.isdir(path):  # If it's a folder, add it to the list
                if not self.is_folder_already_added(path):
                    item = QListWidgetItem(path)
                    item.setData(Qt.UserRole, path)
                    self.file_list_widget.addItem(item)

    def is_folder_already_added(self, folder_path):
        """Check if a folder is already added to the list."""
        for i in range(self.file_list_widget.count()):
            if self.file_list_widget.item(i).data(Qt.UserRole) == folder_path:
                return True
        return False

    def remove_selected_items(self):
        """Remove selected items from the file list widget."""
        for item in self.file_list_widget.selectedItems():
            self.file_list_widget.takeItem(self.file_list_widget.row(item))

    def load_files_in_folders(self):
        """Load all supported files from all selected folders into the file list widget."""
        self.file_list_widget.clear()
        supported_extensions = ('.jpg', '.png', '.jpeg', '.tiff', '.bmp')
        for i in range(self.file_list_widget.count()):
            folder_path = self.file_list_widget.item(i).data(Qt.UserRole)
            try:
                for file_name in sorted(os.listdir(folder_path)):
                    if file_name.lower().endswith(supported_extensions):
                        item = QListWidgetItem(os.path.join(folder_path, file_name))
                        item.setData(Qt.UserRole, os.path.join(folder_path, file_name))  # Set full file path
                        self.file_list_widget.addItem(item)
            except PermissionError:
                logging.warning(f"Permission denied: {folder_path}")
                continue

    def process_batch(self):
        """Process all files from the selected folders."""
        # Always load the currently selected model from the dropdown before batch
        self.load_model_from_dropdown()
        selected_files = []
        output_folders = {}

        for i in range(self.file_list_widget.count()):
            item = self.file_list_widget.item(i)
            path = item.data(Qt.UserRole)

            if os.path.isdir(path):  # If it's a folder, add all files in the folder
                supported_extensions = ('.jpg', '.png', '.jpeg', '.tiff', '.bmp')
                try:
                    for file_name in sorted(os.listdir(path)):
                        file_path = os.path.join(path, file_name)
                        if os.path.isfile(file_path) and file_name.lower().endswith(supported_extensions):
                            selected_files.append(file_path)
                            output_folders[file_path] = os.path.join(path, "mattes")  # Output folder for this file
                except PermissionError:
                    logging.warning(f"Permission denied: {path}")
                    continue
            elif os.path.isfile(path):  # If it's a file, add it directly
                selected_files.append(path)
                output_folders[path] = os.path.join(os.path.dirname(path), "mattes")  # Output folder for this file
            else:
                logging.warning(f"Skipping invalid path: {path}")

        if not selected_files:
            QMessageBox.warning(self, "Warning", "No valid files selected for batch processing.")
            return

        # Create output folders
        for folder in set(output_folders.values()):
            os.makedirs(folder, exist_ok=True)

        filename_prefix = self.filename_prefix_edit2.text()
        output_format = self.output_format_combo2.currentText()

        # Create a worker thread for batch processing
        self.worker = BatchProcessor(selected_files, output_folders, filename_prefix, output_format, self.model, self.device)
        self.worker.progress_updated.connect(self.update_batch_progress)  # Connect progress signal
        self.worker.processing_finished.connect(self.on_batch_processing_finished)  # Connect completion signal
        self.worker.start()

    def update_batch_progress(self, progress, time_remaining):
        """Update batch progress bar and estimated time remaining label."""
        self.batch_progress_bar.setValue(progress)
        if time_remaining > 0:
            minutes, seconds = divmod(int(time_remaining), 60)
            self.batch_progress_label.setText(f"Estimated time remaining: {minutes:02d}:{seconds:02d}")
        else:
            self.batch_progress_label.setText("Estimated time remaining: N/A")

    def on_batch_processing_finished(self):
        """Handle the completion of batch processing."""
        self.batch_progress_label.setText("Batch processing completed.")
        QMessageBox.information(self, "Info", "Batch processing completed successfully.")

    def on_processing_finished(self):
        """Handle the completion of video processing."""
        output_folder = os.path.join(os.path.dirname(self.input_path), "mattes")
        self.drop_label.setText(f"Video mattes saved to: {output_folder}")
        self.generate_button.setEnabled(True)
        self.stop_button.setEnabled(False)  # Disable stop button after processing
        self.play_pause_button.setEnabled(True)
        self.frame_slider.setEnabled(True)  # Enable the frame slider
        self.view_output_button.setEnabled(True)  # Enable view output button

    def on_processing_error(self, message):
        """Handle errors reported by the video processing thread."""
        QMessageBox.critical(self, "Error", message)  # Show an error message box
        self.drop_label.setText("An error occurred during video processing. See message box.")
        self.generate_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.play_pause_button.setEnabled(True)
        self.frame_slider.setEnabled(True)  # Enable the frame slider
        self.view_output_button.setEnabled(True)  # Enable view output button

    def wheelEvent(self, event):
        """Zoom in/out on mouse wheel."""
        if event.angleDelta().y() > 0:
            factor = 1.1
        else:
            factor = 0.9
        self.zoom_factor *= factor
        self.graphics_view.scale(factor, factor)

    def apply_erosion_effect(self):
        """Apply erosion to the current output matte."""
        if self.output_image is None:
            QMessageBox.warning(self, "Warning", "No matte available to apply erosion.")
            return
        try:
            matte_np = np.array(self.output_image)
            kernel = np.ones((5, 5), np.uint8)
            eroded = cv2.erode(matte_np, kernel, iterations=1)
            self.output_image = Image.fromarray(eroded)
            self.update_preview()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Erosion failed: {e}")

    def apply_dilation_effect(self):
        """Apply dilation to the current output matte."""
        if self.output_image is None:
            QMessageBox.warning(self, "Warning", "No matte available to apply dilation.")
            return
        try:
            matte_np = np.array(self.output_image)
            kernel = np.ones((5, 5), np.uint8)
            dilated = cv2.dilate(matte_np, kernel, iterations=1)
            self.output_image = Image.fromarray(dilated)
            self.update_preview()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Dilation failed: {e}")

    def apply_feather_effect(self):
        """Apply feather (blur) to the current output matte."""
        if self.output_image is None:
            QMessageBox.warning(self, "Warning", "No matte available to apply feather.")
            return
        try:
            matte_np = np.array(self.output_image)
            # Use Gaussian blur for feathering
            feathered = cv2.GaussianBlur(matte_np, (11, 11), sigmaX=5)
            self.output_image = Image.fromarray(feathered)
            self.update_preview()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Feather failed: {e}")

    def apply_smoothing_effect(self):
        """Apply smoothing (median blur) to the current output matte."""
        if self.output_image is None:
            QMessageBox.warning(self, "Warning", "No matte available to apply smoothing.")
            return
        try:
            matte_np = np.array(self.output_image)
            # Use median blur for smoothing
            smoothed = cv2.medianBlur(matte_np, 7)
            self.output_image = Image.fromarray(smoothed)
            self.update_preview()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Smoothing failed: {e}")

    def save_current_matte(self):
        """Save the current output matte to disk."""
        if self.output_image is None:
            QMessageBox.warning(self, "Warning", "No matte available to save.")
            return
        try:
            # Suggest a default path in the same folder as input, or user's home if not available
            if self.input_path:
                default_dir = os.path.dirname(self.input_path)
                base_filename = os.path.splitext(os.path.basename(self.input_path))[0]
            else:
                default_dir = os.path.expanduser("~")
                base_filename = "matte"
            output_format = self.output_format_combo.currentText() if hasattr(self, "output_format_combo") else "PNG"
            filename_prefix = self.filename_prefix_edit.text() if hasattr(self, "filename_prefix_edit") else ""
            filename_prefix = filename_prefix + "_" if filename_prefix else ""
            default_filename = f"{filename_prefix}{base_filename}_effect.{output_format.lower()}"
            file_path, _ = QFileDialog.getSaveFileName(
                self,
                "Save Matte",
                os.path.join(default_dir, default_filename),
                f"{output_format} Files (*.{output_format.lower()});;All Files (*)"
            )
            if file_path:
                self.output_image.save(file_path, format=output_format)
                QMessageBox.information(self, "Saved", f"Matte saved to: {file_path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save matte: {e}")

    def start_object_selection(self):
        """Enable bounding box selection mode for SAM on the preview image."""
        if not self.sam_loaded:
            QMessageBox.warning(self, "Warning", "SAM model not loaded.")
            return
        if self.input_image is None:
            QMessageBox.warning(self, "Warning", "No image loaded.")
            return
        self.sam_points = []
        self.sam_point_mode = False
        self.sam_box_mode = True
        self.box_start = None
        self.box_rect_item = None
        self.graphics_view.setCursor(Qt.CrossCursor)
        QMessageBox.information(self, "SAM Selection", "Drag to draw a bounding box around the object/person.")

    def mousePressEvent(self, event):
        """Capture mouse events for bounding box selection."""
        if getattr(self, "sam_box_mode", False) and event.button() == Qt.LeftButton:
            pos = self.graphics_view.mapToScene(event.pos())
            self.box_start = pos
            if self.box_rect_item:
                self.graphics_scene.removeItem(self.box_rect_item)
            self.box_rect_item = None
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        """Draw bounding box as user drags."""
        if getattr(self, "sam_box_mode", False) and self.box_start is not None:
            pos = self.graphics_view.mapToScene(event.pos())
            rect = QRectF(self.box_start, pos).normalized()
            if self.box_rect_item:
                self.graphics_scene.removeItem(self.box_rect_item)
            pen = QPen(Qt.green, 2, Qt.DashLine)
            self.box_rect_item = self.graphics_scene.addRect(rect, pen)
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        """Finish bounding box selection and run SAM."""
        if getattr(self, "sam_box_mode", False) and event.button() == Qt.LeftButton and self.box_start is not None:
            pos = self.graphics_view.mapToScene(event.pos())
            rect = QRectF(self.box_start, pos).normalized()
            x0, y0, x1, y1 = int(rect.left()), int(rect.top()), int(rect.right()), int(rect.bottom())
            self.sam_box_mode = False
            self.graphics_view.setCursor(Qt.ArrowCursor)
            if self.box_rect_item:
                self.graphics_scene.removeItem(self.box_rect_item)
                self.box_rect_item = None
            self.run_sam_with_box([x0, y0, x1, y1])
            self.box_start = None
        else:
            super().mouseReleaseEvent(event)

    def run_sam_with_box(self, box):
        """Run SAM with the selected bounding box and use the mask for BiRefNet matte generation."""
        image_np = np.array(self.input_image)
        self.sam_predictor.set_image(image_np)
        input_box = np.array(box)
        masks, scores, logits = self.sam_predictor.predict(
            box=input_box[None, :],
            multimask_output=False
        )
        mask = masks[0].astype(np.uint8) * 255
        self.on_object_selected(mask, box)

    def on_object_selected(self, mask, box=None):
        """Handle the completion of object selection and trigger BiRefNet matte for the box region."""
        self.user_mask = mask
        # Clear previous overlays
        for item in self.graphics_scene.items():
            if item is not self.image_item:
                self.graphics_scene.removeItem(item)
        # Show the mask overlay
        if mask is not None:
            overlay = QPixmap.fromImage(QImage(mask.data, mask.shape[1], mask.shape[0], QImage.Format_Grayscale8))
            overlay_item = QGraphicsPixmapItem(overlay)
            overlay_item.setOpacity(0.4)
            self.graphics_scene.addItem(overlay_item)
        # If a box is provided, crop the image and mask, run BiRefNet only on the box region, then composite result
        if box is not None:
            x0, y0, x1, y1 = box
            img_crop = self.input_image.crop((x0, y0, x1, y1))
            mask_crop = mask[y0:y1, x0:x1]
            # Run BiRefNet only on the cropped region
            matte_crop = generate_matte_from_image(img_crop, self.model, self.device)
            matte_crop_img = Image.fromarray(matte_crop.astype(np.uint8))
            # Paste the matte_crop back into a full-size matte
            full_matte = Image.new("L", self.input_image.size, 0)
            full_matte.paste(matte_crop_img, (x0, y0))
            self.output_image = full_matte
            self.update_preview()
            QMessageBox.information(self, "Info", "Matte generated for selected region. You can now save or process further.")
        else:
            QMessageBox.information(self, "Info", "Object mask generated with SAM. You can now generate the matte.")

    def integrate_matanyone(self):
        """
        Integrate MatAnyone's background removal pipeline.
        1. Save video frames to temp folder.
        2. Save first-frame mask as PNG.
        3. Call MatAnyone inference (subprocess or API).
        4. Load alpha matte sequence for preview/export.
        """
        # Step 1: Save video frames
        temp_dir = tempfile.mkdtemp()
        video_frames_dir = os.path.join(temp_dir, "frames")
        os.makedirs(video_frames_dir, exist_ok=True)
        self.save_video_frames(self.input_path, video_frames_dir)

        # Step 2: Save first-frame mask
        first_frame_mask_path = os.path.join(temp_dir, "first_frame_mask.png")
        self.save_first_frame_mask(first_frame_mask_path)

        # Step 3: Call MatAnyone inference
        matanyone_output_dir = os.path.join(temp_dir, "matte_output")
        os.makedirs(matanyone_output_dir, exist_ok=True)
        self.run_matanyone_inference(
            video_frames_dir,
            first_frame_mask_path,
            matanyone_output_dir
        )

        # Step 4: Load alpha matte sequence for preview/export
        self.load_matanyone_mattes(matanyone_output_dir)

        # Clean up temp_dir if needed
        # shutil.rmtree(temp_dir)

    def save_video_frames(self, video_path, output_dir):
        """Extract frames from video and save as images."""
        # ...implement frame extraction using cv2.VideoCapture...
        pass

    def save_first_frame_mask(self, mask_path):
        """Save the user-drawn mask for the first frame as a PNG."""
        # ...implement mask saving logic...
        pass

    def run_matanyone_inference(self, frames_dir, mask_path, output_dir):
        """Call MatAnyone's inference script as a subprocess or API."""
        # Example subprocess call (adjust path and args as needed)
        # subprocess.run([
        #     "python", "path/to/matanyone/infer.py",
        #     "--frames", frames_dir,
        #     "--mask", mask_path,
        #     "--output", output_dir
        # ])
        pass

    def load_matanyone_mattes(self, matte_dir):
        """Load the generated alpha mattes for preview/export."""
        # ...implement loading of PNG mattes for UI preview...
        pass
