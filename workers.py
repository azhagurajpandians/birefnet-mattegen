from PySide6.QtCore import QThread, Signal
import cv2
import os
import tempfile
import time
import torch
from PIL import Image
from torch.cuda.amp import autocast, GradScaler
import logging
import numpy as np
from utils import generate_matte_from_image  # Ensure this is imported


class VideoProcessor(QThread):
    progress_updated = Signal(int, float)  # Add estimated time remaining
    processing_finished = Signal()
    error_occurred = Signal(str)

    def __init__(self, input_path, output_folder, model, device, batch_size=4, filename_prefix="", output_format="PNG"):
        super().__init__()
        self.input_path = input_path
        self.output_folder = output_folder
        self.model = model
        self.device = device
        self.is_running = True
        self.batch_size = batch_size
        self.scaler = GradScaler()  # For mixed precision
        self.start_time = None
        self.frame_count = 0
        self.frame_indices = []
        self.filename_prefix = filename_prefix
        self.output_format = output_format

    def run(self):
        temp_dir = tempfile.mkdtemp()
        cap = cv2.VideoCapture(self.input_path)
        if not cap.isOpened():
            self.error_occurred.emit("Could not open video file.")
            return
        self.frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)  # Get frames per second
        self.progress_updated.emit(0, 0.0)  # Initial progress

        frame_batch = []
        self.frame_indices = []
        base_filename = os.path.splitext(os.path.basename(self.input_path))[0]

        self.start_time = time.time()  # Record start time

        try:
            for i in range(self.frame_count):
                if not self.is_running:
                    break

                ret, frame = cap.read()
                if not ret:
                    logging.warning(f"Could not read frame {i}.  Stopping processing.")
                    break

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(frame)
                frame_batch.append(image)
                self.frame_indices.append(i)

                if len(frame_batch) == self.batch_size or i == self.frame_count - 1:
                    # Process batch
                    try:
                        matte_batch = self.generate_matte_from_image_batch(frame_batch)
                        for j, matte in enumerate(matte_batch):
                            frame_index = self.frame_indices[j]  # Get frame index from the correct position
                            filename_prefix = self.filename_prefix + "_" if self.filename_prefix else ""  # add condition for the filename prefix
                            output_filename = f"Matte_{base_filename}_{frame_index + 1:04d}.{self.output_format.lower()}"  # Updated naming
                            output_path = os.path.join(self.output_folder, output_filename)
                            matte.save(output_path, format=self.output_format)
                    except Exception as e:
                        logging.error(f"Error processing batch: {e}")
                        self.error_occurred.emit(f"Error processing batch: {e}")
                        self.stop()  # Stop processing on error
                        break

                    # Update progress bar and estimated time remaining
                    elapsed_time = time.time() - self.start_time
                    frames_processed = i + 1
                    estimated_time_remaining = (self.frame_count - frames_processed) * (elapsed_time / frames_processed)
                    progress = int((frames_processed / self.frame_count) * 100)
                    self.progress_updated.emit(progress, estimated_time_remaining)

                    frame_batch = []
                    self.frame_indices = []  # Reset frame indices

        except Exception as e:
            logging.exception("An unexpected error occurred during video processing.")  # Log full traceback
            self.error_occurred.emit(f"An unexpected error occurred: {e}")

        finally:  # Ensure resources are released even if errors occur
            cap.release()
            self.progress_updated.emit(100, 0.0)
            self.processing_finished.emit()

    def stop(self):
        self.is_running = False

    def generate_matte_from_image_batch(self, image_batch):
        """Generate mattes from a batch of images using BiRefNet."""
        original_sizes = [image.size for image in image_batch]
        resized_images = [image.resize((1024, 1024)) for image in image_batch]
        image_tensors = [torch.tensor(np.array(img) / 255.0, dtype=torch.float32).permute(2, 0, 1) for img in resized_images]
        image_tensor = torch.stack(image_tensors).to(self.device)

        with torch.no_grad():
            with autocast():
                output = self.model(image_tensor)

                if isinstance(output, list):
                    output = output[0]  # Assume the first element is the tensor
                elif isinstance(output, dict):
                    output = output["logits"]  # Assume the tensor is under the "logits" key

                if not isinstance(output, torch.Tensor):
                    raise TypeError(f"Expected output to be a tensor, but got {type(output)}")

                matte_batch = torch.sigmoid(output).cpu().numpy()

        matte_images = []
        for i, matte in enumerate(matte_batch):
            # Ensure that matte is a 2D array before type conversion
            if matte.ndim == 3:  # If it has a color channel, take the average
                matte = np.mean(matte, axis=0)
                logging.warning("Matte had unexpected dimensions. Taking first channel")
            # Convert to uint8
            matte = (matte * 255).astype(np.uint8)
            # check the shape of the matte
            if matte.shape != original_sizes[i][::-1]:
                logging.warning(f"Resizing from {matte.shape} to {original_sizes[i][::-1]}")

            try:
                matte = Image.fromarray(matte)
                matte = matte.resize(original_sizes[i], Image.Resampling.LANCZOS)

            except Exception as e:
                logging.error(f"Error creating image from array: {e}")
                raise

            matte_images.append(matte)

        return matte_images


class BatchProcessor(QThread):
    progress_updated = Signal(int, float)  # Signal for progress and estimated time remaining
    processing_finished = Signal()  # Signal for completion

    def __init__(self, file_paths, output_folders, filename_prefix, output_format, model, device):
        super().__init__()
        self.file_paths = file_paths
        self.output_folders = output_folders  # Dictionary mapping file paths to output folders
        self.filename_prefix = filename_prefix
        self.output_format = output_format
        self.model = model
        self.device = device
        self.is_running = True

    def run(self):
        start_time = time.time()
        total_files = len(self.file_paths)

        for index, file_path in enumerate(self.file_paths):
            if not self.is_running:
                break

            try:
                input_image = Image.open(file_path).convert("RGB")
                matte = generate_matte_from_image(input_image, self.model, self.device)
                matte = Image.fromarray(matte.astype(np.uint8))

                base_filename = os.path.splitext(os.path.basename(file_path))[0]
                filename_prefix = self.filename_prefix + "_" if self.filename_prefix else ""
                output_filename = f"{filename_prefix}{base_filename}.{self.output_format.lower()}"
                output_folder = self.output_folders[file_path]
                output_path = os.path.join(output_folder, output_filename)

                matte.save(output_path, format=self.output_format)
                logging.info(f"Processed and saved: {output_path}")

                # Emit progress
                elapsed_time = time.time() - start_time
                progress = int(((index + 1) / total_files) * 100)
                estimated_time_remaining = (total_files - (index + 1)) * (elapsed_time / (index + 1))
                self.progress_updated.emit(progress, estimated_time_remaining)

            except Exception as e:
                logging.error(f"Error processing file {file_path}: {e}")

        self.progress_updated.emit(100, 0.0)  # Emit 100% progress when done
        self.processing_finished.emit()  # Emit completion signal

    def stop(self):
        self.is_running = False