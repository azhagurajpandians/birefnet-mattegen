# BiRefNet Matte Generator

This project provides a GUI tool for generating alpha mattes from images and videos using the BiRefNet model. It supports single image, video, and batch processing, as well as integration with advanced segmentation models like SAM and MatAnyone.

## Features
- Load and run BiRefNet models for image/video matting
- Drag-and-drop or browse for input files
- Batch processing of folders
- Interactive object/person selection (SAM integration)
- Advanced matte effects: erosion, dilation, feather, smoothing
- Save and export mattes in PNG, JPEG, or TIFF
- Integration with MatAnyone for background removal

## Installation
1. Clone the repository:
   ```powershell
   git clone https://github.com/azhagurajpandians/birefnet-mattegen.git
   cd birefnet-mattegen
   ```
2. Install dependencies:
   ```powershell
   pip install -r requirements.txt
   ```

## Setup: BiRefNet Model Code and Weights

1. **Download BiRefNet Source Code**
   - Clone the official BiRefNet repository inside your project folder:
     ```powershell
     git clone https://github.com/ZhengPeng7/BiRefNet.git
     ```
   - This will create a `BiRefNet/` subfolder with all model code.

2. **Download Pretrained Weights**
   - Go to the [BiRefNet releases or model zoo](https://github.com/ZhengPeng7/BiRefNet) and download the desired `.pth` model weights.
   - Place the downloaded `.pth` files into the `weights/` folder in this project (create the folder if it does not exist).

3. **Verify Folder Structure**
   - Your project should look like this:
     ```
     BiRefNet/           # (cloned model code)
     weights/            # (contains .pth model files)
     ui.py               # (main GUI)
     ...
     ```

## Usage
Run the GUI application:
```powershell
python run.py
```

- Select a model from the dropdown or set a custom weights folder.
- Drag and drop an image or video, or use the Browse button.
- Use the Generate Matte button to process the input.
- Use advanced effects or batch processing as needed.

## Project Structure
- `ui.py` - Main GUI application
- `image_proc.py`, `workers.py`, `utils.py` - Supporting modules
- `models/`, `weights/` - Model code and weights

## Requirements
See `requirements.txt` for dependencies.

## License
See `LICENSE` for details.
