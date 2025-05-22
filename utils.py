import logging
import os
import torch
from torchvision import transforms
import numpy as np
import random
import cv2
from PIL import Image
import subprocess
import OpenEXR
import Imath


def path_to_image(path, size=(1024, 1024), color_type=['rgb', 'gray'][0]):
    if color_type.lower() == 'rgb':
        image = cv2.imread(path)
    elif color_type.lower() == 'gray':
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    else:
        print('Select the color_type to return, either to RGB or gray image.')
        return
    if size:
        image = cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)
    if color_type.lower() == 'rgb':
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).convert('RGB')
    else:
        image = Image.fromarray(image).convert('L')
    return image


def check_state_dict(state_dict, unwanted_prefixes=['module.', '_orig_mod.']):
    for k in list(state_dict.keys()):
        new_key = k
        for unwanted_prefix in unwanted_prefixes:
            if new_key.startswith(unwanted_prefix):
                new_key = new_key[len(unwanted_prefix):]
        if new_key != k:
            state_dict[new_key] = state_dict.pop(k)
    return state_dict


def generate_smoothed_gt(gts):
    epsilon = 0.001
    new_gts = (1-epsilon)*gts+epsilon/2
    return new_gts


class Logger():
    def __init__(self, path="log.txt"):
        self.logger = logging.getLogger('BiRefNet')
        self.file_handler = logging.FileHandler(path, "w")
        self.stdout_handler = logging.StreamHandler()
        self.stdout_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
        self.file_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
        self.logger.addHandler(self.file_handler)
        self.logger.addHandler(self.stdout_handler)
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False
    
    def info(self, txt):
        self.logger.info(txt)
    
    def close(self):
        self.file_handler.close()
        self.stdout_handler.close()


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0.0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(state, path, filename="latest.pth"):
    torch.save(state, os.path.join(path, filename))


def save_tensor_img(tenor_im, path):
    im = tenor_im.cpu().clone()
    im = im.squeeze(0)
    tensor2pil = transforms.ToPILImage()
    im = tensor2pil(im)
    im.save(path)


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def generate_matte_from_image(image, model, device):
    """
    Generates a matte from the input image using the provided model.
    Args:
        image: Input image as a PIL Image or NumPy array.
        model: Pre-trained model for matte generation.
        device: Device to run the model on (e.g., 'cpu' or 'cuda').
    Returns:
        matte: Generated matte as a NumPy array.
    """
    model = model.to(device)  # Ensure model is on the correct device
    model.eval()
    with torch.no_grad():
        # Convert image to tensor
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        original_size = image.size
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        input_tensor = transform(image.resize((1024, 1024))).unsqueeze(0).to(device)

        # Generate matte
        output = model(input_tensor)
        if isinstance(output, list):  # Handle list output
            output = output[0]
        matte = torch.sigmoid(output).squeeze().cpu().numpy()

        # Handle multi-channel output
        if matte.ndim == 3:  # If it has multiple channels, take the average
            matte = matte.mean(axis=0)

        # Resize matte back to original size
        matte = Image.fromarray((matte * 255).clip(0, 255).astype(np.uint8))
        matte = matte.resize(original_size, Image.Resampling.LANCZOS)
    return np.array(matte)


def save_exr(image, output_path):
    """
    Saves a NumPy array as an OpenEXR file.
    Args:
        image: Input image as a NumPy array (H x W x C) with float32 values.
        output_path: Path to save the EXR file.
    """
    if image.dtype != np.float32:
        raise ValueError("Image must be of type float32.")
    if len(image.shape) != 3 or image.shape[2] not in [1, 3]:
        raise ValueError("Image must have shape (H, W, C) where C is 1 or 3.")

    header = OpenEXR.Header(image.shape[1], image.shape[0])
    if image.shape[2] == 1:  # Grayscale
        channel_data = [image[:, :, 0].tobytes()]
        header['channels'] = {'Y': Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT))}
    else:  # RGB
        channel_data = [image[:, :, i].tobytes() for i in range(3)]
        header['channels'] = {
            'R': Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT)),
            'G': Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT)),
            'B': Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT))
        }

    exr_file = OpenEXR.OutputFile(output_path, header)
    exr_file.writePixels({'R': channel_data[0], 'G': channel_data[1], 'B': channel_data[2]} if len(channel_data) == 3 else {'Y': channel_data[0]})
    exr_file.close()


def create_mp4_from_pngs(png_folder, output_path, fps):
    """
    Creates an MP4 video from a sequence of PNG images.
    Args:
        png_folder: Folder containing PNG images.
        output_path: Path to save the MP4 video.
        fps: Frames per second for the video.
    """
    png_files = sorted([os.path.join(png_folder, f) for f in os.listdir(png_folder) if f.endswith('.png')])
    if not png_files:
        raise ValueError("No PNG files found in the specified folder.")

    # Build ffmpeg command
    command = [
        'ffmpeg',
        '-y',  # Overwrite output file if it exists
        '-framerate', str(fps),
        '-i', os.path.join(png_folder, '%04d.png'),  # Assumes files are named sequentially (e.g., 0001.png, 0002.png)
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        output_path
    ]

    # Run the command
    subprocess.run(command, check=True)




