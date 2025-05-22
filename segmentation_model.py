import torch
import numpy as np
from torchvision import models, transforms
from PIL import Image

class Segmenter:
    def __init__(self):
        """Initialize the segmentation model."""
        self.model = models.segmentation.deeplabv3_resnet101(pretrained=True).eval()  # Use DeepLabV3
        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def segment(self, image):
        """Perform segmentation on the input image."""
        input_tensor = self.transform(image).unsqueeze(0)  # Add batch dimension
        with torch.no_grad():
            output = self.model(input_tensor)['out'][0]  # Get the segmentation output
        mask = output.argmax(0).byte().cpu().numpy()  # Convert to binary mask
        return mask
