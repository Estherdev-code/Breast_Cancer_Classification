"""
preprocess.py
-------------
Stage 1: Image Intelligence Layer
CLAHE preprocessing pipeline matching the trained ResNet50 model.
"""

import cv2
import numpy as np
from tensorflow.keras.applications.resnet50 import preprocess_input


def load_image(image_path):
    """Load a mammogram image in grayscale."""
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not load image at: {image_path}")
    return image


def denoise(image):
    """Apply Gaussian blur to suppress random noise."""
    return cv2.GaussianBlur(image, (3, 3), 0)


def apply_clahe(image):
    """Contrast Limited Adaptive Histogram Equalization."""
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    return clahe.apply(image)


def resize(image, target_size=(224, 224)):
    """Resize to ResNet50 input size."""
    return cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)


def convert_to_rgb(image):
    """Convert grayscale to 3-channel RGB."""
    return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)


def preprocess(image_path):
    """Full preprocessing pipeline for a file path."""
    image = load_image(image_path)
    image = denoise(image)
    image = apply_clahe(image)
    image = resize(image)
    image = convert_to_rgb(image)
    return image


def preprocess_from_array(image_array):
    """Same pipeline but for an already-loaded numpy array."""
    image = denoise(image_array)
    image = apply_clahe(image)
    image = resize(image)
    image = convert_to_rgb(image)
    return image


def prepare_for_model(image):
    """Apply ResNet50-specific normalization."""
    image = image.astype("float32")
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)
    return image