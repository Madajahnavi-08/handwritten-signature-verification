"""
Signature Image Preprocessing Pipeline for Siamese Neural Network.
Author: Mada Jahnavi
Domain: Deep Learning & Computer Vision
"""

import cv2
import numpy as np
from PIL import Image
import torch
from torchvision import transforms


def preprocess_signature_image(
    image_input,
    target_size: tuple = (100, 100),
    invert_colors: bool = True
) -> np.ndarray:
    """
    Structured signature preprocessing pipeline:
      1. Format normalization (Loads PIL / OpenCV array / File path)
      2. Grayscale conversion
      3. Background noise reduction using Gaussian Blur
      4. Otsu's adaptive binarization to separate ink from background paper
      5. Morphological smoothing to clean micro-artifacts
      6. Tight bounding box crop around signature strokes
      7. Aspect-ratio preserving resize and centered canvas padding
      8. Normalization to [0, 1] range float32 array
    """
    # Step 1: Input handling
    if isinstance(image_input, str):
        img = cv2.imread(image_input)
        if img is None:
            raise FileNotFoundError(f"Could not load image at path: {image_input}")
    elif isinstance(image_input, Image.Image):
        img = cv2.cvtColor(np.array(image_input), cv2.COLOR_RGB2BGR)
    elif isinstance(image_input, np.ndarray):
        img = image_input.copy()
    else:
        raise TypeError("image_input must be a filepath (str), PIL.Image, or numpy.ndarray")

    # Step 2: Convert to Grayscale
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    # Step 3: Noise filtering
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    # Step 4: Otsu Binarization
    # Paper is typically bright (200-255) and ink is dark (0-100)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Step 5: Morphological closing to seal pen stroke gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # Step 6: Bounding Box Detection
    coords = cv2.findNonZero(cleaned)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        # Add 5px padding if possible
        pad = 6
        h_orig, w_orig = cleaned.shape
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(w_orig, x + w + pad)
        y2 = min(h_orig, y + h + pad)
        cropped = cleaned[y1:y2, x1:x2]
    else:
        cropped = cleaned

    # Step 7: Aspect-ratio preserving resize to target_size (e.g. 100x100)
    target_w, target_h = target_size
    h_c, w_c = cropped.shape
    
    if h_c == 0 or w_c == 0:
        cropped = np.zeros((target_h, target_w), dtype=np.uint8)
        h_c, w_c = target_h, target_w

    scale = min(target_w / w_c, target_h / h_c)
    new_w = max(1, int(w_c * scale))
    new_h = max(1, int(h_c * scale))
    
    resized = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Create empty canvas with 0 (black background for inverted strokes)
    canvas = np.zeros((target_h, target_w), dtype=np.uint8)
    
    x_offset = (target_w - new_w) // 2
    y_offset = (target_h - new_h) // 2
    canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized

    # If inverted representation is not desired, invert back
    if not invert_colors:
        canvas = 255 - canvas

    # Step 8: Normalize to [0.0, 1.0] float32
    normalized = canvas.astype(np.float32) / 255.0
    return normalized


def preprocess_to_tensor(image_input, device=None) -> torch.Tensor:
    """
    Convenience function that converts image to PyTorch tensor ready for SiameseNet:
    Shape: (1, 1, 100, 100)
    """
    preprocessed_np = preprocess_signature_image(image_input)
    tensor = torch.from_numpy(preprocessed_np).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    if device is not None:
        tensor = tensor.to(device)
    return tensor
