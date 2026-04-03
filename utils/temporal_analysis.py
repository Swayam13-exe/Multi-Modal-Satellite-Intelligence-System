import numpy as np
from PIL import Image, ImageChops

def compare_images(img1: Image.Image, img2: Image.Image):
    """
    Returns an absolute difference pixel map.
    """
    # Resize image 2 to match image 1 if needed
    img2_resized = img2.convert("RGB").resize(img1.size)
    img1_rgb = img1.convert("RGB")
    diff = ImageChops.difference(img1_rgb, img2_resized)
    return diff

def compute_change_mask(diff_image: Image.Image, threshold=20):
    """
    Computes a binary highlighted change mask based on RGB absolute difference.
    Useful for urban growth or vegetation loss detection.
    """
    diff_np = np.array(diff_image)
    # Convert to grayscale
    gray = np.mean(diff_np, axis=2)
    
    # Apply Threshold
    mask = gray > threshold
    
    # Create an overlay (red for changes)
    highlight = np.zeros_like(diff_np)
    highlight[mask] = [255, 0, 0] # Red
    
    return Image.fromarray(highlight), np.mean(mask) * 100 # return % change

def compute_ndvi_difference(ndvi_1: float, ndvi_2: float):
    return ndvi_2 - ndvi_1
