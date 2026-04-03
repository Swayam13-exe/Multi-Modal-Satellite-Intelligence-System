import cv2
import numpy as np
from PIL import Image

def generate_vegetation_heatmap(image_size, intensity_score):
    """
    Generates a uniform gradient heatmap based on the vegetation health score.
    """
    import matplotlib.pyplot as plt
    heatmap = np.full((image_size[1], image_size[0]), intensity_score, dtype=np.float32)
    cmap = plt.get_cmap('RdYlGn')
    heatmap_colored = cmap(heatmap)
    heatmap_img = (heatmap_colored[:, :, :3] * 255).astype(np.uint8)
    return heatmap_img

def overlay_heatmap(image: Image.Image, heatmap_np: np.ndarray, alpha=0.5):
    """
    Blends an RGB heatmap over a PIL image.
    If the heatmap is 2D (like GradCAM output), it applies a JET color map first.
    """
    img_np = np.array(image.convert("RGB"))
    
    if len(heatmap_np.shape) == 2:
        # Grayscale -> Jet Colormap
        heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_np), cv2.COLORMAP_JET)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    else:
        heatmap_colored = heatmap_np

    heatmap_resized = cv2.resize(heatmap_colored, (img_np.shape[1], img_np.shape[0]))
    
    # Blending
    overlay = cv2.addWeighted(img_np, 1 - alpha, heatmap_resized, alpha, 0)
    return Image.fromarray(overlay)
