import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def generate_vegetation_heatmap(image_size, intensity_score):
    """
    Generates a uniform gradient heatmap based on the vegetation health score.
    Uses RdYlGn colormap: red = low vegetation, green = high vegetation.
    """
    heatmap = np.full((image_size[1], image_size[0]), intensity_score, dtype=np.float32)
    cmap = plt.get_cmap('RdYlGn')
    heatmap_colored = cmap(heatmap)
    heatmap_img = (heatmap_colored[:, :, :3] * 255).astype(np.uint8)
    return heatmap_img


def apply_jet_colormap(gray: np.ndarray) -> np.ndarray:
    """
    Applies a JET-style colormap to a 2D grayscale array [0, 1].
    Pure numpy/matplotlib replacement for cv2.applyColorMap + cv2.COLORMAP_JET.
    Returns an RGB uint8 array of the same H x W shape.
    """
    cmap = plt.get_cmap('jet')
    colored = cmap(np.clip(gray, 0.0, 1.0))          # H x W x 4 (RGBA float)
    return (colored[:, :, :3] * 255).astype(np.uint8) # H x W x 3 (RGB uint8)


def overlay_heatmap(image: Image.Image, heatmap_np: np.ndarray, alpha: float = 0.5) -> Image.Image:
    """
    Blends an RGB heatmap over a PIL image.

    Parameters
    ----------
    image      : PIL.Image  — base satellite patch (any size)
    heatmap_np : np.ndarray — either:
                   • 2D float array [0, 1]  → JET colormap applied automatically
                   • 3D uint8  H×W×3 array  → used directly as RGB overlay
    alpha      : float      — heatmap opacity (0 = invisible, 1 = full heatmap)

    Returns
    -------
    PIL.Image — blended result
    """
    img_np = np.array(image.convert("RGB"), dtype=np.float32)
    target_h, target_w = img_np.shape[:2]

    if heatmap_np.ndim == 2:
        # GradCAM / grayscale saliency — apply JET colormap
        heatmap_colored = apply_jet_colormap(heatmap_np).astype(np.float32)
    else:
        heatmap_colored = heatmap_np.astype(np.float32)

    # Resize heatmap to match image using PIL (no cv2 needed)
    heatmap_pil = Image.fromarray(heatmap_colored.astype(np.uint8)).resize(
        (target_w, target_h), resample=Image.BILINEAR
    )
    heatmap_resized = np.array(heatmap_pil, dtype=np.float32)

    # Alpha blend: out = img * (1 - alpha) + heatmap * alpha
    blended = np.clip(img_np * (1.0 - alpha) + heatmap_resized * alpha, 0, 255).astype(np.uint8)
    return Image.fromarray(blended)