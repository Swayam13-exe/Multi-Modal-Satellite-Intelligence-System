import numpy as np

def normalize_coordinates(lat, lon):
    """
    Normalize latitude [-90, 90] to [0, 1]
    Normalize longitude [-180, 180] to [0, 1]
    """
    lat_norm = (lat + 90.0) / 180.0
    lon_norm = (lon + 180.0) / 360.0
    return lat_norm, lon_norm

def encode_month(month):
    """
    Encode month [1-12] into continuous cyclic representation using sin and cos.
    """
    month_rad = 2.0 * np.pi * (month - 1) / 12.0
    return np.sin(month_rad), np.cos(month_rad)

def approximate_ndvi(image_tensor):
    """
    Approximates NDVI from RGB image tensor using (G - R) / (G + R)
    Expects image_tensor of shape (C, H, W) normalized to [0, 1].
    """
    if image_tensor is None or image_tensor.shape[0] < 3:
        return 0.0
    r_mean = image_tensor[0].mean()
    g_mean = image_tensor[1].mean()
    # Compute proxy NDVI
    ndvi = (g_mean - r_mean) / (g_mean + r_mean + 1e-6)
    return np.clip(ndvi.item(), -1.0, 1.0)

def extract_meta_features(lat, lon, month, image_tensor=None):
    """
    Combines coordinate normalization and month encoding.
    Returns: numpy array of 5 features [lat_norm, lon_norm, sin_mo, cos_mo, ndvi]
    """
    lat_norm, lon_norm = normalize_coordinates(lat, lon)
    sin_mo, cos_mo = encode_month(month)
    ndvi = approximate_ndvi(image_tensor)
    return np.array([lat_norm, lon_norm, sin_mo, cos_mo, ndvi], dtype=np.float32)

def derive_vegetation_proxy(image_tensor):
    """
    Optional: derive vegetation proxy from the green channel.
    Expects image_tensor of shape (C, H, W) normalized to [0, 1] or similar.
    We compute the average intensity of the green channel relative to others.
    """
    if image_tensor.shape[0] >= 3:
        # Assuming RGB format: index 0 is R, 1 is G, 2 is B
        r_mean = image_tensor[0].mean()
        g_mean = image_tensor[1].mean()
        b_mean = image_tensor[2].mean()
        
        # Simple Greenness Ratio
        veg_score = g_mean / (r_mean + g_mean + b_mean + 1e-6)
        return veg_score.item()
    return 0.0
