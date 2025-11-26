import numpy as np
import xarray as xr
import pandas as pd
import requests
import os

def load_sar_image(path):
    """Load SAR image (e.g., .tif, .png, .npy)."""
    if path.endswith(".npy"):
        return np.load(path)
    elif path.endswith(".tif") or path.endswith(".tiff"):
        import cv2
        return cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    else:
        raise ValueError("Unsupported SAR format")

def load_uv_data(path):
    """Load UV fluorescence or optical sensor data."""
    df = pd.read_csv(path)
    return df

def load_current_data(path):
    """Load ocean current data (u/v components)."""
    ds = xr.open_dataset(path)
    return ds['u'], ds['v']

def load_wind_data(path):
    """Load wind speed/direction."""
    ds = xr.open_dataset(path)
    return ds['wind_speed'], ds['wind_dir']

def load_temperature(path):
    """Sea surface temperature."""
    ds = xr.open_dataset(path)
    return ds['sst']

def download_from_url(url, save_path):
    """For automatic dataset download."""
    r = requests.get(url)
    with open(save_path, "wb") as f:
        f.write(r.content)
    return save_path

