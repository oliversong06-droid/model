import numpy as np
from scipy.ndimage import gaussian_filter

def calculate_ssim(img1, img2, data_range=1.0, win_size=11, sigma=1.5):
    """
    Calculate Structural Similarity Index (SSIM) between two images.
    
    Parameters:
    -----------
    img1, img2 : np.ndarray
        Input images (2D).
    data_range : float
        Dynamic range of the images (default 1.0 for 0-1 normalized images).
    win_size : int
        Window size for Gaussian filter (must be odd).
    sigma : float
        Standard deviation for Gaussian kernel.
        
    Returns:
    --------
    ssim_val : float
        Mean SSIM value.
    """
    if img1.shape != img2.shape:
        raise ValueError("Input images must have the same dimensions.")
    
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    
    # Constants
    K1 = 0.01
    K2 = 0.03
    C1 = (K1 * data_range) ** 2
    C2 = (K2 * data_range) ** 2
    
    # Mean calculation using Gaussian filter (equivalent to convolution)
    mu1 = gaussian_filter(img1, sigma)
    mu2 = gaussian_filter(img2, sigma)
    
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    
    # Variance and Covariance
    sigma1_sq = gaussian_filter(img1 ** 2, sigma) - mu1_sq
    sigma2_sq = gaussian_filter(img2 ** 2, sigma) - mu2_sq
    sigma12 = gaussian_filter(img1 * img2, sigma) - mu1_mu2
    
    # SSIM formula
    numerator = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
    denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    
    ssim_map = numerator / denominator
    return np.mean(ssim_map)

def calculate_metrics(y_true, y_pred):
    """
    Calculate evaluation metrics for image prediction.
    
    Returns:
    --------
    dict containing:
        - MSE (Mean Squared Error)
        - MAE (Mean Absolute Error)
        - RMSE (Root Mean Squared Error)
        - SSIM (Structural Similarity Index)
        - PSNR (Peak Signal-to-Noise Ratio)
    """
    # Ensure inputs are numpy arrays
    y_true = np.array(y_true, dtype=np.float64)
    y_pred = np.array(y_pred, dtype=np.float64)
    
    # MSE
    mse = np.mean((y_true - y_pred) ** 2)
    
    # MAE
    mae = np.mean(np.abs(y_true - y_pred))
    
    # RMSE
    rmse = np.sqrt(mse)
    
    # PSNR
    # Avoid division by zero
    if mse == 0:
        psnr = 100.0
    else:
        max_pixel = 1.0 # Assuming normalized data [0, 1]
        psnr = 20 * np.log10(max_pixel / rmse)
        
    # SSIM
    # We assume 2D images (H, W). If 3D (C, H, W) or (N, ...), handle accordingly.
    # For this project, we usually compare single frames (H, W).
    if y_true.ndim == 2:
        ssim_val = calculate_ssim(y_true, y_pred, data_range=1.0)
    elif y_true.ndim == 3: # (C, H, W) or (T, H, W) -> Average over channels/time
        ssims = []
        for i in range(y_true.shape[0]):
            ssims.append(calculate_ssim(y_true[i], y_pred[i], data_range=1.0))
        ssim_val = np.mean(ssims)
    else:
        ssim_val = 0.0 # Not supported for >3 dims yet
        
    return {
        "MSE": mse,
        "MAE": mae,
        "RMSE": rmse,
        "PSNR": psnr,
        "SSIM": ssim_val
    }
