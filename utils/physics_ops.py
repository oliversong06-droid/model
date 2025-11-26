import numpy as np
from scipy.ndimage import convolve

def laplacian(field):
    """Compute ∇² using convolution."""
    kernel = np.array([[0, 1, 0],
                       [1,-4, 1],
                       [0, 1, 0]])
    return convolve(field, kernel, mode='reflect')

def apply_diffusion(oil, D, dt):
    """Oil diffusion term."""
    return oil + D * laplacian(oil) * dt

def apply_wind_advection(oil, wind_u, wind_v, alpha, dt):
    """Wind-driven surface transport."""
    shifted = np.roll(oil, shift=(int(wind_v*alpha*dt), int(wind_u*alpha*dt)), axis=(0,1))
    return shifted

def apply_current_advection(oil, u, v, beta, dt):
    """Ocean current transport."""
    shifted = np.roll(oil, shift=(int(v*beta*dt), int(u*beta*dt)), axis=(0,1))
    return shifted

