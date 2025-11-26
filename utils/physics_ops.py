# utils/physics_ops.py
"""
Physics utilities for synthetic oil spill generation.

This module provides:
- Initial oil slick generator
- Time-varying current fields
- Simple diffusion + advection step

All parameters are deliberately simple and can be later
replaced by literature-based values.
"""

from __future__ import annotations
import numpy as np
from scipy.ndimage import gaussian_filter


def generate_initial_oil(
    H: int,
    W: int,
    max_radius: float = 5.0,
    min_radius: float = 2.0,
) -> np.ndarray:
    """
    Generate an initial oil patch as a 2D Gaussian blob.

    Returns:
        oil: (H, W) float32, values in [0, ~1]
    """
    y = np.linspace(0, H - 1, H, dtype=np.float32)
    x = np.linspace(0, W - 1, W, dtype=np.float32)
    X, Y = np.meshgrid(x, y)

    # Center is chosen somewhere in the middle area
    cx = np.random.uniform(W * 0.25, W * 0.75)
    cy = np.random.uniform(H * 0.25, H * 0.75)
    radius = np.random.uniform(min_radius, max_radius)

    blob = np.exp(-(((X - cx) ** 2 + (Y - cy) ** 2) / (2.0 * radius**2)))
    blob /= (blob.max() + 1e-8)

    # Scale to arbitrary "thickness"
    blob *= np.random.uniform(0.4, 1.0)

    return blob.astype(np.float32)


def generate_current_field(
    T: int,
    H: int,
    W: int,
    base_speed_min: float = 0.01,
    base_speed_max: float = 0.05,
    noise_level: float = 0.01,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate time-varying current fields (U, V) with mild randomness.

    Args:
        T: number of time steps
        H, W: grid size
        base_speed_min, base_speed_max: range of mean current speed
        noise_level: spatial noise added each step

    Returns:
        U: (T, H, W) float32
        V: (T, H, W) float32
    """
    theta0 = np.random.uniform(0, 2 * np.pi)
    base_speed = np.random.uniform(base_speed_min, base_speed_max)

    U = np.zeros((T, H, W), dtype=np.float32)
    V = np.zeros((T, H, W), dtype=np.float32)

    for t in range(T):
        # Small perturbation in direction and speed
        dtheta = np.random.normal(scale=0.03)
        dspeed = np.random.normal(scale=0.005)

        speed_t = max(base_speed + dspeed, 0.0)
        theta_t = theta0 + dtheta * t

        ux_t = speed_t * np.cos(theta_t)
        uy_t = speed_t * np.sin(theta_t)

        U[t, :, :] = ux_t + np.random.normal(scale=noise_level, size=(H, W))
        V[t, :, :] = uy_t + np.random.normal(scale=noise_level, size=(H, W))

    return U.astype(np.float32), V.astype(np.float32)


def apply_diffusion(
    oil: np.ndarray,
    D: float,
    dt: float,
    dx: float,
) -> np.ndarray:
    """
    Very simple diffusion model using Gaussian blur as a proxy.

    Args:
        oil: (H, W)
        D: diffusion coefficient (arbitrary units)
        dt: time step
        dx: grid spacing

    Returns:
        oil_diffused: (H, W)
    """
    # Very approximate relation for sigma
    sigma = np.sqrt(max(2.0 * D * dt, 0.0)) / (dx + 1e-8)

    if sigma < 1e-3:
        return oil.astype(np.float32)

    blurred = gaussian_filter(oil, sigma=float(sigma))
    return blurred.astype(np.float32)


def apply_advection(
    oil: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    dt: float,
    dx: float,
) -> np.ndarray:
    """
    Semi-Lagrangian style advection: backward tracing.

    Args:
        oil: (H, W)
        u, v: (H, W) current fields
        dt: time step
        dx: grid spacing

    Returns:
        oil_new: (H, W)
    """
    H, W = oil.shape
    y = np.arange(H, dtype=np.float32)
    x = np.arange(W, dtype=np.float32)
    X, Y = np.meshgrid(x, y)

    Xb = X - (u * dt / (dx + 1e-8))
    Yb = Y - (v * dt / (dx + 1e-8))

    Xb = np.clip(Xb, 0.0, W - 1.0)
    Yb = np.clip(Yb, 0.0, H - 1.0)

    x0 = np.floor(Xb).astype(int)
    x1 = np.clip(x0 + 1, 0, W - 1)
    y0 = np.floor(Yb).astype(int)
    y1 = np.clip(y0 + 1, 0, H - 1)

    wx = Xb - x0
    wy = Yb - y0

    Ia = oil[y0, x0]
    Ib = oil[y0, x1]
    Ic = oil[y1, x0]
    Id = oil[y1, x1]

    oil_new = (
        Ia * (1.0 - wx) * (1.0 - wy)
        + Ib * wx * (1.0 - wy)
        + Ic * (1.0 - wx) * wy
        + Id * wx * wy
    )

    return oil_new.astype(np.float32)


def step_physics(
    oil: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    D: float,
    dt: float,
    dx: float,
) -> np.ndarray:
    """
    One full physics step: diffusion + advection.

    Args:
        oil: (H, W)
        u, v: (H, W)
        D, dt, dx: physical parameters

    Returns:
        oil_next: (H, W) in [0, 1]
    """
    oil_diffused = apply_diffusion(oil, D=D, dt=dt, dx=dx)
    oil_advected = apply_advection(oil_diffused, u=u, v=v, dt=dt, dx=dx)
    oil_clamped = np.clip(oil_advected, 0.0, 1.0)
    return oil_clamped.astype(np.float32)
