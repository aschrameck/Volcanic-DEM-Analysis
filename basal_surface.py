"""
Basal surface reconstruction following Hunt et al. (2020)
--------------------------------------------------------

Reconstructs the pre-eruptive basal surface beneath a volcanic cone
using elevations sampled from a ring surrounding the cone footprint.

Key properties:
- Uses only exterior terrain (no cone/crater interior)
- Rasterized sampling (fast, vectorized)
- Robust to breached cones and elongate shapes
- Safe handling of masked arrays and nodata
"""

import numpy as np
from rasterio.features import rasterize


# Custom exceptions
class BasalSurfaceError(Exception):
    """Raised when basal surface reconstruction is ill-posed."""
    def __init__(self, message="Basal surface reconstruction error"):
        self.message = message
        super().__init__(self.message)


# --- Utility functions ---
def sanitize_dem(dem):
    """
    Ensure DEM is a plain float ndarray with NaN nodata.
    """
    if np.ma.isMaskedArray(dem):
        dem = dem.filled(np.nan)

    dem = np.asarray(dem, dtype=float)
    dem[~np.isfinite(dem)] = np.nan

    return dem


def pixel_grid(dem, transform):
    """
    Return map-coordinate grids X, Y for a DEM.
    """
    rows, cols = dem.shape
    xs, ys = np.meshgrid(np.arange(cols), np.arange(rows))

    X = transform.c + xs * transform.a
    Y = transform.f + ys * transform.e

    return X, Y


# --- Basal ring sampling ---
def sample_basal_ring(
    dem,
    transform,
    cone_poly,
    inner_buffer=0.0,
    outer_buffer=None,
    min_points=50,
):
    """
    Sample elevations from a ring surrounding the cone footprint.
    """

    # Sanitize DEM (critical)
    dem = sanitize_dem(dem)

    # Estimate outer buffer if not provided
    if outer_buffer is None:
        equiv_radius = np.sqrt(cone_poly.area / np.pi)
        outer_buffer = max(0.2 * equiv_radius, 50.0)

    # Build basal ring geometry
    inner = cone_poly.buffer(inner_buffer)
    outer = cone_poly.buffer(outer_buffer)
    ring = outer.difference(inner)

    # Rasterize ring mask
    mask = rasterize(
        [(ring, 1)],
        out_shape=dem.shape,
        transform=transform,
        fill=0,
        dtype="uint8",
    ).astype(bool)

    if not np.any(mask):
        raise BasalSurfaceError("Basal ring rasterized to empty mask")

    # Extract coordinates and elevations
    X, Y = pixel_grid(dem, transform)
    valid = mask & ~np.isnan(dem)

    z = dem[valid]
    x = X[valid]
    y = Y[valid]

    if z.size < min_points:
        raise BasalSurfaceError(
            f"Insufficient basal control points ({z.size} found)"
        )

    return x, y, z


# --- Surface fitting ---
def fit_basal_surface(x, y, z, order=1):
    """
    Fit a low-order surface to basal control points.
    """
    if order == 1:
        A = np.column_stack([x, y, np.ones_like(x)])
    elif order == 2:
        A = np.column_stack([
            x, y, x * y, x**2, y**2, np.ones_like(x)
        ])
    else:
        raise ValueError("Only order=1 or order=2 supported")

    coeffs, *_ = np.linalg.lstsq(A, z, rcond=None)
    return coeffs


def evaluate_surface(coeffs, X, Y, order=1):
    """
    Evaluate fitted basal surface over a grid.
    """
    if order == 1:
        a, b, c = coeffs
        return a * X + b * Y + c
    else:
        a, b, c, d, e, f = coeffs
        return (
            a * X +
            b * Y +
            c * X * Y +
            d * X**2 +
            e * Y**2 +
            f
        )


# --- Main function ---
def basal_surface_from_dem(
    dem,
    transform,
    cone_poly,
    order=1,
    outer_buffer=None,
):
    """
    Reconstruct basal surface beneath a volcanic cone.

    Returns
    -------
    basal : 2D ndarray
        Reconstructed basal surface
    """

    # Final DEM sanitation (belt-and-suspenders)
    dem = sanitize_dem(dem)

    # Sample basal control points
    x, y, z = sample_basal_ring(
        dem=dem,
        transform=transform,
        cone_poly=cone_poly,
        outer_buffer=outer_buffer,
    )

    # Fit surface
    coeffs = fit_basal_surface(x, y, z, order=order)

    # Evaluate over full grid
    X, Y = pixel_grid(dem, transform)
    basal = evaluate_surface(coeffs, X, Y, order=order)

    return basal
