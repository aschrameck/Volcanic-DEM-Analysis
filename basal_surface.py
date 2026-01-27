"""
Basal surface reconstruction following Hunt et al. (2020)
--------------------------------------------------------

This module reconstructs the pre-eruptive (basal) surface of a volcanic cone
using elevations sampled from a local ring *outside* the cone footprint.

Conceptual basis:
- Only terrain surrounding the cone represents pre-eruption topography
- Interior cone/crater elevations are excluded
- A low-order surface (planar by default) approximates the basal surface

This approach is robust to breached cones and elongated geometries and avoids
unconstrained interpolation across large interior voids.

Intended usage:
    basal = basal_surface_from_dem(dem, transform, cone_polygon)

Dependencies: numpy, shapely
"""

import numpy as np
from shapely.geometry import Point


class BasalSurfaceError(Exception):
    """Raised when basal surface reconstruction is ill-posed."""
    def __init__(self, message="Basal surface reconstruction error"):
        self.message = message
        super().__init__(self.message)


def _pixel_grid(dem, transform):
    """Return map-coordinate grids X, Y for a DEM."""
    rows, cols = dem.shape
    xs, ys = np.meshgrid(np.arange(cols), np.arange(rows))
    X = transform.c + xs * transform.a
    Y = transform.f + ys * transform.e
    return X, Y


def sample_basal_ring(dem, transform, cone_poly, inner_buffer=0.0,
                      outer_buffer=None, min_points=50):
    """
    Sample elevations from a ring surrounding the cone footprint.

    Parameters
    ----------
    dem : 2D ndarray
        DEM elevations
    transform : affine.Affine
        Raster transform
    cone_poly : shapely Polygon
        Cone basal footprint
    inner_buffer : float, optional
        Inner buffer distance (meters). Default 0.
    outer_buffer : float, optional
        Outer buffer distance (meters). If None, estimated from cone size.
    min_points : int
        Minimum required control points

    Returns
    -------
    x, y, z : 1D ndarrays
        Sampled basal control points
    """
    # Estimate a reasonable outer buffer if not provided
    if outer_buffer is None:
        # ~10–20% of equivalent cone diameter
        equiv_radius = np.sqrt(cone_poly.area / np.pi)
        outer_buffer = max(0.2 * equiv_radius, 50.0)

    # Create basal sampling ring
    inner = cone_poly.buffer(inner_buffer)
    outer = cone_poly.buffer(outer_buffer)
    ring = outer.difference(inner)

    X, Y = _pixel_grid(dem, transform)

    mask = np.zeros(dem.shape, dtype=bool)

    # Point-in-polygon test (explicit loop for robustness)
    for i in range(dem.shape[0]):
        for j in range(dem.shape[1]):
            if not np.isfinite(dem[i, j]):
                continue
            if ring.contains(Point(X[i, j], Y[i, j])):
                mask[i, j] = True

    z = dem[mask]
    x = X[mask]
    y = Y[mask]

    # Check for sufficient points
    if len(z) < min_points:
        raise BasalSurfaceError(
            f"Insufficient basal control points ({len(z)} found)"
        )

    return x, y, z


def fit_basal_surface(x, y, z, order=1):
    """
    Fit a low-order surface to basal control points.

    Parameters
    ----------
    x, y, z : 1D ndarrays
        Control point coordinates
    order : int
        1 = planar surface
        2 = quadratic surface

    Returns
    -------
    coeffs : ndarray
        Least-squares coefficients
    """
    if order == 1:
        A = np.column_stack([x, y, np.ones_like(x)])
    elif order == 2:
        A = np.column_stack([
            x, y, x * y, x ** 2, y ** 2, np.ones_like(x)
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
        return a * X + b * Y + c * X * Y + d * X ** 2 + e * Y ** 2 + f


def basal_surface_from_dem(
    dem,
    transform,
    cone_poly,
    order=1,
    outer_buffer=None,
):
    """
    Reconstruct basal surface beneath a volcanic cone following
    Hunt et al. (2020).

    Parameters
    ----------
    dem : 2D ndarray
        Cone DEM
    transform : affine.Affine
        Raster transform
    cone_poly : shapely Polygon
        Cone footprint
    order : int
        Surface order (1 = planar, 2 = quadratic)
    outer_buffer : float, optional
        Basal control buffer distance (meters)

    Returns
    -------
    basal : 2D ndarray
        Reconstructed basal surface
    """
    x, y, z = sample_basal_ring(
        dem,
        transform,
        cone_poly,
        outer_buffer=outer_buffer,
    )

    coeffs = fit_basal_surface(x, y, z, order=order)

    X, Y = _pixel_grid(dem, transform)

    basal = evaluate_surface(coeffs, X, Y, order=order)
    return basal
