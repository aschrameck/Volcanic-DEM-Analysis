import os
import time
import csv
import numpy as np
import geopandas as gpd
import rasterio
from rasterio import mask
from rasterio.features import rasterize
from shapely.geometry import LineString, Point
from shapely.ops import unary_union
from scipy.stats import skew, kurtosis
import matplotlib.pyplot as plt
import traceback

from adaptive_dem_segment import dem_segment, NullError, DownloadError, DiskSpaceError
from basal_surface import basal_surface_from_dem, BasalSurfaceError


# --- Custom Exceptions---
class CRS_Error(Exception):
    def __init__(self, message="DEM is in a geographic CRS (degrees). "
                               "Reproject to a projected CRS (meters) before computing metrics."):
        self.message = message
        super().__init__(self.message)


# --- Helper Functions ---
def run_diagnostics(
    dem, transform,
    cone_poly, crater_poly,
    cone_centroid, crater_centroid,
    cone_major_axis, cone_minor_axis, cone_orientation,
    crater_major_axis, crater_minor_axis, crater_orientation,
    relief, crater_fill,
    res_x, res_y,
    ray_length=40000
):
    print("\n=== DIAGNOSTIC MODE ===\n")

    # 1. DEM properties
    print("DEM shape:", dem.shape)
    print("Resolution (x, y):", res_x, res_y)
    if abs(res_x - res_y) > 1e-6:
        print("⚠ WARNING: Non-square pixels detected.")

    # 2. DEM + polygons (correct georeferencing)
    height, width = dem.shape
    xmin = transform.c
    ymax = transform.f
    xmax = xmin + width * transform.a
    ymin = ymax + height * transform.e  # transform.e is negative

    plt.figure(figsize=(6, 6))
    plt.imshow(
        dem,
        cmap="terrain",
        extent=[xmin, xmax, ymin, ymax],
        origin="upper"
    )
    plt.plot(*cone_poly.exterior.xy, "r", lw=2, label="Cone")
    plt.plot(*crater_poly.exterior.xy, "b", lw=2, label="Crater")
    plt.scatter(cone_centroid.x, cone_centroid.y, c="yellow", s=30, label="Cone centroid")
    plt.scatter(crater_centroid.x, crater_centroid.y, c="cyan", s=30, label="Crater centroid")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("DEM with Cone and Crater Polygons")
    plt.legend()
    plt.gca().set_aspect("equal")
    plt.show()

    # 3. Mask rasterization sanity check
    out_shape = dem.shape
    cone_mask = rasterize(
        [(cone_poly, 1)],
        out_shape=out_shape,
        transform=transform,
        fill=0,
        dtype="uint8"
    ).astype(bool)

    crater_mask = rasterize(
        [(crater_poly, 1)],
        out_shape=out_shape,
        transform=transform,
        fill=0,
        dtype="uint8"
    ).astype(bool)

    plt.figure(figsize=(6, 6))
    plt.imshow(dem, cmap="gray")
    plt.imshow(cone_mask, alpha=0.4, cmap="Reds")
    plt.title("Cone Mask Check")
    plt.axis("off")
    plt.show()

    plt.figure(figsize=(6, 6))
    plt.imshow(dem, cmap="gray")
    plt.imshow(crater_mask, alpha=0.4, cmap="Blues")
    plt.title("Crater Mask Check")
    plt.axis("off")
    plt.show()

    # 4. Manual width check (0° ray)
    ray = LineString([
        (cone_centroid.x - ray_length, cone_centroid.y),
        (cone_centroid.x + ray_length, cone_centroid.y)
    ])
    inter = cone_poly.intersection(ray)
    width_manual = inter.length if not inter.is_empty else np.nan
    print(f"Manual width at 0°: {width_manual:.1f} m")

    # 5. Slope validation (avoid edges)
    i = dem.shape[0] // 2
    j = dem.shape[1] // 2

    if 1 <= i < dem.shape[0] - 1 and 1 <= j < dem.shape[1] - 1:
        dzdx = (dem[i, j + 1] - dem[i, j - 1]) / (2 * res_x)
        dzdy = (dem[i + 1, j] - dem[i - 1, j]) / (2 * res_y)
        slope_manual = np.degrees(np.arctan(np.sqrt(dzdx**2 + dzdy**2)))
        print(f"Manual slope (DEM center): {slope_manual:.2f}°")

    dy_arr, dx_arr = np.gradient(dem, res_y, res_x)
    slope_map = np.degrees(np.arctan(np.sqrt(dx_arr**2 + dy_arr**2)))
    print(f"Slope from gradient map (center): {slope_map[i, j]:.2f}°")

    # 6. Polar width distribution (cone)
    widths = np.full(360, np.nan)
    for ang in range(360):
        t = np.radians(ang)
        dx = ray_length * np.cos(t)
        dy = ray_length * np.sin(t)
        ray = LineString([
            (cone_centroid.x - dx, cone_centroid.y - dy),
            (cone_centroid.x + dx, cone_centroid.y + dy)
        ])
        inter = cone_poly.intersection(ray)
        if not inter.is_empty:
            widths[ang] = inter.length

    plt.figure(figsize=(6, 6))
    plt.polar(np.radians(np.arange(360)), widths)
    plt.title("Radial Cone Widths")
    plt.show()

    # 7. MVEE visualization (cone)
    from matplotlib.patches import Ellipse

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(*cone_poly.exterior.xy, "r", lw=2)
    ax.add_patch(Ellipse(
        xy=cone_centroid.coords[0],
        width=cone_major_axis,
        height=cone_minor_axis,
        angle=cone_orientation,
        edgecolor="blue",
        facecolor="none",
        lw=2
    ))
    ax.set_aspect("equal")
    ax.set_title("Cone MVEE Fit")
    ax.text(cone_centroid.x, cone_centroid.y,
            f"{cone_orientation:.1f}°", color="blue")
    plt.show()

    # 8. MVEE visualization (crater)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(*crater_poly.exterior.xy, "b", lw=2)
    ax.add_patch(Ellipse(
        xy=crater_centroid.coords[0],
        width=crater_major_axis,
        height=crater_minor_axis,
        angle=crater_orientation,
        edgecolor="green",
        facecolor="none",
        lw=2
    ))
    ax.set_aspect("equal")
    ax.set_title("Crater MVEE Fit")
    ax.text(crater_centroid.x, crater_centroid.y,
            f"{crater_orientation:.1f}°", color="green")
    plt.show()

    # 9. Basal-corrected relief
    plt.figure(figsize=(6, 6))
    plt.imshow(relief, cmap="inferno")
    plt.colorbar(label="Height above basal (m)")
    plt.title("Basal-Corrected Cone Relief")
    plt.axis("off")
    plt.show()

    # 10. Crater fill volume
    plt.figure(figsize=(6, 6))
    plt.imshow(crater_fill, cmap="Blues")
    plt.colorbar(label="Crater Fill Elevation (m)")
    plt.title("Crater Fill Elevation")
    plt.axis("off")
    plt.show()

    print("Diagnostics complete.\n")


def describe_stats(values):
    """
    Computes descriptive statistics for a list/array of values.
    Returns a dictionary with max, min, mean, median, std, skew, kurtosis
    """
    arr = np.array(values, dtype=float)

    arr = arr[~np.isnan(arr)]

    if arr.size == 0:
        return dict(max=np.nan, min=np.nan, mean=np.nan, median=np.nan,
                    std=np.nan, skew=np.nan, kurtosis=np.nan)

    stats = dict(
        max=np.nanmax(arr),
        min=np.nanmin(arr),
        mean=np.nanmean(arr),
        median=np.nanmedian(arr),
        std=np.nanstd(arr),
        skew=skew(arr, nan_policy="omit"),
        kurtosis=kurtosis(arr, nan_policy="omit")
    )
    return stats


def raster_values_within_polygon(raster_path, polygon):
    """
    Extracts raster values within a given polygon, safely handling nodata.
    """
    with rasterio.open(raster_path) as src:
        out_image, _ = mask.mask(src, [polygon], crop=True)
        data = out_image[0].astype(float)

        nodata = src.nodata
        if nodata is not None:
            data[data == nodata] = np.nan

        return data[np.isfinite(data)]


def slope_from_dem(raster_path):
    """
    Computes slope in degrees from a DEM raster.
    """
    with rasterio.open(raster_path) as src:
        dem = src.read(1)
        nodata = src.nodata

        dem = np.where(dem == nodata, np.nan, dem)
        res_x, res_y = src.res

    dy, dx = np.gradient(dem, res_y, res_x)
    slope = np.degrees(np.arctan(np.sqrt(dx**2 + dy**2)))

    return slope


def radial_widths(polygon, centroid):
    """
    Measures polygon width at every 1 degree from 0–359.
    Returns a list of 360 width values.
    """
    ray_length = max(polygon.length, polygon.area**0.5) * 2
    widths = []

    for angle in range(360):
        theta = np.radians(angle)

        # Build ray that goes both directions
        dx = ray_length * np.cos(theta)
        dy = ray_length * np.sin(theta)

        p1 = Point(centroid.x - dx, centroid.y - dy)
        p2 = Point(centroid.x + dx, centroid.y + dy)
        ray = LineString([p1, p2])

        # Intersection with polygon
        inter = polygon.intersection(ray)

        if inter.is_empty:
            widths.append(np.nan)
            continue

        # Handle LineString or MultiLineString
        if inter.geom_type == "LineString":
            widths.append(inter.length)

        elif inter.geom_type == "MultiLineString":
            # take the longest continuous segment
            widths.append(max(seg.length for seg in inter.geoms))

        else:
            widths.append(np.nan)

    return widths


def fit_mvee(polygon, tol=1e-3):
    """
    Fits a minimum-volume enclosing ellipse (MVEE) to a polygon.
    Returns:
        major_axis (float)
        minor_axis (float)
        orientation_deg (float)  # degrees CCW from +X, [0, 180)
    """
    # Extract exterior coordinates
    pts = np.asarray(polygon.exterior.coords[:-1])
    if pts.shape[0] < 3:
        return np.nan, np.nan, np.nan

    # MVEE algorithm
    N, d = pts.shape
    Q = np.vstack([pts.T, np.ones(N)])
    u = np.full(N, 1 / N)
    err = tol + 1

    # Max iterations cap
    max_iter = 1000
    iters = 0

    while err > tol and iters < max_iter:
        X = Q @ np.diag(u) @ Q.T
        M = np.einsum("ij,ji->i", Q.T @ np.linalg.inv(X), Q)
        j = np.argmax(M)
        step = (M[j] - d - 1) / ((d + 1) * (M[j] - 1))
        u_new = (1 - step) * u
        u_new[j] += step
        err = np.linalg.norm(u_new - u)
        u = u_new

        iters += 1
        if iters == max_iter:
            return np.nan, np.nan, np.nan

    center = pts.T @ u
    # The shape matrix A defines the ellipse: (x-c)T * A * (x-c) = 1
    A = np.linalg.inv(pts.T @ np.diag(u) @ pts - np.outer(center, center)) / d

    # Decomposition for geometric parameters
    eigenvals, eigenvecs = np.linalg.eigh(A)

    order = np.argsort(eigenvals)  # smallest eigenval → largest axis
    major = 2 / np.sqrt(eigenvals[order[0]])
    minor = 2 / np.sqrt(eigenvals[order[1]])
    angle = np.degrees(np.arctan2(
        eigenvecs[1, order[0]],
        eigenvecs[0, order[0]]
    )) % 180

    return major, minor, angle


def safe_div(numerator, denominator):
    """Safely divides two numbers, returning NaN if denominator is zero or NaN."""
    if denominator is None or np.isnan(denominator) or denominator == 0:
        return np.nan
    return numerator / denominator


def csv_writing(lock, cone_dem, output_csv, base_name, data):

    """Writes metrics to a CSV file."""

    if output_csv is None:
        output_csv = os.path.dirname(cone_dem)
    if os.path.isdir(output_csv):
        csv_path = os.path.join(output_csv, f"{base_name}_metrics.csv")
    else:
        csv_path = output_csv

    new_file = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)

        headers = [
                "Warnings", "Number", "Latitude", "Longitude",
                "Cone_Height_Max", "Cone_Volume", "Cone_Elev_Max", "Cone_Elev_Min",
                "Cone_Elev_Mean", "Cone_Elev_Median", "Cone_Elev_Std", "Cone_Elev_Skew",
                "Cone_Elev_Kurt", "Cone_Basal_Perimeter", "Cone_Basal_Area",
                "Cone_Width_Max", "Cone_Width_Min", "Cone_Width_Mean", "Cone_Width_Median",
                "Cone_Width_Std", "Cone_Width_Skew", "Cone_Width_Kurt",
                "Cone_Slope_Max", "Cone_Slope_Min", "Cone_Slope_Mean", "Cone_Slope_Median",
                "Cone_Slope_Std", "Cone_Slope_Skew", "Cone_Slope_Kurt",
                "Crater_Depth_Max", "Crater_Fill_Volume", "Crater_Basal_Perimeter",
                "Crater_Basal_Area", "Crater_Width_Max", "Crater_Width_Min", "Crater_Width_Mean",
                "Crater_Width_Median", "Crater_Width_Std", "Crater_Width_Skew",
                "Crater_Width_Kurt", "Crater_Slope_Max", "Crater_Slope_Min",
                "Crater_Slope_Mean", "Crater_Slope_Median", "Crater_Slope_Std",
                "Crater_Slope_Skew", "Crater_Slope_Kurt",
                "Cone_Elongation", "Cone_Circularity", "Cone_Eccentricity",
                "Cone_Ellipse_MajorAxis", "Cone_Ellipse_MinorAxis", "Cone_Ellipse_Orientation",
                "Crater_Elongation", "Crater_Circularity", "Crater_Eccentricity",
                "Crater_Ellipse_MajorAxis", "Crater_Ellipse_MinorAxis", "Crater_Ellipse_Orientation",
                "ConeHeight/ConeAvgWidth", "ConeHeight/ConeMaxWidth",
                "CraterDepth/CraterAvgWidth", "ConeHeight/CraterAvgWidth",
                "CraterAvgWidth/ConeAvgWidth", "CraterDepth/ConeHeight"
            ]
        if lock is not None:
            with lock:
                if new_file:
                    writer.writerow(headers)
                writer.writerow(data)
        else:
            if new_file:
                writer.writerow(headers)
            writer.writerow(data)

    return csv_path


# --- Main Function ---
def cone_metrics(lat, lon, num, cone_dem, cone_boundary, crater_boundary, WARNING, warning_reasons,
                 output_csv=None, diag=False, lock=None):
    """
    Calculate morphometric parameters for cone and crater from segmented raster outputs.

    Records:
      Cone Number, Latitude, Longitude
      Cone: height, basal elevation (max, min, mean, median, std, skew, kurtosis),
            basal area (perimeter, area),
            width (max, min, mean, median, std, skew, kurtosis),
            slope (max, min, mean, median, std, skew, kurtosis)
      Crater: depth, basal area (perimeter, area),
              width (max, min, mean, median, std, skew, kurtosis),
              slope (max, min, mean, median, std, skew, kurtosis)
      Calculated: ellipticity, circularity, and eccentricity
      Ratios: Cone Height/Cone Average Width, Cone Height/Cone Max Width,
              Crater Depth/Crater Average Width, Cone Height/Crater Average Width,
              Crater Average Width/Cone Average Width, and Crater Depth/Cone Height

    Parameters
    ----------
    lat, lon : float
        Cone center coordinates (latitude and longitude).
    num : int
        Cone number (for naming purposes).
    cone_dem : str
        Path to segmented cone DEM.
    cone_boundary : str
        Path to cone boundary shapefile or feature class.
    crater_boundary : str
        Path to crater boundary shapefile or feature class.
    WARNING : bool
        Flag indicating if there were warnings during DEM segmentation.
    warning_reasons : list of str
        List of warning reasons from DEM segmentation.
    output_csv : str, optional
        Path to output CSV file or directory.
        Default: None
    diag : bool, optional
        If True, runs diagnostic checks and visualizations.
        Default: False
    lock : multiprocessing.Lock, optional
        Lock for synchronizing CSV writing in parallel processing.
        Default: None
    """
    # --- Set up ---
    start = time.perf_counter()

    if diag:
        print(f"\nStarting cone_metrics for cone #{num}: ({lat}, {lon})")

    lat_str = str(lat).replace('.', '_')
    lon_str = str(lon).replace('.', '_')
    base_name = f"{num}_{lat_str}x{lon_str}"

    # Load DEM and shapefiles
    if diag:
        print("\nLoading DEM and polygons...")
    with rasterio.open(cone_dem) as src:
        dem = src.read(1)
        transform = src.transform
        res_x, res_y = src.res
        dem_crs = src.crs

    cone_gdf = gpd.read_file(cone_boundary)
    crater_gdf = gpd.read_file(crater_boundary)

    # Reproject polygons to a projected CRS (meters) if DEM is geographic
    if dem_crs is None or dem_crs.is_geographic:  # degrees
        if diag:
            print("⚠ DEM is geographic (lat/lon). Reprojecting polygons to projected CRS in meters.")

        raise CRS_Error(
            f"DEM for cone #{num} is in a geographic CRS (degrees). "
            "Reproject DEM to a projected CRS (meters) before computing metrics."
            )

    # Ensure DEM and polygons CRS match
    if cone_gdf.crs != dem_crs:
        cone_gdf = cone_gdf.to_crs(dem_crs)
    if crater_gdf.crs != dem_crs:
        crater_gdf = crater_gdf.to_crs(dem_crs)

    cone_poly = unary_union(cone_gdf.geometry)
    crater_poly = unary_union(crater_gdf.geometry)

    cone_centroid = cone_poly.centroid
    crater_centroid = crater_poly.centroid

    # Ellipse (MVEE) metrics
    if diag:
        print("\nFitting ellipses to cone and crater polygons...")
    cone_major_axis, cone_minor_axis, cone_orientation = fit_mvee(cone_poly)
    crater_major_axis, crater_minor_axis, crater_orientation = fit_mvee(crater_poly)

    # --- Extract raster values within polygons ---
    if diag:
        print("\nExtracting raster values within cone and crater polygons...")
    cone_elevs = raster_values_within_polygon(cone_dem, cone_poly)

    # Compute slope values
    slope = slope_from_dem(cone_dem)

    with rasterio.open(cone_dem) as src:
        out_transform = src.transform
        out_shape = (src.height, src.width)

    cone_mask = rasterize(
        [(cone_poly, 1)],
        out_shape=out_shape,
        transform=out_transform,
        fill=0,
        dtype="uint8"
    ).astype(bool)

    crater_mask = rasterize(
        [(crater_poly, 1)],
        out_shape=out_shape,
        transform=out_transform,
        fill=0,
        dtype="uint8"
    ).astype(bool)

    cone_slope_vals = slope[cone_mask]
    crater_slope_vals = slope[crater_mask]

    # Calculate widths
    cone_widths = radial_widths(cone_poly, cone_centroid)
    crater_widths = radial_widths(crater_poly, crater_centroid)

    # --- Basal surface correction ---
    if diag:
        print("\nComputing basal surface and height-above-basal DEM...")
    try:
        basal_surface = basal_surface_from_dem(
            dem=dem,
            transform=transform,
            cone_poly=cone_poly,
            order=1  # planar, as in Hunt et al.
        )
        basal_surface[~cone_mask] = np.nan
    except Exception as e:
        raise BasalSurfaceError(f"Basal surface fitting failed: {e}")

    # Height-above-basal DEM
    relief = dem - basal_surface

    # Mask invalid areas
    relief[~cone_mask] = np.nan
    rim_candidates = relief[cone_mask & ~crater_mask]

    if np.nanstd(rim_candidates) > 0.5 * np.nanmean(rim_candidates):
        warning_reasons.append("Crater rim elevation highly variable (possible breach)")

    if diag:
        print("\nStarting metric calculations...")
    # --- Slope corrected metrics ---
    # Cone max height
    cone_max_height = np.nanmax(relief)

    # Cone volume
    pixel_area = abs(res_x * res_y)
    cone_volume = np.nansum(np.clip(relief, 0, None) * pixel_area)

    # --- Morphometrically computed metrics ---
    cone_elev_stats = describe_stats(cone_elevs)
    cone_slope_stats = describe_stats(cone_slope_vals)
    crater_slope_stats = describe_stats(crater_slope_vals)
    cone_width_stats = describe_stats(cone_widths)
    crater_width_stats = describe_stats(crater_widths)

    # Crater fill volume
    rim_mask = cone_mask & ~crater_mask
    rim_elev = np.nanpercentile(dem[rim_mask], 95)

    crater_fill = rim_elev - dem
    crater_fill[~crater_mask] = np.nan
    crater_fill[crater_fill < 0] = 0

    crater_fill_volume = np.nansum(crater_fill) * pixel_area

    # Crater depth
    if np.count_nonzero(rim_mask) < 10:
        warning_reasons.append("Insufficient rim pixels for reliable crater metrics")
        crater_max_depth = np.nan
    else:
        crater_floor = np.nanmin(dem[crater_mask])
        crater_max_depth = rim_elev - crater_floor

    # Area and perimeter
    cone_area = cone_poly.area
    cone_perimeter = cone_poly.length
    crater_area = crater_poly.area
    crater_perimeter = crater_poly.length

    # Calculated metrics
    pi = np.pi
    cone_circularity = safe_div(4 * pi * cone_area, (cone_perimeter ** 2))
    crater_circularity = safe_div(4 * pi * crater_area, (crater_perimeter ** 2))
    cone_elongation = safe_div(cone_minor_axis, cone_major_axis)
    crater_elongation = safe_div(crater_minor_axis, crater_major_axis)
    cone_eccentricity = np.sqrt(1 - (safe_div(cone_width_stats["min"] ** 2, cone_width_stats["max"] ** 2)))
    crater_eccentricity = np.sqrt(1 - (safe_div(crater_width_stats["min"] ** 2, crater_width_stats["max"] ** 2)))

    # Ratios
    cone_avg_width = cone_width_stats["mean"]
    crater_avg_width = crater_width_stats["mean"]
    ratios = dict(
        cone_h_avg_w=safe_div(cone_max_height, cone_avg_width),
        cone_h_max_w=safe_div(cone_max_height, cone_width_stats["max"]),
        crater_d_avg_w=safe_div(crater_max_depth, crater_avg_width),
        cone_h_crater_avg_w=safe_div(cone_max_height, crater_avg_width),
        crater_avg_w_cone_avg_w=safe_div(crater_avg_width, cone_avg_width),
        crater_d_cone_h=safe_div(crater_max_depth, cone_max_height)
    )

    # --- Warnings ---
    if diag:
        print("\nEvaluating warnings...")
    shape_mismatch = False

    if np.isfinite(cone_minor_axis) and np.isfinite(crater_minor_axis):
        log_axis_diff = abs(np.log(
            (cone_minor_axis / cone_major_axis) /
            (crater_minor_axis / crater_major_axis)
        ))
        shape_mismatch |= log_axis_diff > np.log(2)

    if np.isfinite(cone_eccentricity) and np.isfinite(crater_eccentricity):
        shape_mismatch |= abs(cone_eccentricity - crater_eccentricity) > 0.3

    if shape_mismatch:
        warning_reasons.append(
            "Cone and crater shapes differ significantly (possible misidentification)"
        )
    if cone_circularity > 0.9:
        warning_reasons.append("Cone shape is nearly circular (circularity > 0.9)")
    if cone_elongation < 0.3:
        warning_reasons.append("Highly elongated feature (likely fissure)")

    # Initialize warning string
    warning = ""
    if WARNING:
        warning = "WARNING"
        if warning_reasons:
            warning += " (" + "; ".join(warning_reasons) + ")"

    # --- Write to CSV ---
    if diag:
        print("\nWriting metrics to CSV...")
    data = [
        warning, num, lat, lon,
        cone_max_height, cone_volume,
        cone_elev_stats["max"], cone_elev_stats["min"], cone_elev_stats["mean"],
        cone_elev_stats["median"], cone_elev_stats["std"], cone_elev_stats["skew"],
        cone_elev_stats["kurtosis"], cone_perimeter, cone_area,
        cone_width_stats["max"], cone_width_stats["min"], cone_width_stats["mean"],
        cone_width_stats["median"], cone_width_stats["std"], cone_width_stats["skew"],
        cone_width_stats["kurtosis"], cone_slope_stats["max"], cone_slope_stats["min"],
        cone_slope_stats["mean"], cone_slope_stats["median"], cone_slope_stats["std"],
        cone_slope_stats["skew"], cone_slope_stats["kurtosis"], crater_max_depth,
        crater_fill_volume, crater_perimeter, crater_area, crater_width_stats["max"],
        crater_width_stats["min"], crater_width_stats["mean"],
        crater_width_stats["median"], crater_width_stats["std"],
        crater_width_stats["skew"], crater_width_stats["kurtosis"],
        crater_slope_stats["max"], crater_slope_stats["min"],
        crater_slope_stats["mean"], crater_slope_stats["median"],
        crater_slope_stats["std"], crater_slope_stats["skew"],
        crater_slope_stats["kurtosis"], cone_elongation, cone_circularity,
        cone_eccentricity, cone_major_axis, cone_minor_axis, cone_orientation,
        crater_elongation, crater_circularity, crater_eccentricity,
        crater_major_axis, crater_minor_axis, crater_orientation,
        ratios["cone_h_avg_w"], ratios["cone_h_max_w"],
        ratios["crater_d_avg_w"], ratios["cone_h_crater_avg_w"],
        ratios["crater_avg_w_cone_avg_w"], ratios["crater_d_cone_h"]]

    csv_path = csv_writing(lock, cone_dem, output_csv, base_name, data)

    # --- Diagnostics ---
    if diag:
        run_diagnostics(
            dem=dem,
            transform=transform,
            cone_poly=cone_poly,
            crater_poly=crater_poly,
            cone_centroid=cone_centroid,
            crater_centroid=crater_centroid,
            cone_major_axis=cone_major_axis,
            cone_minor_axis=cone_minor_axis,
            cone_orientation=cone_orientation,
            crater_major_axis=crater_major_axis,
            crater_minor_axis=crater_minor_axis,
            crater_orientation=crater_orientation,
            relief=relief, crater_fill=crater_fill,
            res_x=res_x, res_y=res_y
        )
        print(f"\nRuntime: {time.perf_counter() - start:.2f} sec")
        print(f"Metrics saved to: {csv_path}")

    return csv_path


# --- Test Cases ---
if __name__ == "__main__":
    start = time.perf_counter()
    polygon_folder = r"D:\Cone_Polygons"
    dem_folder = r"D:\Cone_DEMS"
    csv_out = r"D:\Metrics.csv"

    print("Starting test cases for cone_metrics...\n")

    test_cases = [
            {"lat": 35.597220, "lon": -111.610612, "num": 1},  # Crater 01 (very elongated crater, more like a fissure)
            {"lat": 35.579547, "lon": -111.581650, "num": 2},  # Crater 02 (low elevation rim and open on one side)
            {"lat": 35.558799, "lon": -111.605790, "num": 3},  # Crater 03 (slightly elongated crater)
            {"lat": 35.3641, "lon": -111.5033, "num": 4},      # Sunset Crater - GOOD
            {"lat": 35.582329, "lon": -111.631927, "num": 5},  # SP Crater
            {"lat": 35.543845, "lon": -111.637273, "num": 6},  # Colton Crater !!! FAILS CONE CHECK !!!
            {"lat": 0, "lon": 0, "num": 7},                # Ocean (Null Error)
            {"lat": 39.7392, "lon": -104.9903, "num": 8},  # Denver, CO (Cone Error)
        ]

    for case in test_cases:
        print("\n--- Running coordinates:", case["lat"], case["lon"], "---")

        try:
            dem = dem_segment(case["lat"], case["lon"], case["num"], polygon_folder, dem_folder, diag=True)
            cone_dem_path, cone_polygon_path, crater_polygon_path, WARNING, warning_reasons = dem
            cone_metrics(
                lat=case["lat"],
                lon=case["lon"],
                num=case["num"],
                cone_dem=cone_dem_path,
                cone_boundary=cone_polygon_path,
                crater_boundary=crater_polygon_path,
                WARNING=WARNING,
                warning_reasons=warning_reasons,
                output_csv=csv_out,
                diag=True
            )

        except (NullError, DownloadError, DiskSpaceError, BasalSurfaceError) as e:
            print(f"Expected error: {e}")
        except Exception:
            print(traceback.format_exc())

    end = time.perf_counter()
    print(f"\nAll test cases completed in {end - start:.2f} seconds.")
