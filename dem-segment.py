import os
import time
import math
import traceback
import requests
import rasterio
import numpy as np
import geopandas as gpd
from rasterio.merge import merge
from rasterio.mask import mask
from pyproj import Transformer
from shapely.geometry import Polygon, LineString, mapping


# --- Custom exceptions---
class NullError(Exception):
    def __init__(self, message="Encountered null values, points in an invalid DEM."):
        self.message = message
        super().__init__(self.message)


class DownloadError(Exception):
    def __init__(self, message="Failed to download DEM from TNM."):
        self.message = message
        super().__init__(self.message)


# --- Helper: Save shapefile ---
def save_shapefile(geom, path, crs="EPSG:4326", extra_fields=None):
    if not isinstance(geom, list):
        geom = [geom]
    gdf = gpd.GeoDataFrame(extra_fields or {}, geometry=geom, crs=crs)
    gdf.to_file(path)
    return path


# --- Main Function ---
def dem_segment(lat, lon, num, radius, polygon_folder, dem_folder, diag=False):
    """
    Query TNM Access API to find and download the 'best' available 3DEP DEM intersecting the bbox
    around (lat, lon) with given radius in meters. Returns path to downloaded GeoTIFF.

    Note: This function uses the public TNM Access API documented by USGS.

    Parameters
    ----------
    lat, lon : float
        Cone center coordinates (latitude and longitude).
    num : int
        Cone number (for naming purposes).
    radius : float
        Radius of the cone (in meters).
    polygon_folder : str
        Folder where boundary and crater polygons will be saved.
    dem_folder : str
        Folder where output DEMs will be saved.
    diag : bool
        If True, prints diagnostic info and saves radial lines shapefile.
        Default: False
    """

    # --- Set Up ---
    start_time = time.perf_counter()
    os.makedirs(dem_folder, exist_ok=True)
    os.makedirs(polygon_folder, exist_ok=True)

    # Safe filenames
    lat_str = str(lat).replace('.', '_')
    lon_str = str(lon).replace('.', '_')
    base_name = f"{num}_{lat_str}x{lon_str}"

    print(f"\nStarting DEM download for point {num}: ({lat}, {lon}), radius={radius} m")

    # Compute bounding box
    deg_lat = radius / 111_000  # 1 degree ≈ 111 km
    deg_lon = radius / (111_000 * math.cos(math.radians(lat)))
    minx, maxx = lon - deg_lon, lon + deg_lon
    miny, maxy = lat - deg_lat, lat + deg_lat

    bbox_str = f"{minx},{miny},{maxx},{maxy}"

    # --- Query TNM Access API ---
    # Official API docs: https://apps.nationalmap.gov/tnmaccess/api/v1/docs
    url = "https://tnmaccess.nationalmap.gov/api/v1/products"
    params = {
        "datasets": "Digital Elevation Model (DEM) 1 meter",
        "bbox": bbox_str,
        "outputFormat": "JSON",
        "sort": "acquisitionDate",
        "order": "desc"
    }

    try:
        response = requests.get(url, params=params, timeout=60)
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        raise DownloadError(f"Failed API query: {e}")

    items = data.get("items", [])

    if not items:
        raise NullError()

    # Download tiles
    tifs = []
    for i, item in enumerate(items[:]):
        download_url = item.get("downloadURL")
        if not download_url:
            continue
        tile_path = os.path.join(dem_folder, f"{base_name}_tile{i+1}.tif")

        print(f"Downloading DEM tile {i+1}...")
        try:
            with requests.get(download_url, stream=True, timeout=120) as r:
                r.raise_for_status()
                with open(tile_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
            tifs.append(tile_path)
        except Exception as e:
            print(f"Warning: could not download tile {i+1}: {e}")

    if not tifs:
        raise DownloadError("Failed to download any DEM tiles.")

    # Mosaic tiles if needed
    if len(tifs) > 1:
        print("Creating DEM mosaic...")
        src_files = [rasterio.open(fp) for fp in tifs]
        mosaic, out_trans = merge(src_files)
        out_meta = src_files[0].meta.copy()
        out_meta.update({
            "driver": "GTiff",
            "height": mosaic.shape[1],
            "width": mosaic.shape[2],
            "transform": out_trans,
            "compress": "lzw"
        })
        raster_path = os.path.join(dem_folder, f"{base_name}_mosaic.tif")
        with rasterio.open(raster_path, "w", **out_meta) as dest:
            dest.write(mosaic)
        for s in src_files:
            s.close()
    else:
        raster_path = tifs[0]

    # Read DEM
    with rasterio.open(raster_path) as src:
        dem_array = src.read(1).astype(float)
        transform = src.transform
        nrows, ncols = dem_array.shape
        extent = src.bounds
        cell_size = transform[0]

    # Create transformer once (outside function)
    transformer = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)

    # --- Helper Function ---
    def get_elevation(lon, lat):
        """
        Get elevation (in meters) for a geographic coordinate (lon, lat).
        Automatically transforms to the DEM CRS if needed.
        """
        # Convert from lon/lat (WGS84) to DEM CRS (usually UTM)
        x, y = transformer.transform(lon, lat)

        # Convert projected coords to raster row/col
        col = int((x - extent.left) / cell_size)
        row = int((extent.top - y) / abs(transform[4]))

        if 0 <= row < nrows and 0 <= col < ncols:
            val = dem_array[row, col]
            return None if np.isnan(val) else val
        return None

    center_x, center_y = lon, lat

    center_elev = get_elevation(center_x, center_y)
    if center_elev is None:
        for tif in tifs:
            try:
                os.remove(tif)
            except Exception:
                pass
        if len(tifs) > 1:
            try:
                os.remove(raster_path)
            except Exception:
                pass
        raise NullError("Cone validation failed: missing elevation data.")

    # Parameters
    radial_steps = 72
    radial_angles = [i * 360 / radial_steps for i in range(radial_steps)]
    cone_edge_distances, crater_rim_distances = [], []
    rim_search_limit = max(radius / 3, 400)
    point_spacing = 0.5  # meters

    # --- Radial Sampling ---
    for angle_deg in radial_angles:
        rim_r = radius
        angle_rad = math.radians(angle_deg)
        rim_elev = -float('inf')
        elev_samples = []

        # Flat terrain pre-check
        for r in np.arange(0, radius, 100):
            x = center_x + (r / 111_000) * math.cos(angle_rad)
            y = center_y + (r / 111_000) * math.sin(angle_rad)
            elev = get_elevation(x, y)
            if elev is not None:
                elev_samples.append(elev)

        if len(elev_samples) > 3 and (max(elev_samples) - min(elev_samples)) < 0.5:
            # Flat terrain: use full radius directly
            cone_edge_distances.append(radius)
            crater_rim_distances.append(rim_search_limit / 2)
            continue

        # Find crater rim
        for r in np.arange(0, rim_search_limit, point_spacing):
            x = center_x + (r / 111_000) * math.cos(angle_rad)
            y = center_y + (r / 111_000) * math.sin(angle_rad)
            elev = get_elevation(x, y)
            if elev is None:
                continue
            if elev > rim_elev:
                rim_elev = elev
                rim_r = r
        crater_rim_distances.append(rim_r)

        # Find cone base
        r = rim_r
        prev_elev = rim_elev
        min_elev = float('inf')
        min_r = r
        prev_slope = 0.0

        rising_count = 0
        flat_count = 0
        slope_points = 3  # consecutive points to confirm slope change
        slope_change_threshold = 0.2  # m/m change to indicate slope break

        # Skip any initial descent from crater rim toward center
        center_elev = get_elevation(center_x, center_y)
        while prev_elev > center_elev and r < radius:
            r += point_spacing
            x = center_x + (r / 111_000) * math.cos(angle_rad)
            y = center_y + (r / 111_000) * math.sin(angle_rad)
            elev = get_elevation(x, y)
            if elev is None:
                continue
            prev_elev = elev

        # Search for min elevation before ground flattening, rising, or slope break
        stop_reason = "max radius reached"
        while r <= radius:
            x = center_x + (r / 111_000) * math.cos(angle_rad)
            y = center_y + (r / 111_000) * math.sin(angle_rad)
            elev = get_elevation(x, y)
            if elev is None:
                r += point_spacing
                continue

            # Track minimum elevation
            if elev < min_elev and rising_count == 0:
                min_elev = elev
                min_r = r
                flat_count = 0
            # Rising
            elif elev > prev_elev:
                rising_count += 1
                flat_count = 0
                if rising_count >= slope_points:
                    stop_reason = f"sustained rise detected after {slope_points} points"
                    break
            # Flattening
            elif abs(elev - prev_elev) < 0.2:
                flat_count += 1
                if flat_count >= slope_points:
                    stop_reason = f"flat terrain ({flat_count} points within 0.25 m)"
                    break
            # Declining
            else:
                rising_count = 0
                flat_count = 0

            # Compute short-term slope
            r_back = max(r - slope_points, rim_r)
            x_back = center_x + (r_back / 111_000) * math.cos(angle_rad)
            y_back = center_y + (r_back / 111_000) * math.sin(angle_rad)
            elev_back = get_elevation(x_back, y_back)
            if elev_back is not None and abs(r - r_back) > 1e-6:
                slope = (elev - elev_back) / (r - r_back)
            else:
                slope = 0

            # Stop if slope changes abruptly (terrain inflection)
            if abs(slope - prev_slope) > slope_change_threshold and r > min_r + slope_points:
                stop_reason = f"slope inflection ({slope} - {prev_slope} > {slope_change_threshold:.3f} m/m"
                break

            prev_slope = slope
            prev_elev = elev
            r += point_spacing

        # Fallback: use radius
        if math.isnan(min_r) or min_r == rim_r or min_r >= radius:
            min_r = radius
            if stop_reason == "max radius reached":
                stop_reason = "no valid base found — fallback to radius"

        # Use min_r as the detected base
        cone_edge_distances.append(min_r)

        if diag:
            print(f"Radial {angle_deg:>5.1f}° ended at {min_r:.1f} m ({stop_reason})")

    # --- Create Output ---
    smoothed_distances = []
    window = 2
    for i in range(radial_steps):
        neighbors = [cone_edge_distances[(i + j) % radial_steps] for j in range(-window, window + 1)]
        smoothed_distances.append(sum(neighbors) / len(neighbors))

    # Create crater and cone polygons
    crater_pts = []
    base_pts = []
    crater_elevs, base_elevs = [], []
    for i, angle_deg in enumerate(radial_angles):
        a = math.radians(angle_deg)
        r_rim = crater_rim_distances[i]
        r_base = smoothed_distances[i]
        crater_x = center_x + r_rim * math.cos(a) / 111_000
        crater_y = center_y + r_rim * math.sin(a) / 111_000
        base_x = center_x + r_base * math.cos(a) / 111_000
        base_y = center_y + r_base * math.sin(a) / 111_000
        crater_pts.append((crater_x, crater_y))
        base_pts.append((base_x, base_y))
        elev_c = get_elevation(crater_x, crater_y)
        elev_b = get_elevation(base_x, base_y)
        if elev_c:
            crater_elevs.append(elev_c)
        if elev_b:
            base_elevs.append(elev_b)

    crater_poly = Polygon(crater_pts)
    base_poly = Polygon(base_pts)

    # Save shapefiles
    crater_path = os.path.join(polygon_folder, f"{base_name}_crater.shp")
    base_path = os.path.join(polygon_folder, f"{base_name}_base.shp")
    save_shapefile(crater_poly, crater_path)
    save_shapefile(base_poly, base_path)

    # Clip DEM to base polygon
    clipped_dem_path = os.path.join(dem_folder, f"{base_name}_DEM.tif")
    with rasterio.open(raster_path) as src:
        # Reproject polygon to match DEM CRS
        base_poly_proj = gpd.GeoSeries([base_poly], crs="EPSG:4326").to_crs(src.crs).iloc[0]
        out_image, out_transform = mask(src, [mapping(base_poly_proj)], crop=True)
        out_meta = src.meta.copy()
        out_meta.update({
            "driver": "GTiff",
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform,
            "compress": "lzw"
        })
        with rasterio.open(clipped_dem_path, "w", **out_meta) as dest:
            dest.write(out_image)

    # Delete intermediate tiles
    for tif in tifs:
        try:
            os.remove(tif)
        except Exception:
            pass
    if len(tifs) > 1:
        try:
            os.remove(raster_path)
        except Exception:
            pass

    # Diagnostics
    if diag:
        radials = []
        for i, angle_deg in enumerate(radial_angles):
            a = math.radians(angle_deg)
            r = smoothed_distances[i]
            x2 = center_x + r * math.cos(a) / 111_000
            y2 = center_y + r * math.sin(a) / 111_000
            radials.append(LineString([(center_x, center_y), (x2, y2)]))
        radials_path = os.path.join(polygon_folder, f"{base_name}_radials.shp")
        save_shapefile(radials, radials_path)

    end_time = time.perf_counter()
    print(f"Completed in {end_time - start_time:.1f} s")
    print(f"Final DEM saved: {clipped_dem_path}")

    return clipped_dem_path, base_path, crater_path


# --- Testing ---
if __name__ == "__main__":
    polygon_folder = r"D:\NASA_Research_Project\Cone_Polygons"
    dem_folder = r"D:\NASA_Research_Project\Cone_DEMS"

    test_cases = [
            {"lat": 35.3641, "lon": -111.5033, "radius": 4000},  # Sunset Crater
            {"lat": 0, "lon": 0, "radius": 4000},                # Ocean (Null Error)
        ]

    for case in test_cases:
        print("\n--- Testing coordinates:", case["lat"], case["lon"], "Radius:", case["radius"], "---")

        try:
            dem_segment(case["lat"], case["lon"], 1, case["radius"], polygon_folder, dem_folder, diag=True)

        except (NullError, DownloadError) as e:
            print(f"Expected error: {e}")
        except Exception:
            print(traceback.format_exc())
