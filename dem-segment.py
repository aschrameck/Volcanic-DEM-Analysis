import os
import time
import math
import traceback
import requests
import rasterio
import shutil
import numpy as np
import geopandas as gpd
from rasterio.merge import merge
from rasterio.mask import mask
from pyproj import Transformer
from shapely.geometry import Polygon, LineString, mapping


# --- Custom Exceptions---
class NullError(Exception):
    def __init__(self, message="Encountered null values, points in an invalid DEM."):
        self.message = message
        super().__init__(self.message)


class DownloadError(Exception):
    def __init__(self, message="Failed to download DEM from TNM."):
        self.message = message
        super().__init__(self.message)


class ConeError(Exception):
    def __init__(self, message="Cone validation failed. Expected: crater > center > base."):
        self.message = message
        super().__init__(self.message)


class DiskSpaceError(Exception):
    def __init__(self, message="Insufficient disk space to download DEM tiles."):
        super().__init__(message)


# --- Helper Functions ---
def save_shapefile(geom, path, crs="EPSG:4326", extra_fields=None):
    if not isinstance(geom, list):
        geom = [geom]
    gdf = gpd.GeoDataFrame(extra_fields or {}, geometry=geom, crs=crs)
    gdf.to_file(path)
    return path


def ensure_free_space(folder, required_bytes):
    """
    Ensure that the drive containing `folder` has at least `required_bytes` free.
    Raises DiskSpaceError if not enough space is available.
    """
    try:
        usage = shutil.disk_usage(folder)
        free = usage.free
    except Exception as e:
        raise DiskSpaceError(f"Unable to check free disk space: {e}")

    if free < required_bytes:
        gb_free = free / (1024**3)
        gb_req = required_bytes / (1024**3)
        raise DiskSpaceError(
            f"Not enough disk space (free: {gb_free:.2f} GB, required: {gb_req:.2f} GB)."
        )


# --- Main Function ---
def dem_segment(lat, lon, num, polygon_folder, dem_folder, diag=False):
    """
    Queries USGS TNM Access API for 1-meter 3DEP DEMs around center coordinate (lat, lon), mosaicking
    tiles if needed. Uses radial sampling with adaptive edge detection to automatically locate crater
    and basal boundaries, then clips the DEM and saves boundary polygons.

    Returns file paths to the processed outputs.

    Parameters
    ----------
    lat, lon : float
        Cone center coordinates (latitude and longitude).
    num : int
        Cone number (for naming purposes).
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

    print(f"\nStarting DEM download for cone #{num}: ({lat}, {lon})")

    # Constants
    initial_radius = 2000  # 2 km
    max_radius = 400000  # 400 km
    increment = 2000  # 2 km
    current_radius = initial_radius

    all_tiles = []
    raster_path = None
    final_found = False

    # --- Outer Loop: Expand search area until found ---
    while current_radius <= max_radius and not final_found:
        print(f"\nAttempting radius {current_radius/1000:.1f} km:")

        # Compute bounding box
        deg_lat = current_radius / 111_000
        deg_lon = current_radius / (111_000 * math.cos(math.radians(lat)))

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
            try:
                data = response.json()
            except Exception:
                print("\n--- RAW RESPONSE FROM TNM ---")
                print(response.text[:1000])   # let’s inspect first 1k chars
                print("----------------------------\n")
                raise DownloadError("TNM did not return JSON.")
        except Exception as e:
            # Delete temporary raster files
            try:
                if raster_path and os.path.exists(raster_path):
                    os.remove(raster_path)
            except Exception as e:
                print(f"Warning: failed to delete temp raster: {e}")

            for fp in all_tiles:
                try:
                    if os.path.exists(fp):
                        os.remove(fp)
                except Exception:
                    pass

            end_time = time.perf_counter()
            print(f"Function finished in {end_time - start_time:.1f} s")
            raise DownloadError(f"Failed API query: {e}")

        items = data.get("items", [])
        print(f"TNM metadata returned at radius {current_radius:.1f} km: {len(items)}")

        if not items:
            current_radius += increment
            continue

        # Download new tiles
        for i, item in enumerate(items[:]):
            download_url = item.get("downloadURL")

            tile_name = os.path.basename(download_url).split("?")[0]
            tile_path = os.path.join(dem_folder, tile_name)

            try:
                # First request: HEAD to get file size
                head = requests.head(download_url, timeout=30)
                head.raise_for_status()
                size_bytes = int(head.headers.get("Content-Length", "0"))

                # Require at least 1.25× the expected file size (buffer for GeoTIFF overhead)
                required = int(size_bytes * 1.25) if size_bytes > 0 else 200 * 1024 * 1024

                ensure_free_space(dem_folder, required)

                # Second request: actual download
                with requests.get(download_url, stream=True, timeout=120) as resp:
                    resp.raise_for_status()
                    print(f"Downloading {tile_name} ...")
                    with open(tile_path, "wb") as f:
                        for chunk in resp.iter_content(chunk_size=65536):
                            if chunk:
                                f.write(chunk)

                all_tiles.append(tile_path)
                print(f"Downloaded: {tile_name}")

            except Exception as e:
                # Delete temporary raster files
                try:
                    if raster_path and os.path.exists(raster_path):
                        os.remove(raster_path)
                except Exception as e:
                    print(f"Warning: failed to delete temp raster: {e}")

                for fp in all_tiles:
                    try:
                        if os.path.exists(fp):
                            os.remove(fp)
                    except Exception:
                        pass

                stop_time = time.perf_counter()
                print(f"Function finished in {stop_time - start_time:.1f} s")
                raise DiskSpaceError(f"Failed to download tile {tile_name}: {e}")

        if not all_tiles:
            stop_time = time.perf_counter()
            print(f"Function finished in {stop_time - start_time:.1f} s")
            raise DownloadError("Failed to download DEM tiles.")

        # --- Mosaic tiles together ---
        srcs = [rasterio.open(fp) for fp in all_tiles]
        mosaic, out_trans = merge(srcs)

        meta = srcs[0].meta.copy()
        meta.update({
            "driver": "GTiff",
            "height": mosaic.shape[1],
            "width": mosaic.shape[2],
            "transform": out_trans,
            "compress": "lzw"
        })

        raster_path = os.path.join(dem_folder, f"{base_name}_mosaic_{int(current_radius)}m.tif")

        with rasterio.open(raster_path, "w", **meta) as dest:
            dest.write(mosaic)

        for s in srcs:
            s.close()

        # Read DEM
        with rasterio.open(raster_path) as src:
            dem_array = src.read(1).astype(float)
            transform = src.transform
            extent = src.bounds
            cell_size = transform[0]
            nrows, ncols = dem_array.shape
            dem_crs = src.crs

        transformer = Transformer.from_crs("EPSG:4326", dem_crs, always_xy=True)

        # --- Helper Function ---
        def get_elevation(lon_wgs, lat_wgs):
            """
            Get elevation (in meters) for a geographic coordinate (lon, lat).
            Automatically transforms to the DEM CRS if needed.
            """
            # Project lon/lat to DEM CRS, then map to row/col
            x_proj, y_proj = transformer.transform(lon_wgs, lat_wgs)
            col = int((x_proj - extent.left) / cell_size)
            row = int((extent.top - y_proj) / abs(transform[4]))

            if 0 <= row < nrows and 0 <= col < ncols:
                val = dem_array[row, col]
                return None if np.isnan(val) else float(val)
            return None

        center_elev = get_elevation(lon, lat)

        if center_elev is None:
            stop_time = time.perf_counter()
            print(f"Function finished in {stop_time - start_time:.1f} s")
            raise NullError("Center elevation missing, invalid DEM.")

        # Parameters
        radial_steps = 72
        radial_angles = [i * 360 / radial_steps for i in range(radial_steps)]

        cone_edge_distances, crater_rim_distances = [], []
        point_spacing = 0.5  # meters
        rim_search_limit = max(current_radius / 3, 400)
        edge_found = False

        # --- Radial Sampling ---
        for angle_deg in radial_angles:
            angle_rad = math.radians(angle_deg)
            rim_r = float('nan')
            rim_elev = -float('inf')
            elev_samples = []

            # Flat terrain pre-check
            for r in np.arange(0, current_radius, 100):
                x = lon + (r / 111_000) * math.cos(angle_rad)
                y = lat + (r / 111_000) * math.sin(angle_rad)
                elev = get_elevation(x, y)
                if elev is not None:
                    elev_samples.append(elev)

            if len(elev_samples) > 3 and (max(elev_samples) - min(elev_samples)) < 0.5:
                # Flat terrain: use full radius directly
                cone_edge_distances.append(current_radius)
                crater_rim_distances.append(rim_search_limit / 2)
                if diag:
                    print(f"Radial {angle_deg:>5.1f}° ended at {current_radius:.1f} m (flat terrain)")
                continue

            # Find crater rim
            for r in np.arange(0, rim_search_limit, point_spacing):
                x_deg = lon + (r / 111_000) * math.cos(angle_rad)
                y_deg = lat + (r / 111_000) * math.sin(angle_rad)
                elev = get_elevation(x_deg, y_deg)
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
            while prev_elev > center_elev and r < current_radius:
                r += point_spacing
                x_deg = lon + (r / 111_000) * math.cos(angle_rad)
                y_deg = lat + (r / 111_000) * math.sin(angle_rad)
                elev = get_elevation(x_deg, y_deg)
                if elev is None:
                    continue
                prev_elev = elev

            # Search for min elevation before ground flattening, rising, or slope break
            stop_reason = "max radius reached"
            while r <= current_radius:
                x_deg = lon + (r / 111_000) * math.cos(angle_rad)
                y_deg = lat + (r / 111_000) * math.sin(angle_rad)
                elev = get_elevation(x_deg, y_deg)
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
                        edge_found = True
                        break
                # Flattening
                elif abs(elev - prev_elev) < 0.2:
                    flat_count += 1
                    if flat_count >= slope_points:
                        stop_reason = f"flat terrain ({flat_count} points within 0.25 m)"
                        edge_found = True
                        break
                # Declining
                else:
                    rising_count = 0
                    flat_count = 0

                # Compute short-term slope
                r_back = max(r - slope_points, rim_r)
                x_back = lon + (r_back / 111_000) * math.cos(angle_rad)
                y_back = lat + (r_back / 111_000) * math.sin(angle_rad)
                elev_back = get_elevation(x_back, y_back)
                if elev_back is not None and abs(r - r_back) > 1e-6:
                    slope = (elev - elev_back) / (r - r_back)
                else:
                    slope = 0

                # Stop if slope changes abruptly (terrain inflection)
                if abs(slope - prev_slope) > slope_change_threshold and r > min_r + slope_points:
                    stop_reason = f"slope inflection ({slope} - {prev_slope} > {slope_change_threshold:.3f} m/m"
                    edge_found = True
                    break

                prev_slope = slope
                prev_elev = elev
                r += point_spacing

            # Fallback: use radius
            if min_r >= current_radius:
                min_r = current_radius
                if stop_reason == "max radius reached":
                    stop_reason = "no valid base found — fallback to radius"

            # Use min_r as the detected base
            cone_edge_distances.append(min_r)

            if diag:
                print(f"Radial {angle_deg:>5.1f}° ended at {min_r:.1f} m ({stop_reason})")

        avg_base = np.mean(cone_edge_distances)
        if edge_found and avg_base < current_radius * 0.95:
            print(f"Crater/cone edge detected at ~{avg_base:.0f} m radius.")
            final_found = True
        else:
            print(f"No clear edge found at {current_radius/1000:.1f} km. Expanding search...")
            current_radius += increment

    # --- If not found after full loop ---
    if not final_found:
        end_time = time.perf_counter()
        print(f"Function finished in {end_time - start_time:.1f} s")
        raise NullError("Crater/cone edge not found within max radius.")

    # --- Create Output ---
    smoothed_distances = []
    window = 2
    for i in range(radial_steps):
        neighbors = [cone_edge_distances[(i + j) % radial_steps] for j in range(-window, window + 1)]
        smoothed_distances.append(sum(neighbors) / len(neighbors))

    # Create crater polygon
    crater_points = []
    crater_elevations = []
    for i, angle_deg in enumerate(radial_angles):
        r = crater_rim_distances[i]
        angle_rad = math.radians(angle_deg)
        x_deg = lon + (r / 111_000) * math.cos(angle_rad)
        y_deg = lat + (r / 111_000) * math.sin(angle_rad)
        crater_points.append((x_deg, y_deg))

        elev = get_elevation(x_deg, y_deg)
        if elev is not None:
            crater_elevations.append(elev)

    # Create cone base polygon
    base_points = []
    base_elevations = []
    for i, angle_deg in enumerate(radial_angles):
        r = smoothed_distances[i]
        angle_rad = math.radians(angle_deg)
        x_deg = lon + (r / 111_000) * math.cos(angle_rad)
        y_deg = lat + (r / 111_000) * math.sin(angle_rad)
        base_points.append((x_deg, y_deg))

        elev = get_elevation(x_deg, y_deg)
        if elev is not None:
            base_elevations.append(elev)

    crater_poly = Polygon(crater_points)
    base_poly = Polygon(base_points)

    # --- Cone validation check (rim > center > base) ---
    center_x, center_y = lon, lat
    center_elev = get_elevation(center_x, center_y)

    if crater_elevations and base_elevations and center_elev is not None:
        avg_crater_elev = np.nanmean(crater_elevations)
        avg_base_elev = np.nanmean(base_elevations)

        if not (avg_crater_elev > center_elev > avg_base_elev):
            # Delete temporary raster files
            try:
                if raster_path and os.path.exists(raster_path):
                    os.remove(raster_path)
            except Exception as e:
                print(f"Warning: failed to delete temp raster: {e}")

            for fp in all_tiles:
                try:
                    if os.path.exists(fp):
                        os.remove(fp)
                except Exception:
                    pass

            end_time = time.perf_counter()
            print(f"Cone validation failed after {end_time - start_time:.3f} seconds")
            raise ConeError()

    else:
        raise ValueError(
            "Cone validation failed: unable to compute crater, base, or center elevations"
        )

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
    for tif in all_tiles:
        try:
            os.remove(tif)
        except Exception:
            pass
    if len(all_tiles) > 1:
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
            x2 = lon + (r / 111_000) * math.cos(a)
            y2 = lat + (r / 111_000) * math.sin(a)
            radials.append(LineString([(lon, lat), (x2, y2)]))
        radials_path = os.path.join(polygon_folder, f"{base_name}_radials.shp")
        save_shapefile(radials, radials_path)

    end_time = time.perf_counter()
    print(f"Function finished in {end_time - start_time:.1f} s")
    print(f"Final DEM saved: {clipped_dem_path}")

    return clipped_dem_path, base_path, crater_path


# --- Testing ---
if __name__ == "__main__":
    polygon_folder = r"D:\NASA_Research_Project\Cone_Polygons"
    dem_folder = r"D:\NASA_Research_Project\Cone_DEMS"

    test_cases = [
            {"lat": 35.3641, "lon": -111.5033},  # Sunset Crater
            {"lat": 0, "lon": 0},                # Ocean (Null Error)
            {"lat": 39.7392, "lon": -104.9903},  # Denver, CO (Cone Error)
        ]

    for case in test_cases:
        print("\n--- Testing coordinates:", case["lat"], case["lon"], "---")

        try:
            dem_segment(case["lat"], case["lon"], 1, polygon_folder, dem_folder, diag=True)

        except (NullError, DownloadError, ConeError) as e:
            print(f"Expected error: {e}")
        except Exception:
            print(traceback.format_exc())
