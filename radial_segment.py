import arcpy, os, math, numpy, time, traceback
from arcpy import env
from arcpy.sa import *

# --- Custom exceptions ---
class RadiusError(Exception):
    """
    Custom exception for invalid radius input.
    ----------
    Attributes:
        message (str): Explanation of the error.
    """
    def __init__(self, message="Total search distance (radius + buffer) exceeds maximum allowed (4 km)."):
        self.message = message
        super().__init__(self.message)

class MapError(Exception):
    """
    Custom exception for invalid map layer.
    ----------
    Attributes:
        message (str): Explanation of the error.
    """
    def __init__(self, message="Map not found in project."):
        self.message = message
        super().__init__(self.message)

class LayerError(Exception):
    """
    Custom exception for nonexistant map layer.
    ----------
    Attributes:
        message (str): Explanation of the error.
    """
    def __init__(self, message="Specified DEM layer not found."):
        self.message = message
        super().__init__(self.message)

class NullError(Exception):
    """
    Custom exception for null values.
    ----------
    Attributes:
        message (str): Explanation of the error.
    """
    def __init__(self, message="Encountered null values, points in an invalid DEM."):
        self.message = message
        super().__init__(self.message)

class ConeError(Exception):
    """
    Custom exception for invalid cone geometry.
    ----------
    Attributes:
        message (str): Explanation of the error.
    """
    def __init__(self, message="Cone validation failed. Expected: crater > center > base."):
        self.message = message
        super().__init__(self.message)

# --- Main Function ---
def radial_segment(aprx_path, map_name, service_layer_name, polygon_folder, dem_folder, lat, lon, 
                     radius = 4000, radial_steps=72, outward_buffer=0, diag = False):
    """
    Extracts the DEM of a cone using radial slope sampling and adaptive edge detection,
    adds an outward buffer, and saves the raster, crater, and base boundaries.

    Max cone diameter of 8 km (4000 m radius).
    
    Parameters
    ----------
    aprx_path : str
        Path to ArcGIS Pro project (.aprx).
    map_name : str
        Name of the map containing the DEM layer.
    service_layer_name : str
        Name of the DEM service layer.
    polygon_folder : str
        Folder where boundary and crater polygons will be saved.
    dem_folder : str
        Folder where output DEMs will be saved.
    lat, lon : float
        Cone center coordinates (latitude and longitude).
    radius : float
        Radius of the cone (in meters).
        Default: 4000 m (max allowed)
    radial_steps: int
        Number of radial lines to sample.
        Default: 72 (every 5 degrees)
    outward_buffer : float
        Additional buffer (in meters) to add beyond detected edge.
        Default: 0 (no buffer)
    diag : bool
        If True, prints diagnostic info and saves radial lines shapefile.
        Default: False
    """
    # --- Set Up ---
    start_time = time.perf_counter()

    # Preliminary checks
    if radius + outward_buffer > 4000:

        end_time = time.perf_counter()
        print(f"Function took {end_time - start_time:.3f} seconds")

        raise RadiusError

    print(f"Segmenting cone at lat: {lat}, lon: {lon}")

    # Safe filenames
    lat_str = str(lat).replace('.', '_')
    lon_str = str(lon).replace('.', '_')
    base_name = f"{lat_str}x{lon_str}"

    env.workspace = "in_memory"
    env.overwriteOutput = True

    # Set up Spatial Analyst
    if arcpy.CheckExtension("Spatial") != "Available":
        raise RuntimeError("Spatial Analyst extension is not available")
    else:
        arcpy.CheckOutExtension("Spatial")

    # Load project and map
    aprx = arcpy.mp.ArcGISProject(aprx_path)
    map_obj_list = aprx.listMaps(map_name)
    if not map_obj_list:
        arcpy.CheckInExtension("Spatial")
        arcpy.ClearEnvironment("workspace")
        arcpy.ClearWorkspaceCache_management()

        end_time = time.perf_counter()
        print(f"Function took {end_time - start_time:.3f} seconds")

        raise MapError(f"Map '{map_name}' not found in {aprx_path}")
    map_obj = map_obj_list[0]

    # Get DEM service layer
    layer = next((lyr for lyr in map_obj.listLayers() if lyr.name == service_layer_name), None)
    if not layer:
        arcpy.CheckInExtension("Spatial")
        arcpy.ClearEnvironment("workspace")
        arcpy.ClearWorkspaceCache_management()

        end_time = time.perf_counter()
        print(f"Function took {end_time - start_time:.3f} seconds")

        raise LayerError
    
    dem_sr = arcpy.Describe(layer).spatialReference
    
    # Project point to DEM's spatial reference
    pt = arcpy.PointGeometry(arcpy.Point(lon, lat), arcpy.SpatialReference(4326))
    pt_proj = pt.projectAs(dem_sr)
    center_x, center_y = pt_proj.centroid.X, pt_proj.centroid.Y
    print(f"Projected point to layer spatial reference: {dem_sr.name}")

    # --- Clip Local DEM ---
    print("Extracting local DEM subset...")

    buffer_geom = pt_proj.buffer(radius + outward_buffer)
    dem_clip = ExtractByMask(layer, buffer_geom)

    # Save raster temporarily to get properties
    tmp_path = os.path.join(dem_folder, "tmp_dem.tif")
    dem_clip.save(tmp_path)
    dem_clip_raster = arcpy.Raster(tmp_path)

    # Convert to NumPy array
    dem_array = arcpy.RasterToNumPyArray(dem_clip_raster, nodata_to_value=numpy.nan)
    extent = dem_clip_raster.extent
    cell_size = dem_clip_raster.meanCellWidth
    nrows, ncols = dem_array.shape

    print(f"Local DEM size: {ncols} x {nrows}, cell: {cell_size:.2f} m")

    # Check if all points are in a valid DEM
    if numpy.all(numpy.isnan(dem_array)):
        # Clean up
        try:
            arcpy.management.Delete(dem_clip)
        except Exception as e:
            pass
        arcpy.CheckInExtension("Spatial")
        arcpy.ClearEnvironment("workspace")
        arcpy.ClearWorkspaceCache_management()

        end_time = time.perf_counter()
        print(f"Function took {end_time - start_time:.3f} seconds")

        raise NullError

    # --- Helper Function ---
    def get_elevation(x, y):
        col = int((x - extent.XMin) / cell_size)
        row = int((extent.YMax - y) / cell_size)
        if 0 <= row < nrows and 0 <= col < ncols:
            val = dem_array[row, col]
            return None if numpy.isnan(val) else float(val)
        return None
    
    # --- Parameters ---
    radial_angles = [i * 360 / radial_steps for i in range(radial_steps)]
    cone_edge_distances = []
    crater_rim_distances = []
    
    rim_search_limit = int(max(radius / 3, 400)) # meters to search for crater rim
    point_spacing = 0.5  # meters along each radial

    # --- Radial Sampling ---
    for angle_deg in radial_angles:
        angle_rad = math.radians(angle_deg)
        rim_elev = -float('inf')

        # Flat terrain pre-check
        elev_samples = []
        for r in numpy.arange(0, radius, 100):
            x = center_x + r * math.cos(angle_rad)
            y = center_y + r * math.sin(angle_rad)
            elev = get_elevation(x, y)
            if elev is not None:
                elev_samples.append(elev)
        if len(elev_samples) > 3 and (max(elev_samples) - min(elev_samples)) < 0.5:
            # Flat terrain: use full radius directly
            cone_edge_distances.append(radius)
            crater_rim_distances.append(rim_search_limit / 2)
            continue

        # Find crater rim
        for r in numpy.arange(0, rim_search_limit, point_spacing):
            x = center_x + r * math.cos(angle_rad)
            y = center_y + r * math.sin(angle_rad)
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
            x = center_x + r * math.cos(angle_rad)
            y = center_y + r * math.sin(angle_rad)
            elev = get_elevation(x, y)
            if elev is None:
                continue
            prev_elev = elev

        # Search for min elevation before ground flattening, rising, or slope break
        stop_reason = "max radius reached"
        while r <= radius:
            x = center_x + r * math.cos(angle_rad)
            y = center_y + r * math.sin(angle_rad)
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
            x_back = center_x + r_back * math.cos(angle_rad)
            y_back = center_y + r_back * math.sin(angle_rad)
            elev_back = get_elevation(x_back, y_back)
            if elev_back is not None and abs(r - r_back) > 1e-6:
                slope = (elev - elev_back) / (r - r_back)
            else:
                slope = 0

            # Stop if slope changes abruptly (terrain inflection)
            if abs(slope - prev_slope) > slope_change_threshold and r > min_r + slope_points:
                stop_reason = f"slope inflection ({slope} - {prev_slope} = {slope-prev_slope}) > {slope_change_threshold:.3f} m/m"
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

    # --- Clean Output ---
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
        x = center_x + r * math.cos(angle_rad)
        y = center_y + r * math.sin(angle_rad)
        crater_points.append(arcpy.Point(x, y))
        
        elev = get_elevation(x, y)
        if elev is not None:
            crater_elevations.append(elev)

    crater_geom = arcpy.Polygon(arcpy.Array(crater_points), dem_sr)

    # Create cone base polygon
    base_points = []
    base_elevations = []
    for i, angle_deg in enumerate(radial_angles):
        r = smoothed_distances[i]
        angle_rad = math.radians(angle_deg)
        x = center_x + r * math.cos(angle_rad)
        y = center_y + r * math.sin(angle_rad)
        base_points.append(arcpy.Point(x, y))

        elev = get_elevation(x, y)
        if elev is not None:
            base_elevations.append(elev)

    polygon_geom = arcpy.Polygon(arcpy.Array(base_points), dem_sr)

    # Optional: create diagnostic radials
    if diag:
        radial_lines = []

        for i, angle_deg in enumerate(radial_angles):
            angle_rad = math.radians(angle_deg)

            # Start (center) and end (base)
            start_x, start_y = center_x, center_y
            end_r = cone_edge_distances[i]
            end_x = center_x + end_r * math.cos(angle_rad)
            end_y = center_y + end_r * math.sin(angle_rad)

            # Create polyline geometry
            radial_line = arcpy.Polyline(
                arcpy.Array([arcpy.Point(start_x, start_y), arcpy.Point(end_x, end_y)]),
                dem_sr
            )

            # Store both geometry and its azimuth
            radial_lines.append((radial_line, angle_deg))

        # Save radial lines to shapefile
        radials_fc = os.path.join(polygon_folder, f"{base_name}_radials.shp")
        radials_mem = arcpy.management.CreateFeatureclass(
            "in_memory", "radials", "POLYLINE", spatial_reference=dem_sr
        )

        # Add field for degree value
        arcpy.management.AddField(radials_mem, "ANGLE_DEG", "DOUBLE")

        # Insert geometry and degree attribute
        with arcpy.da.InsertCursor(radials_mem, ["SHAPE@", "ANGLE_DEG"]) as cur:
            for geom, angle_deg in radial_lines:  # now valid tuple unpack
                cur.insertRow([geom, angle_deg])
    
    # Cone validation check (rim > center > base)
    center_elev = get_elevation(center_x, center_y)

    if crater_elevations and base_elevations and center_elev is not None:
        avg_crater_elev = numpy.nanmean(crater_elevations)
        avg_base_elev = numpy.nanmean(base_elevations)
        
        if not (avg_crater_elev > center_elev > avg_base_elev):
            # Clean up temporary raster
            try:
                arcpy.management.Delete(dem_clip)
                print("Temporary files deleted.")
            except Exception as e:
                print(f"Warning: failed to delete some temp data: {e}")

            arcpy.CheckInExtension("Spatial")
            arcpy.ClearEnvironment("workspace")
            arcpy.ClearWorkspaceCache_management()

            end_time = time.perf_counter()
            print(f"Function took {end_time - start_time:.3f} seconds")
            
            raise ConeError
    else:
        raise ValueError("Cone validation failed: unable to compute elevations for crater, base, or center point")

    # --- Save Outputs ---
    crater_path = os.path.join(polygon_folder, f"{base_name}_crater.shp")
    boundary_path = os.path.join(polygon_folder, f"{base_name}_boundary.shp")
    raster_path = os.path.join(dem_folder, f"{base_name}_DEM.tif")

    arcpy.CopyFeatures_management(crater_geom, crater_path)
    arcpy.CopyFeatures_management(polygon_geom, boundary_path)
    
    if diag:
        arcpy.management.CopyFeatures(radials_mem, radials_fc)
        print(f"Saved diagnostic radial lines: {radials_fc}")

        arcpy.management.Delete(radials_mem)

    # --- Clip Final DEM ---
    buffered_polygon = polygon_geom.buffer(outward_buffer)
    dem_final = ExtractByMask(layer, buffered_polygon)
    dem_final.save(raster_path)

    # --- Clean Up ---
    try:
        arcpy.management.Delete(dem_clip)
    except Exception as e:
        print(f"Warning: failed to delete some temp data: {e}")

    arcpy.CheckInExtension("Spatial")

    print("Extraction complete.")

    end_time = time.perf_counter()
    print(f"Function took {end_time - start_time:.3f} seconds")

    return raster_path, boundary_path, crater_path

# --- Testing ---
if __name__ == "__main__":
    aprx_path = r"E:\\NASA_Research_Project\ArcGIS_Project\\3DEP_OGC-WCS_Server_Data.aprx"
    map_name = "2D_3DEP_OGC_Map"
    service_layer_name = "3DEPElevation"
    polygon_folder = r"E:\\NASA_Research_Project\\Cone_Polygons"
    dem_folder = r"E:\\NASA_Research_Project\\Cone_DEMS"

    test_cases = [
            {"lat": 35.3641, "lon": -111.5033, "radius": 4000},  # Sunset Crater
            {"lat": 0, "lon": 0, "radius": 4000},                # Ocean
            {"lat": 39.7392, "lon": -104.9903, "radius": 4000},  # Denver, CO
            {"lat": 35.3641, "lon": -111.5033, "radius": 4500}   # Too large radius
        ]

    for case in test_cases:
        print("\n--- Testing coordinates:", case["lat"], case["lon"], "Radius:", case["radius"], "---")
        try:
            radial_segment(aprx_path, map_name, service_layer_name,
                             polygon_folder, dem_folder, case["lat"], 
                             case["lon"], radius=case["radius"], 
                             radial_steps=72, outward_buffer=0, diag=False)
            
        except (RadiusError, LayerError, NullError, ConeError) as e:
            print(f"Expected error: {e}")
        except Exception as e:
            print(traceback.format_exc())
