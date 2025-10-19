import arcpy, os, math
from arcpy import env
from arcpy.sa import *


def segment_cone_dem(aprx_path, map_name, service_layer_name, output_folder, lat, lon, radius):
    """
    Extracts a user-defined radius DEM around the given lat/lon point and saves both
    the raster and polygon boundary files.
    
    Parameters
    ----------
    aprx_path : str
        Path to ArcGIS Pro project (.aprx).
    map_name : str
        Name of the map containing the DEM layer.
    service_layer_name : str
        Name of the DEM service layer (e.g., "USGS 3DEP Elevation").
    output_folder : str
        Folder to save output GeoTIFF and boundary shapefile.
    lat, lon : float
        Cone center coordinates (latitude and longitude).
    radius : float
        Radius of the cone (in meters).
    """
    # Print coordinates being segmented
    print(f"Segmenting cone at lat: {lat}, lon: {lon}")

    # Safe filenames
    lat_str = str(lat).replace('.', '_')
    lon_str = str(lon).replace('.', '_')
    base_name = f"{lat_str}x{lon_str}"

    # Create a folder for this cone
    cone_name = f"{base_name}_folder"
    cone_folder = os.path.join(output_folder, cone_name)
    os.makedirs(cone_folder, exist_ok=True)

    env.workspace = "in_memory"
    env.overwriteOutput = True

    # Set up Spatial Analyst
    if arcpy.CheckExtension("Spatial") != "Available":
        raise RuntimeError("Spatial Analyst extension is not available")
    else:
        arcpy.CheckOutExtension("Spatial")

    # Load project and map
    aprx = arcpy.mp.ArcGISProject(aprx_path)
    map_obj = aprx.listMaps(map_name)[0]

    # Get DEM service layer
    layer = next((lyr for lyr in map_obj.listLayers() if lyr.name == service_layer_name), None)
    if not layer:
        raise ValueError(f"Layer '{service_layer_name}' not found")

    # Create point om WGS84
    pt = arcpy.PointGeometry(arcpy.Point(lon, lat), arcpy.SpatialReference(4326))
    
    # Project point to DEM's spatial reference
    dem_sr = arcpy.Describe(layer).spatialReference
    if dem_sr.linearUnitName.lower() in ["meter", "meters"]:
        # DEM is already in meters
        pt_proj = pt.projectAs(dem_sr)
        print("DEM is in meters, no projection needed")
    else:
        # DEM is in degrees → project to UTM zone automatically
        # Use approximate UTM zone from longitude
        utm_zone = int((lon + 180) / 6) + 1
        is_northern = lat >= 0
        utm_epsg = 32600 + utm_zone if is_northern else 32700 + utm_zone
        pt_proj = pt.projectAs(arcpy.SpatialReference(utm_epsg))
        print(f"Projected point to UTM zone {utm_zone} ({'N' if is_northern else 'S'})")

    # Create buffer polygon with user defined radius
    if (radius <= 3500):
        radius += 500  # Add buffer to small radius
    buffer_geom = pt_proj.buffer(radius)

    # Define output paths
    boundary_path = os.path.join(cone_folder, f"{base_name}_boundary.shp")
    raster_path = os.path.join(cone_folder, f"{base_name}_DEM.tif")

    # Save buffer shapefile
    arcpy.CopyFeatures_management(buffer_geom, boundary_path)
    print(f"Saved buffer polygon: {boundary_path}")

    # Extract DEM by mask and save
    dem_clipped = ExtractByMask(layer, buffer_geom)
    dem_clipped.save(raster_path)
    print(f"Saved clipped DEM: {raster_path}")

    # Check in Spatial Analyst license
    arcpy.CheckInExtension("Spatial")
    print("Extraction complete.")

    return raster_path, boundary_path


# Testing
if __name__ == "__main__":
    aprx_path = r"E:\\NASA_Research_Project\ArcGIS_Project\\3DEP_OGC-WCS_Server_Data.aprx"
    map_name = "2D_3DEP_OGC_Map"
    service_layer_name = "3DEPElevation"
    lat, lon = 35.3641, -111.5033 # Sunset Crater Coordinates
    output_folder = r"E:\\NASA_Research_Project\\Cone_DEMS"
    radius = 4000

    segment_cone_dem(aprx_path, map_name, service_layer_name, output_folder, lat, lon, radius)
