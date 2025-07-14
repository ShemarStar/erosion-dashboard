import geopandas as gpd
from fiona import Env  # For setting GDAL config

with Env(SHAPE_RESTORE_SHX="YES"):
    # Load the existing Shapefile (will restore .shx)
    gdf = gpd.read_file("sachsen_roads_near_water_deduped.shp")

# Set CRS (use EPSG:4326 for OSM data; change if different)
gdf.crs = "EPSG:4326"

# Save to regenerate companions (.shx, .prj)
gdf.to_file("sachsen_roads_near_water_deduped_fixed.shp")

print("Regenerated Shapefile saved as sachsen_roads_near_water_deduped_fixed.shp with all files.")
