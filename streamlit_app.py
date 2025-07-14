import streamlit as st
import osmnx as ox
import geopandas as gpd
import folium
from streamlit_folium import st_folium
import pandas as pd
import numpy as np
from shapely.geometry import LineString
import logging
import traceback

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

st.title("Dynamic Erosion Control Monitoring Dashboard")
st.write("Identify high-risk roads near water bodies for construction compliance in Sachsen, Germany (or any region). Data fetched live from OpenStreetMap.")

# User inputs
region = st.text_input("Enter Region (e.g., 'Dresden, Sachsen, Germany')", value="Dresden, Sachsen, Germany")  # Smaller default
buffer_distance = st.slider("Water Buffer Distance (meters) for Risk Assessment", 10, 200, 50)
fetch_data = st.button("Fetch and Analyze Data")

filtered_gdf = pd.DataFrame()

if fetch_data:
    try:
        st.write("Fetching roads and water bodies...")
        
        # Fetch road network
        graph = ox.graph_from_place(region, network_type="all")
        roads_gdf = ox.graph_to_gdfs(graph, nodes=False, edges=True)
        roads_gdf = roads_gdf.to_crs("EPSG:4326")
        roads_gdf["length_m"] = roads_gdf.length * 111320  # Approx meters
        roads_gdf["fclass"] = roads_gdf["highway"]
        
        # Fetch water bodies
        water_tags = {"natural": "water", "waterway": True}
        water_gdf = ox.geometries_from_place(region, tags=water_tags)
        water_gdf = water_gdf.to_crs("EPSG:4326")
        
        # Buffer water
        water_buffer = water_gdf.buffer(buffer_distance / 111320)
        
        # Find roads near water
        roads_near_water = gpd.sjoin(roads_gdf, gpd.GeoDataFrame(geometry=water_buffer), how="inner", predicate="intersects")
        roads_near_water = roads_near_water.drop_duplicates(subset="osmid")
        
        # Simulate erosion risk
        def calculate_risk(row):
            if row["length_m"] > 100 and row["fclass"] in ["path", "track", "footway"]:
                return "High"
            else:
                return "Low"
        
        roads_near_water["predicted_risk"] = roads_near_water.apply(calculate_risk, axis=1)
        
        # Limit to 1000
        filtered_gdf = roads_near_water.head(1000)
        st.write(f"Analyzed {len(filtered_gdf)} roads near water.")
        
    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")
        st.error(traceback.format_exc())  # Full traceback for debugging
        logger.error(e)

# Rest of the code (filters, map, local data) remains the same
# ...
