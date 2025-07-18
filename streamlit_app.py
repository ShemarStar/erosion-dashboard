import streamlit as st
import osmnx as ox
import geopandas as gpd
import folium
from streamlit_folium import st_folium
import pandas as pd
import numpy as np
from shapely.geometry import LineString
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

st.title("Dynamic Erosion Control Monitoring Dashboard")
st.write("Identify high-risk roads near water bodies for construction compliance in Sachsen, Germany (or any region). Data fetched live from OpenStreetMap.")

# User inputs
region = st.text_input("Enter Region (e.g., 'Dresden, Sachsen, Germany')", value="Dresden, Sachsen, Germany")
buffer_distance = st.slider("Water Buffer Distance (meters) for Risk Assessment", 10, 200, 50)
fetch_data = st.button("Fetch and Analyze Data")

# Persist data in session state
if 'filtered_gdf' not in st.session_state:
    st.session_state.filtered_gdf = pd.DataFrame()

if fetch_data:
    try:
        st.write("Fetching roads and water bodies...")

        # Fetch road network
        graph = ox.graph_from_place(region, network_type="all")
        roads_gdf = ox.graph_to_gdfs(graph, nodes=False, edges=True)
        roads_gdf = roads_gdf.to_crs("EPSG:32633")  # Projected CRS for accurate measurements (UTM 33N for Sachsen)
        roads_gdf["length_m"] = roads_gdf.length
        roads_gdf["fclass"] = roads_gdf["highway"]

        # Fetch water bodies
        water_tags = {"natural": "water", "waterway": True}
        water_gdf = ox.features_from_place(region, tags=water_tags)
        water_gdf = water_gdf.to_crs("EPSG:32633")

        # Buffer water
        water_buffer = water_gdf.buffer(buffer_distance)

        # Find roads near water
        roads_near_water = gpd.sjoin(roads_gdf, gpd.GeoDataFrame(geometry=water_buffer, crs=water_gdf.crs), how="inner", predicate="intersects")

        # Flatten osmid if it's a list (before dedup)
        roads_near_water["osmid"] = roads_near_water["osmid"].apply(lambda x: x[0] if isinstance(x, list) else x)

        # Dedup
        roads_near_water = roads_near_water.drop_duplicates(subset="osmid")

        # Simulate erosion risk
        def calculate_risk(row):
            if row["length_m"] > 100 and row["fclass"] in ["path", "track", "footway"]:
                return "High"
            else:
                return "Low"

        roads_near_water["predicted_risk"] = roads_near_water.apply(calculate_risk, axis=1)

        # Back to lat/lon for mapping
        roads_near_water = roads_near_water.to_crs("EPSG:4326")

        # Limit to 1000
        st.session_state.filtered_gdf = roads_near_water.head(1000)
        st.write(f"Analyzed {len(st.session_state.filtered_gdf)} roads near water.")

    except Exception as e:
        st.error(f"Error fetching/processing data: {e}")
        logger.error(e)

# Use persisted data
filtered_gdf = st.session_state.filtered_gdf

# Filters
if not filtered_gdf.empty:
    st.sidebar.header("Filter Options")
    # Flatten lists in fclass to fix unhashable error
    filtered_gdf["fclass"] = filtered_gdf["fclass"].apply(lambda x: x[0] if isinstance(x, list) else x)
    fclass_filter = st.sidebar.multiselect("Select Road Types", options=filtered_gdf["fclass"].unique(), default=filtered_gdf["fclass"].unique())
    risk_filter = st.sidebar.multiselect("Select Risk Levels", options=filtered_gdf["predicted_risk"].unique(), default=filtered_gdf["predicted_risk"].unique())

    filtered_gdf = filtered_gdf[filtered_gdf["fclass"].isin(fclass_filter) & filtered_gdf["predicted_risk"].isin(risk_filter)]

    st.subheader("Filtered Roads Table")
    # Reset index to make osmid a column if needed (OSMnx uses multi-index u/v/key for edges)
    filtered_gdf = filtered_gdf.reset_index()
    # Use 'osmid' as osm_id (standard OSM edge ID)
    display_df = filtered_gdf[["osmid", "fclass", "length_m", "predicted_risk"]].rename(columns={"osmid": "osm_id"})
    st.dataframe(display_df)

    # Download
    csv = display_df.to_csv(index=False).encode("utf-8")
    st.download_button("Download Filtered Data", csv, "filtered_roads.csv", "text/csv")

    # Map
    st.subheader("Map of Roads Near Water (Colored by Risk)")
    if not filtered_gdf.empty:
        m = folium.Map(location=[51.053, 13.738], zoom_start=12)

        def style_function(feature):
            risk = feature["properties"]["predicted_risk"]
            color = "red" if risk == "High" else "green"
            return {"color": color, "weight": 2}

        folium.GeoJson(filtered_gdf.to_json(), style_function=style_function, tooltip=folium.GeoJsonTooltip(fields=["fclass", "predicted_risk", "length_m"])).add_to(m)

        st_folium(m, width=700, height=500)

# Fallback for custom data - Use local files if available
st.sidebar.header("Use Local Data (Optional)")
use_local = st.sidebar.checkbox("Load from local CSV and Shapefile", value=False)
if use_local:
    try:
        df = pd.read_csv("erosion_predictions.csv")
        gdf = gpd.read_file("sachsen_roads_near_water_deduped.shp")
        gdf["osm_id"] = gdf["osm_id"].astype("int64")
        df["osm_id"] = df["osm_id"].astype("int64")
        gdf = gdf.merge(df[["osm_id", "predicted_risk"]], on="osm_id", how="left")
        gdf["predicted_risk"] = gdf["predicted_risk"].fillna("Unknown")
        # Rename to match OSM fetch (for consistency)
        gdf = gdf.rename(columns={"osm_id": "osmid"})
        st.session_state.filtered_gdf = gdf.head(1000)
        st.write("Loaded local data successfully.")
    except Exception as e:
        st.error(f"Error loading local data: {e}")

st.write("Note: Risk is simulated; integrate your ML model for accurate predictions.")
