import streamlit as st
import osmnx as ox
import geopandas as gpd
import folium
from streamlit_folium import st_folium
import pandas as pd
import numpy as np
import requests
from fpdf import FPDF
import ee
import logging
import io

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Authenticate Earth Engine
try:
    ee.Initialize()
except Exception as e:
    st.error(f"Earth Engine init error: {e}. Run 'earthengine authenticate' in terminal or set a manual LS factor.")

st.title("Dynamic Erosion Control Monitoring Dashboard")
st.write("""Identify high-risk roads near water bodies for construction compliance in Sachsen, Germany (or any region). 
Data fetched live from OpenStreetMap. Now with RUSLE modeling (GIS slope), weather forecasts, mobile inspections, and PDF reports.""")

region = st.text_input("Enter Region (e.g., 'Dresden, Sachsen, Germany')", value="Dresden, Sachsen, Germany")
buffer_distance = st.slider("Water Buffer Distance (meters)", 10, 200, 50)
fetch_data = st.button("Fetch and Analyze Data")

weather_api_key = st.sidebar.text_input("OpenWeather API Key", type="password")

if 'filtered_gdf' not in st.session_state:
    st.session_state.filtered_gdf = pd.DataFrame()
if 'inspection_notes' not in st.session_state:
    st.session_state.inspection_notes = ""
if 'inspection_photo' not in st.session_state:
    st.session_state.inspection_photo = None
if 'rusle_factors' not in st.session_state:
    st.session_state.rusle_factors = {'r': 100.0, 'k': 0.3, 'ls': 1.5, 'c': 0.2, 'p': 1.0, 'mean_slope': 1.5}

# RUSLE sidebar (persistent, outside fetch_data)
st.sidebar.header("RUSLE Factors")
st.session_state.rusle_factors['r'] = st.sidebar.number_input("R (Rainfall Erosivity)", 0.0, 200.0, st.session_state.rusle_factors['r'])
st.session_state.rusle_factors['k'] = st.sidebar.number_input("K (Soil Erodibility)", 0.0, 1.0, st.session_state.rusle_factors['k'])
st.session_state.rusle_factors['ls'] = st.sidebar.number_input("LS (Slope Length/Steepness)", 0.0, 10.0, st.session_state.rusle_factors['ls'])
st.session_state.rusle_factors['c'] = st.sidebar.number_input("C (Cover Management)", 0.0, 1.0, st.session_state.rusle_factors['c'])
st.session_state.rusle_factors['p'] = st.sidebar.number_input("P (Support Practice)", 0.0, 1.0, st.session_state.rusle_factors['p'])
st.sidebar.write(f"GIS Avg Slope: {st.session_state.rusle_factors['mean_slope']:.2f}%")

if fetch_data:
    try:
        st.write("Fetching roads and water bodies...")
        graph = ox.graph_from_place(region, network_type="all")
        roads_gdf = ox.graph_to_gdfs(graph, nodes=False, edges=True)
        roads_gdf = roads_gdf.to_crs("EPSG:32633")
        roads_gdf["length_m"] = roads_gdf.length
        roads_gdf["fclass"] = roads_gdf["highway"]

        water_tags = {"natural": "water", "waterway": True}
        water_gdf = ox.features_from_place(region, tags=water_tags)
        water_gdf = water_gdf.to_crs("EPSG:32633")
        water_buffer = water_gdf.buffer(buffer_distance)

        roads_near_water = gpd.sjoin(roads_gdf, gpd.GeoDataFrame(geometry=water_buffer, crs=water_gdf.crs), how="inner", predicate="intersects")
        roads_near_water["osmid"] = roads_near_water["osmid"].apply(lambda x: x[0] if isinstance(x, list) else x)
        roads_near_water = roads_near_water.drop_duplicates(subset="osmid")

        # GIS Slope from Earth Engine
        try:
            place = ox.geocode_to_gdf(region)
            bbox = place.total_bounds
            geometry = ee.Geometry.Rectangle([bbox[0], bbox[1], bbox[2], bbox[3]])
            dataset = ee.Image('USGS/SRTMGL1_003').clip(geometry)
            slope = ee.Terrain.slope(dataset.select('elevation'))
            mean_slope = slope.reduceRegion(ee.Reducer.mean(), geometry, 30).get('slope').getInfo() or 1.5
            st.session_state.rusle_factors['mean_slope'] = mean_slope
            st.session_state.rusle_factors['ls'] = float(0.065 + 0.045 * mean_slope + 0.0065 * mean_slope**2)
        except Exception as e:
            st.warning(f"GIS slope failed: {e}. Using default LS factor.")
            st.session_state.rusle_factors['mean_slope'] = 1.5
            st.session_state.rusle_factors['ls'] = 1.5

        def calculate_rusle(row):
            a = (st.session_state.rusle_factors['r'] * st.session_state.rusle_factors['k'] * 
                 st.session_state.rusle_factors['ls'] * st.session_state.rusle_factors['c'] * 
                 st.session_state.rusle_factors['p'])
            return "High" if a > 5 or row["length_m"] > 100 else "Low", a

        roads_near_water.loc[:, ['predicted_risk', 'rusle_a']] = roads_near_water.apply(calculate_rusle, axis=1, result_type='expand')
        roads_near_water = roads_near_water.to_crs("EPSG:4326")
        st.session_state.filtered_gdf = roads_near_water.head(1000)
        st.write(f"Analyzed {len(st.session_state.filtered_gdf)} roads.")

    except Exception as e:
        st.error(f"Error: {e}")
        logger.error(e)

filtered_gdf = st.session_state.filtered_gdf

if not filtered_gdf.empty:
    st.sidebar.header("Filter Options")
    # Remove NaN from unique values for multiselect
    valid_risks = [x for x in filtered_gdf["predicted_risk"].unique() if pd.notna(x)]
    fclass_filter = st.sidebar.multiselect("Road Types", filtered_gdf["fclass"].unique(), default=filtered_gdf["fclass"].unique())
    risk_filter = st.sidebar.multiselect("Risk Levels", valid_risks, default=valid_risks)
    filtered_gdf = filtered_gdf[filtered_gdf["fclass"].isin(fclass_filter) & filtered_gdf["predicted_risk"].isin(risk_filter)]

    if "rusle_a" not in filtered_gdf.columns:
        filtered_gdf["rusle_a"] = np.nan

    st.subheader("Filtered Roads")
    filtered_gdf = filtered_gdf.reset_index()
    available_columns = [col for col in ["osmid", "fclass", "length_m", "predicted_risk", "rusle_a"] if col in filtered_gdf.columns]
    rename_dict = {"osmid": "osm_id", "rusle_a": "RUSLE Soil Loss (tons/acre/yr)"}
    display_df = filtered_gdf[available_columns].rename(columns={k: v for k, v in rename_dict.items() if k in available_columns})
    st.dataframe(display_df)

    csv = display_df.to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV", csv, "filtered_roads.csv", "text/csv")

    if st.button("Generate PDF Report"):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt="Erosion Control Report", ln=1, align="C")
        pdf.cell(200, 10, txt=f"Region: {region}", ln=1)
        pdf.cell(200, 10, txt=f"Analyzed Roads: {len(display_df)}", ln=1)
        col_widths = [40] * len(display_df.columns)
        pdf.set_font("Arial", "B", 10)
        for i, header in enumerate(display_df.columns):
            pdf.cell(col_widths[i], 10, header, border=1)
        pdf.ln()
        pdf.set_font("Arial", size=10)
        for _, row in display_df.iterrows():
            for i, value in enumerate(row):
                pdf.cell(col_widths[i], 10, str(value), border=1)
            pdf.ln()
        pdf_output = pdf.output(dest="S").encode("latin1")
        st.download_button("Download PDF", pdf_output, "erosion_report.pdf", "application/pdf")

    st.subheader("Map of Roads")
    if not filtered_gdf.empty:
        m = folium.Map(location=[51.053, 13.738], zoom_start=12)
        def style_function(feature):
            risk = feature["properties"]["predicted_risk"]
            return {"color": "red" if risk == "High" else "green", "weight": 2}
        tooltip_fields = ["fclass", "predicted_risk", "length_m"]
        if "rusle_a" in filtered_gdf.columns:
            tooltip_fields.append("rusle_a")
        folium.GeoJson(filtered_gdf.to_json(), style_function=style_function, tooltip=folium.GeoJsonTooltip(fields=tooltip_fields)).add_to(m)
        st_folium(m, width=700, height=500)

st.sidebar.header("Weather Forecast")
if weather_api_key and fetch_data:
    try:
        place = ox.geocode_to_gdf(region)
        lat, lon = place.centroid.y[0], place.centroid.x[0]
        url = f"https://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&appid={weather_api_key}&units=metric"
        response = requests.get(url).json()
        if response["cod"] == "200":
            rain = response["list"][0].get("rain", {}).get("3h", 0)
            st.sidebar.write(f"Rain (3h): {rain} mm")
            if rain > 5:
                st.sidebar.warning("High erosion risk!")
        else:
            st.sidebar.error("Invalid API key.")
    except Exception as e:
        st.sidebar.error(f"Weather error: {e}")

st.sidebar.header("Mobile Inspection")
st.session_state.inspection_notes = st.sidebar.text_area("Notes", st.session_state.inspection_notes)
st.session_state.inspection_photo = st.sidebar.camera_input("Photo")
if st.session_state.inspection_photo:
    st.sidebar.image(st.session_state.inspection_photo, caption="Site Photo")

st.sidebar.header("Local Data")
use_local = st.sidebar.checkbox("Load Local CSV/Shapefile", False)
if use_local:
    try:
        df = pd.read_csv("erosion_predictions.csv")
        gdf = gpd.read_file("sachsen_roads_near_water_deduped.shp")
        gdf["osm_id"] = gdf["osm_id"].astype("int64")
        df["osm_id"] = df["osm_id"].astype("int64")
        gdf = gdf.merge(df[["osm_id", "predicted_risk"]], on="osm_id", how="left")
        if "rusle_a" not in gdf.columns:
            gdf["rusle_a"] = np.nan
        st.session_state.filtered_gdf = gdf.head(1000)
        st.write("Local data loaded.")
    except Exception as e:
        st.error(f"Local data error: {e}")

st.write("Note: RUSLE uses GIS slope; add ML for accuracy. Weather is forecast-based.")
