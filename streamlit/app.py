
import streamlit as st
import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import tempfile
import os
import requests
import unidecode
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import zscore

# Cache data loading
import pandas as pd
import streamlit as st
import requests

# Function to download file from Google Drive
def download_file_from_google_drive(url, destination_path):
    file_id = url.split('/')[-2]  # Extract file ID from the URL
    download_url = f"https://drive.google.com/uc?export=download&id={file_id}"
    response = requests.get(download_url, stream=True)
    
    if response.status_code == 200:
        with open(destination_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=128):
                f.write(chunk)
        return destination_path
    else:
        st.error("Failed to download the file.")
        return None

# Streamlit cache function for loading the dataset
@st.cache_data
def load_data(file_path):
    try:
        return pd.read_csv(file_path, sep=";")
    except FileNotFoundError:
        st.error("File not found. Please upload the correct dataset.")
        return None

# URL of the file from Google Drive
file_url = "https://drive.google.com/drive/folders/1RwomGz_w1xV-OM6U8ePraUCuF5pDxexo"

# Download the file to a local destination
destination = "data.csv"  # Change this to the desired destination file name
download_file_from_google_drive(file_url, destination)

# Load the data using the local file
data = load_data(destination)

if data is not None:
    st.write(data)

@st.cache_data
def download_region_geojson():
    url = "https://raw.githubusercontent.com/gregoiredavid/france-geojson/master/regions-version-simplifiee.geojson"
    response = requests.get(url)
    temp_file = os.path.join(tempfile.gettempdir(), "regions.geojson")
    with open(temp_file, 'wb') as f:
        f.write(response.content)
    return temp_file

# Column validation helper
@st.cache_data
def validate_columns(data, required_columns):
    missing_cols = [col for col in required_columns if col not in data.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in DataFrame: {missing_cols}")

# Energy map visualization
@st.cache_data
def create_energy_production_map(data):
    energy_cols = ['Thermique (MW)', 'Nucléaire (MW)', 'Eolien (MW)',
                   'Solaire (MW)', 'Hydraulique (MW)', 'Pompage (MW)',
                   'Bioénergies (MW)']

    validate_columns(data, energy_cols + ['Région'])

    for col in energy_cols:
        data[col] = pd.to_numeric(data[col], errors='coerce').fillna(0)

    data['Production (MW)'] = data[energy_cols].sum(axis=1)
    renewable_cols = ['Eolien (MW)', 'Solaire (MW)', 'Hydraulique (MW)', 'Bioénergies (MW)']
    data['Renouvelable (MW)'] = data[renewable_cols].sum(axis=1)

    production_by_region = data.groupby('Région')[['Production (MW)', 'Renouvelable (MW)']].sum()
    production_by_region['Pourcentage Renouvelable'] = (
        production_by_region['Renouvelable (MW)'] / production_by_region['Production (MW)'].replace(0, 1) * 100
    )

    regions_file = download_region_geojson()
    regions_gdata = gpd.read_file(regions_file)

    regions_gdata['nom'] = regions_gdata['nom'].str.upper().apply(lambda x: unidecode.unidecode(x))
    production_by_region.index = [unidecode.unidecode(x.upper()) for x in production_by_region.index]

    regions_gdata = regions_gdata.merge(production_by_region, left_on='nom', right_index=True, how='left')

    fig, ax = plt.subplots(figsize=(15, 10))
    regions_gdata.plot(
        column='Production (MW)',
        ax=ax,
        legend=True,
        legend_kwds={'label': 'Production Totale (MW)'},
        cmap='YlOrRd',
        missing_kwds={'color': 'lightgrey'}
    )

    for idx, row in regions_gdata.iterrows():
        if row['Production (MW)'] > 0:
            centroid = row.geometry.centroid
            text = f"{row['nom']}\n{row['Production (MW)']/1000:,.1f} GW\n{row['Pourcentage Renouvelable']:.1f}% Ren."
            ax.annotate(
                text,
                xy=(centroid.x, centroid.y),
                horizontalalignment='center',
                verticalalignment='center',
                fontsize=8,
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1)
            )

    ax.set_title("Production d'Électricité par Région", fontsize=14, pad=20)
    ax.axis('off')
    plt.tight_layout()
    return fig

# Time series visualization
@st.cache_data
def create_time_series_plot(data):
    data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
    energy_cols = ['Thermique (MW)', 'Nucléaire (MW)', 'Eolien (MW)',
                   'Solaire (MW)', 'Hydraulique (MW)', 'Pompage (MW)',
                   'Bioénergies (MW)']

    validate_columns(data, ['Date'] + energy_cols)

    for col in energy_cols:
        data[col] = pd.to_numeric(data[col], errors='coerce').fillna(0)

    data['Production (MW)'] = data[energy_cols].sum(axis=1)

    if 'Consommation (MW)' in data.columns:
        data['Consommation (MW)'] = pd.to_numeric(data['Consommation (MW)'], errors='coerce').fillna(0)

    time_series = data.groupby('Date')[['Production (MW)', 'Consommation (MW)']].sum()

    fig, ax = plt.subplots(figsize=(12, 6))
    time_series.plot(ax=ax)
    ax.set_title("Production et Consommation d'Électricité au Fil du Temps")
    ax.set_ylabel("Puissance (MW)")
    ax.set_xlabel("Date")
    plt.tight_layout()
    return fig

# Main app logic
def main():
    st.set_page_config(page_title="Eco2Mix Dashboard", layout="wide")

    uploaded_file = st.sidebar.file_uploader("Upload your dataset", type="csv")
    if uploaded_file:
        data = load_data(uploaded_file)
        if data is not None:
            pages = ["Overview", "Visualizations", "Predictions"]
            selected_page = st.sidebar.selectbox("Navigation", pages)

            if selected_page == "Overview":
                st.title("Energy Production and Consumption in France")
                st.write("Analyzing energy trends from 2013 to 2024.")
                st.write(data.head())

            elif selected_page == "Visualizations":
                st.title("Visualizations")
                visualization = st.selectbox("Choose Visualization", [
                    "Energy Production Map",
                    "Time Series Plot"
                ])

                if visualization == "Energy Production Map":
                    fig = create_energy_production_map(data)
                    st.pyplot(fig)
                elif visualization == "Time Series Plot":
                    fig = create_time_series_plot(data)
                    st.pyplot(fig)

            elif selected_page == "Predictions":
                st.title("Predictions")
                st.write("Prediction features coming soon!")
    else:
        st.info("Please upload a CSV file to proceed.")

if __name__ == "__main__":
    main()
