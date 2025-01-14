# Statistical Study on Electricity Consumption and Production in France

This repository contains an in-depth analysis of electricity production and consumption across France, covering regions, departments, and communes. The study aims to provide actionable insights and predictions using historical electricity data and geospatial analysis.

## Overview
The study utilizes datasets at the regional, departmental, and commune levels. It focuses on trends in electricity production and consumption, identifying patterns, anomalies, and predictive models. The data is sourced from publicly available datasets and includes variables like electricity type (renewable, non-renewable), capacity, demand, and geographical distribution.


## Data Sources
- **Electricity Production and Consumption Data**: from the Open Data Réseaux Énergies (ODRÉ) (https://odre.opendatasoft.com/). The specific dataset used is [éCO2mix Régional Consolidé et Définitif](https://odre.opendatasoft.com/explore/dataset/eco2mix-regional-cons-def/export/?disjunctive.nature&disjunctive.libelle_region), which contains consolidated and definitive regional electricity data from January 2013 to January 2023. 

The data for this analysis was retrieved on **December 10, 2024**.

## Project Structure
- **data/**: Stores the data files (e.g., CSV files, datasets).
- **notebooks/**: Includes data vizualisations notebook and trained machine learning models notebook
- **streamlit/**: Contains Streamlit app files for interactive data exploration and visualization.
- **requirements.txt**: Lists all Python packages and dependencies required to run the project.
- **README.md**: This file provides an overview and instructions for the project.

## Installation and Setup
To replicate the analysis and run the code on your local machine, it is recommended to set up a virtual environment using the provided `requirements.txt` file. This ensures all necessary libraries and dependencies are installed.

### Steps to Install Dependencies
1. **Set up a virtual environment** (optional but recommended):
   ```bash
   python -m venv env
   source env/bin/activate  # On macOS/Linux
   env\Scripts\activate  # On Windows
   ```

2. **Install dependencies from `requirements.txt`:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the notebooks :**
   Open the provided Jupyter Notebooks in the **vizualisations/** directory to explore the analysis.
   Open the provided Jupyter Notebooks in the **Models/** directory to explore the analysis.

4. **Run the Streamlit app:**
   Navigate to the `streamlit/` directory and run the app using:
   ```bash
   streamlit run app.py
   ```

## Objectives
- Analyze electricity production and consumption patterns across different regions and departments.
- Evaluate the contribution of renewable vs. non-renewable energy sources.
- Build predictive models for electricity demand and supply.
- Visualize trends and geographic distributions of electricity usage.
- Provide interactive dashboards for data exploration using Streamlit.

## Contact
For any questions or feedback, please contact the project maintainer: **raphaelhoudouin**.  
GitHub: [raphaelhoudouin](https://github.com/raphaelhoudouin)


