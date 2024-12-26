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

# Charger les données
@st.cache_data
def load_data():
    return pd.read_csv("/content/drive/MyDrive/groupeDeTravail-BDAenergie/eco2mix-regional-cons-defcopiecopy.csv", sep=";")

data = pd.read_csv("/content/drive/MyDrive/groupeDeTravail-BDAenergie/eco2mix-regional-cons-defcopiecopy.csv", sep=";")
def download_region_geojson():
    url = "https://raw.githubusercontent.com/gregoiredavid/france-geojson/master/regions-version-simplifiee.geojson"
    response = requests.get(url)
    temp_dir = tempfile.gettempdir()
    temp_file = os.path.join(temp_dir, "regions.geojson")
    with open(temp_file, 'wb') as f:
        f.write(response.content)
    return temp_file

# Fonction pour vérifier les colonnes
def validate_columns(data, required_columns):
    missing_cols = [col for col in required_columns if col not in data.columns]
    if missing_cols:
        raise ValueError(f"Les colonnes suivantes sont manquantes dans le DataFrame : {missing_cols}")

# Graphique 1 : Carte de production d'électricité par région
def create_energy_production_map(data):
    energy_cols = ['Thermique (MW)', 'Nucléaire (MW)', 'Eolien (MW)',
                   'Solaire (MW)', 'Hydraulique (MW)', 'Pompage (MW)',
                   'Bioénergies (MW)']

    # Valider les colonnes nécessaires
    validate_columns(data, energy_cols + ['Région'])

    # Conversion en numérique et gestion des valeurs manquantes
    for col in energy_cols:
        data[col] = pd.to_numeric(data[col], errors='coerce').fillna(0)

    # Calcul des productions totales et renouvelables
    data['Production (MW)'] = data[energy_cols].sum(axis=1)
    renewable_cols = ['Eolien (MW)', 'Solaire (MW)', 'Hydraulique (MW)', 'Bioénergies (MW)']
    data['Renouvelable (MW)'] = data[renewable_cols].sum(axis=1)

    # Agrégation par région
    production_by_region = data.groupby('Région')[['Production (MW)', 'Renouvelable (MW)']].sum()
    production_by_region['Pourcentage Renouvelable'] = (
        production_by_region['Renouvelable (MW)'] / production_by_region['Production (MW)'].replace(0, 1) * 100
    )

    # Charger les données géographiques
    regions_file = download_region_geojson()
    regions_gdata = gpd.read_file(regions_file)

    # Normalisation des noms de régions
    regions_gdata['nom'] = regions_gdata['nom'].str.upper().apply(lambda x: unidecode.unidecode(x))
    production_by_region.index = [unidecode.unidecode(x.upper()) for x in production_by_region.index]

    # Fusion des données
    regions_gdata = regions_gdata.merge(production_by_region, left_on='nom', right_index=True, how='left')

    # Création de la carte
    fig, ax = plt.subplots(figsize=(15, 10))
    regions_gdata.plot(
        column='Production (MW)',
        ax=ax,
        legend=True,
        legend_kwds={'label': 'Production Totale (MW)'},
        cmap='YlOrRd',
        missing_kwds={'color': 'lightgrey'}
    )

    # Annotation des régions
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

def create_time_series_plot(data):
    # Vérifiez si la colonne 'Date' est présente
    if 'Date' not in data.columns:
        raise ValueError("La colonne 'Date' est manquante dans le DataFrame.")

    # Conversion de la colonne 'Date' au format datetime
    data['Date'] = pd.to_datetime(data['Date'], errors='coerce')

    # Vérifiez si les colonnes de production sont présentes
    energy_cols = ['Thermique (MW)', 'Nucléaire (MW)', 'Eolien (MW)',
                   'Solaire (MW)', 'Hydraulique (MW)', 'Pompage (MW)',
                   'Bioénergies (MW)']

    missing_cols = [col for col in energy_cols if col not in data.columns]
    if missing_cols:
        raise ValueError(f"Les colonnes suivantes sont manquantes dans le DataFrame : {missing_cols}")

    # Assurez-vous que toutes les colonnes d'énergie sont numériques
    for col in energy_cols:
        data[col] = pd.to_numeric(data[col], errors='coerce').fillna(0)

    # Calcul de la production totale d'électricité
    data['Production (MW)'] = data[energy_cols].sum(axis=1)
    print("Premières lignes après calcul de la production :", data[['Date', 'Production (MW)']].head())

    # Vérifiez si la colonne 'Consommation (MW)' est présente et numérisez-la
    if 'Consommation (MW)' not in data.columns:
        raise ValueError("La colonne 'Consommation (MW)' est manquante dans le DataFrame.")

    data['Consommation (MW)'] = pd.to_numeric(data['Consommation (MW)'], errors='coerce').fillna(0)

    # Triez les données par date
    data_sorted = data.sort_values('Date')

    # Somme de la production et consommation par date
    time_series = data_sorted.groupby('Date')[['Production (MW)', 'Consommation (MW)']].sum()

    # Tracer les séries temporelles
    fig, ax = plt.subplots(figsize=(12, 6))
    time_series['Production (MW)'].plot(ax=ax, label='Production Totale (MW)', color='blue')
    time_series['Consommation (MW)'].plot(ax=ax, label='Consommation Totale (MW)', color='red')

    # Ajouter les titres et légendes
    ax.set_title("Production et Consommation d'Électricité au Fil du Temps")
    ax.set_ylabel("Puissance (MW)")
    ax.set_xlabel("Date")
    ax.legend()
    plt.tight_layout()
    return fig




# Graphique 4 : Impact de la COVID-19 sur la production
def create_covid_impact_plot(data):
    # Conversion de la colonne 'Date' au format datetime
    data['Date'] = pd.to_datetime(data['Date'], format='%Y-%m-%d', errors='coerce')

    # Vérifiez si les colonnes de production sont présentes
    energy_cols = ['Thermique (MW)', 'Nucléaire (MW)', 'Eolien (MW)',
                   'Solaire (MW)', 'Hydraulique (MW)', 'Pompage (MW)',
                   'Bioénergies (MW)']

    missing_cols = [col for col in energy_cols if col not in data.columns]
    if missing_cols:
        raise ValueError(f"Les colonnes suivantes sont manquantes dans le DataFrame : {missing_cols}")

    # Assurez-vous que toutes les colonnes d'énergie sont numériques
    for col in energy_cols:
        data[col] = pd.to_numeric(data[col], errors='coerce').fillna(0)

    # Calcul de la production totale d'électricité
    data['Production (MW)'] = data[energy_cols].sum(axis=1)

    # Filtrer les données à partir de mars 2020
    covid_data = data[data['Date'] >= '2020-03-01']
    covid_data_sorted = covid_data.sort_values('Date')

    # Agréger les données
    covid_time_series = covid_data_sorted.groupby('Date')[['Production (MW)', 'Consommation (MW)']].sum()

    # Tracer les séries temporelles
    fig, ax = plt.subplots(figsize=(12, 6))
    covid_time_series['Production (MW)'].plot(ax=ax, label='Production Totale (MW)', color='blue')
    covid_time_series['Consommation (MW)'].plot(ax=ax, label='Consommation Totale (MW)', color='red')
    ax.set_title("Impact de la COVID-19 sur la Production d'Électricité")
    ax.set_ylabel("Puissance (MW)")
    ax.set_xlabel("Date")
    ax.legend()
    plt.tight_layout()
    return fig


# Graphique 5 : Histogramme de la Production et Consommation bimensuelle

def create_biweekly_histogram(data):
    # Conversion de la colonne 'Date' au format datetime avec un format explicite
    data['Date'] = pd.to_datetime(data['Date'], format='%Y-%m-%d', errors='coerce')

    # Vérifiez si les colonnes de production sont présentes
    energy_cols = ['Thermique (MW)', 'Nucléaire (MW)', 'Eolien (MW)',
                   'Solaire (MW)', 'Hydraulique (MW)', 'Pompage (MW)',
                   'Bioénergies (MW)']

    missing_cols = [col for col in energy_cols if col not in data.columns]
    if missing_cols:
        raise ValueError(f"Les colonnes suivantes sont manquantes dans le DataFrame : {missing_cols}")

    # Assurez-vous que toutes les colonnes d'énergie sont numériques
    for col in energy_cols:
        data[col] = pd.to_numeric(data[col], errors='coerce').fillna(0)

    # Calcul de la production totale d'électricité
    data['Production (MW)'] = data[energy_cols].sum(axis=1)

    # Vérifiez si la colonne 'Consommation (MW)' est présente
    if 'Consommation (MW)' not in data.columns:
        raise ValueError("La colonne 'Consommation (MW)' est manquante dans le DataFrame.")

    # Filtrer les données à partir de 2013
    data = data[data['Date'] >= '2013-01-01']

    # Définir 'Date' comme index pour le regroupement
    data.set_index('Date', inplace=True)

    # Regrouper les données par période de deux mois (bimensuel) et sommer
    biweekly_data = data.resample('2M')[['Production (MW)', 'Consommation (MW)']].sum()

    # Tracer l'histogramme empilé
    fig, ax = plt.subplots(figsize=(14, 7))

    # Tracer la production et la consommation empilées
    biweekly_data.plot(kind='bar', stacked=True, ax=ax, colormap='viridis')

    # Formater l'axe x pour afficher les dates au format 'YYYY-MM-DD'
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))  # Afficher une étiquette tous les deux mois
    plt.xticks(rotation=45)  # Rotation pour améliorer la lisibilité

    # Ajouter un titre et des labels
    ax.set_title("Production et Consommation d'Électricité Bimensuelle")
    ax.set_ylabel("Puissance (MW)")
    ax.set_xlabel("Période Bimensuelle")

    # Retourner la figure
    return fig

# Graphique 6 : Répartition globale du TCH
def graphique_6(data):
        # Calculer la somme des TCH par source d'énergie
        total_TCH_france_metro_hors_corse = data[['TCH Thermique (%)', 'TCH Nucléaire (%)',
                                                'TCH Eolien (%)', 'TCH Hydraulique (%)',
                                                'TCH Solaire (%)', 'TCH Bioénergies (%)']].sum()

# Afficher la somme des TCH sous forme de tableau
        st.subheader("Somme des Taux de Charge (TCH) par Source d'Énergie")
        st.dataframe(total_TCH_france_metro_hors_corse)

        # Tracer le graphique circulaire
        st.subheader("Graphique Circulaire de la Répartition des TCH")
        fig, ax = plt.subplots(figsize=(8, 8))

        ax.pie(total_TCH_france_metro_hors_corse,
               labels=total_TCH_france_metro_hors_corse.index,  # Labels issus des colonnes
               autopct='%1.0f%%',  # Afficher les pourcentages
               startangle=140)

        ax.set_title("Répartition du Taux de Charge (TCH) par Source d'Énergie")
        ax.legend(fontsize=8, loc="upper right")
        return fig

# Graphique 7 : Répartition globale du TCO

def graphique_7(data):
    # Vérifier que toutes les colonnes nécessaires existent dans le DataFrame
    required_cols = ['TCO Thermique (%)', 'TCO Nucléaire (%)',
                    'TCO Eolien (%)', 'TCO Hydraulique (%)',
                    'TCO Solaire (%)', 'TCO Bioénergies (%)']

    missing_cols = [col for col in required_cols if col not in data.columns]
    if missing_cols:
        raise ValueError(f"Les colonnes suivantes sont manquantes dans le DataFrame : {missing_cols}")

    # Calculer la somme des TCH par source d'énergie
    total_TCH_france_metro_hors_corse = data[required_cols].sum()

    # Afficher la somme des TCH sous forme de tableau
    st.subheader("Somme des Taux de Charge (TCO) par Source d'Énergie")
    st.dataframe(total_TCH_france_metro_hors_corse)

    # Tracer le graphique circulaire
    st.subheader("Graphique Circulaire de la Répartition du TCO")
    fig, ax = plt.subplots(figsize=(8, 8))

    ax.pie(total_TCH_france_metro_hors_corse,
           labels=total_TCH_france_metro_hors_corse.index,  # Labels issus des colonnes
           autopct='%1.0f%%',  # Afficher les pourcentages
           startangle=140)

    ax.set_title("Répartition du Taux de Charge (TCO) par Source d'Énergie")
    ax.legend(fontsize=8, loc="upper right")



    return fig

def main():
    # Configuration de l'application
    st.set_page_config(page_title="Eco2Mix Dashboard", layout="wide")

    # Charger les données
    data = load_data()

    # Pages disponibles
    pages = ["Accueil", "Présentation", "Préprocessing", "Graphiques", "Prédictions", "Conclusion"]
    selected_page = st.sidebar.selectbox("Navigation", pages)

    # Page d'Accueil
    if selected_page == "Accueil":
        st.title("RAPPORT SUR LA CONSOMMATION ET LA PRODUCTION D'ÉNERGIE EN FRANCE")
        st.markdown("<h2 style='font-size: 28px; font-weight: bold;'>Étude sur la production et la consommation d'électricité</h2>", unsafe_allow_html=True)
        st.write("en France métropolitaine de janvier 2013 à septembre 2024")
        st.image("https://storage.letudiant.fr/mediatheque/letudiant/7/8/2635278-differentes-sources-energie-766x438.jpeg")

        st.markdown("<h2 style='font-size: 28px; font-weight: bold;'>Membres du Projet :</h2>", unsafe_allow_html=True)
        st.write("JALILI Amine")
        st.write("HOUDOUIN Jean-Raphaël")
        st.write("TOURE Mariama Mountaga")
        st.write("YEBGAR Lucien")

        st.markdown("<h2 style='font-size: 28px; font-weight: bold;'>Mentor :</h2>", unsafe_allow_html=True)
        st.write("Alain Ferlac")


    # Page de Présentation
    elif selected_page == "Présentation":
        st.title("Présentation")

        st.write("Description du projet et des objectifs d'analyse.")

    # Page de Préprocessing
    elif selected_page == "Préprocessing":
        st.title("Préprocessing")
        st.subheader("Aperçu des données")

        # Sélectionner les colonnes à afficher
        columns_to_show = st.multiselect("Choisir les colonnes à afficher", data.columns.tolist(), default=data.columns.tolist())
        st.dataframe(data[columns_to_show].head(50))

        # Afficher des statistiques descriptives
        if st.checkbox("Afficher les statistiques descriptives"):
            st.subheader("Statistiques Descriptives")
            st.write(data.describe())

        # Afficher les valeurs manquantes
        if st.checkbox("Afficher les valeurs manquantes"):
            st.subheader("Valeurs Manquantes")
            st.write(data.isnull().sum())

        # Ajouter un bouton pour recharger les données
        if st.button("Recharger les données"):
            data = load_data()
            st.success("Les données ont été rechargées avec succès !")

    # Page de Graphiques
    elif selected_page == "Graphiques":
        st.title("Graphiques")
        graphiques = [
            "Production d'électricité par Région",
            "Production et Consommation d'électricité au fil du temps",
            "Impact de la COVID-19 sur la production",
            "Histogramme de la Production et Consommation bimensuelle",
            "Répartition globale du TCH",
            "Répartition globale du TCO"
        ]
        choix_graphique = st.selectbox("Choisissez un graphique", graphiques)

        # Ajouter des try-except blocks pour chaque graphique
        try:
            if choix_graphique == graphiques[0]:
                fig = create_energy_production_map(data)
                st.pyplot(fig)
            elif choix_graphique == graphiques[1]:
                fig = create_time_series_plot(data)
                st.pyplot(fig)
            elif choix_graphique == graphiques[2]:
                fig = create_covid_impact_plot(data)
                st.pyplot(fig)
            elif choix_graphique == graphiques[3]:
                fig = create_biweekly_histogram(data)
                st.pyplot(fig)
            elif choix_graphique == graphiques[4]:
                fig = graphique_6(data)
                st.pyplot(fig)
            elif choix_graphique == graphiques[5]:
                fig = graphique_7(data)
                st.pyplot(fig)
        except Exception as e:
            st.error(f"Erreur lors de la génération du graphique : {e}")

    # Page de Prédictions
    elif selected_page == "Prédictions":
        st.title("Modélisations")

        # Prétraitement des données (à déplacer avant la division train/test si possible)
        data['Consommation (MW)'] = data['Consommation (MW)'].fillna(0)

        # Sélection des caractéristiques
        features = ['Thermique (MW)', 'Nucléaire (MW)', 'Eolien (MW)', 'Solaire (MW)',
                    'Hydraulique (MW)', 'Pompage (MW)', 'Bioénergies (MW)']

        # Préparation des données
        X = data[features]
        y = data['Consommation (MW)']

        # Division des données
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Modèles
        models = {
            'Linear Regression': LinearRegression(),
            'Decision Tree': DecisionTreeRegressor(random_state=42),
            'Random Forest': RandomForestRegressor(random_state=42)
        }

        # Sélection du modèle
        model_choice = st.selectbox("Choisir un modèle", list(models.keys()))
        model = models[model_choice]

        # Entraînement et prédiction
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Métriques
        st.subheader(f"Performances du modèle {model_choice}")
        st.write(f"MSE: {mean_squared_error(y_test, y_pred):.2f}")
        st.write(f"RMSE: {mean_squared_error(y_test, y_pred, squared=False):.2f}")
        st.write(f"MAE: {mean_absolute_error(y_test, y_pred):.2f}")
        st.write(f"R²: {r2_score(y_test, y_pred):.2f}")

        # Graphiques de performance
        col1, col2, col3 = st.columns(3)
        with col1:
            st.pyplot(plot_predictions_vs_reality(y_test, y_pred, model_choice))
        with col2:
            st.pyplot(plot_error_distribution(y_pred - y_test, model_choice))
        with col3:
            st.pyplot(plot_error_boxplot(y_pred - y_test, model_choice))

    # Page de Conclusion
    elif selected_page == "Conclusion":
        st.title("Conclusion")
        st.write("Résumé des analyses et perspectives futures.")

# Point d'entrée de l'application
if __name__ == "__main__":
    main()