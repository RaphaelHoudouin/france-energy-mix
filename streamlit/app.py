### code streamlit qui a pas fonctionné
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
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import zscore

# Charger les données avec gestion d'erreurs
@st.cache_data
def load_data(data_url="/content/drive/MyDrive/groupeDeTravail-BDAenergie/eco2mix-regional-cons-defcopiecopy.csv", geo_url="https://raw.githubusercontent.com/gregoiredavid/france-geojson/master/regions-version-simplifiee.geojson"):
    # Création du 1er dataframe pour les données
    data = pd.read_csv(data_url, sep=";")

    # Création du 2e dataframe pour les données géographiques
    response = requests.get(geo_url)
    temp_dir = tempfile.gettempdir()
    temp_file = os.path.join(temp_dir, "regions.geojson")
    with open(temp_file, 'wb') as f:
        f.write(response.content)

    # Résultat de la fonction = 2 DataFrames
    return data, temp_file

# Utilisation de la fonction
data_url = "/content/drive/MyDrive/groupeDeTravail-BDAenergie/eco2mix-regional-cons-defcopiecopy.csv"
geo_url = "https://raw.githubusercontent.com/gregoiredavid/france-geojson/master/regions-version-simplifiee.geojson"

data, geo = load_data(data_url, geo_url)
# Exécution de la fonction avec les 2 URLs ci-dessus. La fonction va retourner 2 df appelés data et geo.
# 1er df "data" = données du dataset
# 2nd df "geo" = données géographiques

# Fonction pour valider les colonnes
def validate_columns(data, required_columns):
    missing_cols = [col for col in required_columns if col not in data.columns]
    if missing_cols:
        raise ValueError(f"Les colonnes suivantes sont manquantes dans le DataFrame : {missing_cols}")

# Graphique 1 : Carte de production d'électricité par région
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

    regions_gdata = gpd.read_file(geo)
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

# Graphique 2 : Série temporelle de production et consommation
def create_time_series_plot(data):
    data['Date'] = pd.to_datetime(data['Date'], errors='coerce')

    energy_cols = ['Thermique (MW)', 'Nucléaire (MW)', 'Eolien (MW)',
                   'Solaire (MW)', 'Hydraulique (MW)', 'Pompage (MW)',
                   'Bioénergies (MW)']

    for col in energy_cols:
        data[col] = pd.to_numeric(data[col], errors='coerce').fillna(0)

    data['Production (MW)'] = data[energy_cols].sum(axis=1)

    data['Consommation (MW)'] = pd.to_numeric(data['Consommation (MW)'], errors='coerce').fillna(0)

    time_series = data.groupby('Date')[['Production (MW)', 'Consommation (MW)']].sum()

    fig, ax = plt.subplots(figsize=(12, 6))
    time_series['Production (MW)'].plot(ax=ax, label='Production Totale (MW)', color='blue')
    time_series['Consommation (MW)'].plot(ax=ax, label='Consommation Totale (MW)', color='red')

    ax.set_title("Production et Consommation d'Électricité au Fil du Temps")
    ax.set_ylabel("Puissance (MW)")
    ax.set_xlabel("Date")
    ax.legend()
    plt.tight_layout()
    return fig

# Graphique 3 : Impact de la COVID-19
def create_covid_impact_plot(data):
    data['Date'] = pd.to_datetime(data['Date'], format='%Y-%m-%d', errors='coerce')

    energy_cols = ['Thermique (MW)', 'Nucléaire (MW)', 'Eolien (MW)',
                   'Solaire (MW)', 'Hydraulique (MW)', 'Pompage (MW)',
                   'Bioénergies (MW)']

    for col in energy_cols:
        data[col] = pd.to_numeric(data[col], errors='coerce').fillna(0)

    data['Production (MW)'] = data[energy_cols].sum(axis=1)

    covid_data = data[data['Date'] >= '2020-03-01']
    covid_time_series = covid_data.groupby('Date')[['Production (MW)', 'Consommation (MW)']].sum()

    fig, ax = plt.subplots(figsize=(12, 6))
    covid_time_series['Production (MW)'].plot(ax=ax, label='Production Totale (MW)', color='blue')
    covid_time_series['Consommation (MW)'].plot(ax=ax, label='Consommation Totale (MW)', color='red')

    ax.set_title("Impact de la COVID-19 sur la Production d'Électricité")
    ax.set_ylabel("Puissance (MW)")
    ax.set_xlabel("Date")
    ax.legend()
    plt.tight_layout()
    return fig

# Graphique 4 : Histogramme bimensuel
def create_biweekly_histogram(data):
    data['Date'] = pd.to_datetime(data['Date'], errors='coerce')

    energy_cols = ['Thermique (MW)', 'Nucléaire (MW)', 'Eolien (MW)',
                   'Solaire (MW)', 'Hydraulique (MW)', 'Pompage (MW)',
                   'Bioénergies (MW)']

    for col in energy_cols:
        data[col] = pd.to_numeric(data[col], errors='coerce').fillna(0)

    data['Production (MW)'] = data[energy_cols].sum(axis=1)
    data['Consommation (MW)'] = pd.to_numeric(data['Consommation (MW)'], errors='coerce').fillna(0)

    data.set_index('Date', inplace=True)
    biweekly_data = data.resample('2M')[['Production (MW)', 'Consommation (MW)']].sum()

    fig, ax = plt.subplots(figsize=(14, 7))
    biweekly_data.plot(kind='bar', stacked=True, ax=ax, colormap='viridis')
    ax.set_title("Production et Consommation d'Électricité Bimensuelle")
    plt.xticks(rotation=45)
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
    # Valider la présence des colonnes nécessaires
    tch_columns = [
        'TCH Thermique (%)', 'TCH Nucléaire (%)',
        'TCH Eolien (%)', 'TCH Hydraulique (%)',
        'TCH Solaire (%)', 'TCH Bioénergies (%)'
    ]

    missing_cols = [col for col in tch_columns if col not in data.columns]
    if missing_cols:
        raise ValueError(f"Colonnes manquantes pour le graphique TCH : {missing_cols}")

    # Calculer la somme des TCH par source d'énergie
    total_TCH_france_metro_hors_corse = data[tch_columns].sum()

    # Afficher la somme des TCH sous forme de tableau
    st.subheader("Somme des Taux de Charge (TCH) par Source d'Énergie")
    st.dataframe(total_TCH_france_metro_hors_corse)

    # Tracer le graphique circulaire
    st.subheader("Graphique Circulaire de la Répartition des TCH")
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.pie(
        total_TCH_france_metro_hors_corse,
        labels=total_TCH_france_metro_hors_corse.index,
        autopct='%1.1f%%',
        startangle=90,
        colors=sns.color_palette("pastel")
    )
    ax.set_title("Répartition Globale du Taux de Charge (TCH) par Source d'Énergie", fontsize=14)
    st.pyplot(fig)


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


def graphique_8(data):
    # Remplacer les valeurs manquantes par 0
    cols_to_fill = [
        'Consommation (MW)', 'Thermique (MW)', 'Nucléaire (MW)',
        'Eolien (MW)', 'Solaire (MW)', 'Hydraulique (MW)',
        'Pompage (MW)', 'Bioénergies (MW)'
    ]
    for col in cols_to_fill:
        if col in data.columns:
            if col == 'Eolien (MW)':
                data[col] = pd.to_numeric(data[col], errors='coerce').fillna(0)
            else:
                data[col] = data[col].fillna(0)

    # Calcul de la consommation totale par région
    if 'Région' in data.columns:
        consumption_by_region = data.groupby('Région')['Consommation (MW)'].sum().sort_values(ascending=False)

        # Affichage des données sous forme de tableau
        st.subheader("Consommation d'Électricité par Région")
        st.dataframe(consumption_by_region)

        # Afficher un graphique de la consommation par région
        st.subheader("Graphique : Consommation d'Électricité par Région")
        fig, ax = plt.subplots(figsize=(14, 7))
        consumption_by_region.plot(kind='bar', color='orange', ax=ax)
        ax.set_title("Consommation d'Électricité par Région")
        ax.set_xlabel("Région")
        ax.set_ylabel("Consommation (MW)")
        ax.grid(axis='y')
        plt.tight_layout()
        return fig

def graphique_9(data):
    # Convertir la colonne "Eolien (MW)" en numérique
    data['Eolien (MW)'] = pd.to_numeric(data['Eolien (MW)'], errors='coerce').fillna(0)

    # Liste des colonnes à utiliser
    cols_to_fill = [
        'Consommation (MW)', 'Thermique (MW)', 'Nucléaire (MW)',
        'Eolien (MW)', 'Solaire (MW)', 'Hydraulique (MW)',
        'Pompage (MW)', 'Bioénergies (MW)'
    ]

    # Calcul de la production totale
    data['Production (MW)'] = data[cols_to_fill].sum(axis=1)

    # Groupement des données par région et somme des productions par type
    production_by_region = data.groupby('Région')[cols_to_fill].sum()

    # Ajout d'une colonne pour la production totale par région
    production_by_region['Production Totale'] = production_by_region.sum(axis=1)

    # Trier les régions par production totale
    sorted_values_production_by_region = production_by_region.sort_values(by='Production Totale', ascending=False)

    # Supprimer la colonne "Production Totale" pour le graphique
    sorted_values_production_by_region_plot = sorted_values_production_by_region.drop(columns=['Production Totale'])

    # Affichage des données
    st.subheader("Production par Région")
    st.dataframe(production_by_region)

    # Afficher un histogramme empilé des productions par source et par région
    st.subheader("Graphique : Production d'Électricité par Source et par Région")
    fig, ax = plt.subplots(figsize=(14, 7))
    sorted_values_production_by_region_plot.plot(kind='bar', stacked=True, ax=ax)
    plt.title("Production d'Électricité par Source et par Région")
    plt.ylabel("Production (MW)")
    plt.xlabel("Régions")

    return fig

def main():
    # Chargement des données
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
        st.header("📄 Contexte")
        st.write(
            """
            La plateforme **Open Data Réseaux Énergies (ODRÉ)** propose un jeu de données régional consolidé
            sur la consommation et la production d’électricité en France. Ces données, couvrant la période
            de **janvier 2013 à janvier 2023**, sont :
            - **Actualisées quotidiennement**
            - **Vérifiées et complétées** pour assurer leur fiabilité
            - Détaillées par région
            """
        )

        # Section : Objectifs
        st.header("🎯 Objectifs")
        st.write(
            """
            Ce jeu de données permet de fournir des informations complètes et fiables sur :
            """
        )
        st.markdown("""
        - **Consommation d’électricité** réalisée par région.
        - **Production d’électricité** par filière énergétique, avec une précision à la demi-heure.
        - **Consommation des Stations de Transfert d'Énergie par Pompage (STEP)** pour le stockage et la régulation de l’énergie.
        - **Solde des échanges interrégionaux** pour évaluer l’autosuffisance énergétique et les flux d’électricité.
        """)

        # Section : Finalités
        st.header("📊 Finalités")
        st.markdown("""
        Grâce à ces données, il est possible de :
        - Suivre l'évolution de la **consommation et de la production d’électricité** dans le temps et par région.
        - Analyser la **couverture énergétique** régionale par filière (**TCO**) et la performance des installations (**TCH**).
        - Effectuer des **analyses comparatives** entre régions pour comprendre les dynamiques énergétiques.
        """)


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
            "Répartition globale du TCO",
            "Consommation d'Électricité par Région",
            "Production des Énergies par Région",
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
            elif choix_graphique == graphiques[6]:
                fig = graphique_8(data)
                st.pyplot(fig)
            elif choix_graphique == graphiques[7]:
                fig = graphique_9(data)
                st.pyplot(fig)
        except Exception as e:
            st.error(f"Erreur lors de la génération du graphique : {e}")

    # Page de Prédictions
    elif selected_page == "Modélisations":
        st.title("Modélisations")
        st.write("Section de prédictions basée sur des modèles de machine learning.")

        # Liste déroulante pour les visualisations
        options = [
            'Boxplot',
            'Performance des modèles',
            'Graphique de corrélation',
            'Régression Linéaire',
            'Régression Ridge',
            'Régression Lasso',
            'Arbre de Décision',
            'Forêt Aléatoire',
            'Régression : Île-de-France'
        ]

        selected_option = st.selectbox("Choisissez une visualisation :", options)

        # Dictionnaire des liens vers les images
        images = {
            'Boxplot': "/content/drive/MyDrive/groupeDeTravail-BDAenergie/Boxplot de consomation.png",
            'Performance des modèles': "/content/drive/MyDrive/groupeDeTravail-BDAenergie/Perforamance des modèles.png",
            'Graphique de corrélation': "/content/drive/MyDrive/groupeDeTravail-BDAenergie/Graphique de corrélation.png",
            'Régression Linéaire': "/content/drive/MyDrive/groupeDeTravail-BDAenergie/Régresionn liéaire.png",
            'Régression Ridge': "/content/drive/MyDrive/groupeDeTravail-BDAenergie/Régression de Ridge.png",
            'Régression Lasso': "/content/drive/MyDrive/groupeDeTravail-BDAenergie/Régression Lasso.png",
            'Arbre de Décision': "/content/drive/MyDrive/groupeDeTravail-BDAenergie/Arbre de décision.png",
            'Forêt Aléatoire': "/content/drive/MyDrive/groupeDeTravail-BDAenergie/Modèle foret aléatoire.png",
            'Régression : Île-de-France': "/content/drive/MyDrive/groupeDeTravail-BDAenergie/Régression ile de France.png",
        }

        # Affichage de l'image correspondant à l'option sélectionnée
        if selected_option in images:
            st.image(images[selected_option], caption=f"Visualisation : {selected_option}", use_container_width=True)
        else:
            st.warning("Aucune image disponible pour cette sélection.")

    # Page de conclusion
    elif selected_page == "Conclusion":
      st.title("Conclusion")
      st.write("Résumé des analyses et perspectives futures.")

      st.markdown("""
      - 🔍 L'étude a comparé différents modèles de régression pour prédire la consommation et la production d'énergie en France : régression linéaire, Ridge, Lasso, arbres de décision et forêts aléatoires.
      - 🌳 Les modèles non linéaires, notamment les arbres de décision et les forêts aléatoires, ont offert les meilleures performances (R² élevés, erreurs faibles).
      - ⚡ L'analyse des corrélations a révélé l'importance des bioénergies et des énergies renouvelables dans le mix énergétique.
      - 🌞 Le solaire a montré des corrélations limitées, nécessitant des recherches futures pour mieux comprendre son potentiel.
      - 🔮 Conclusion : les modèles non linéaires sont efficaces, mais doivent être utilisés avec prudence pour éviter le sur-apprentissage. Cette étude constitue une base pour des recherches futures sur la gestion des ressources énergétiques.
      """)

# Point d'entrée de l'application
if __name__ == "__main__":
    main()
