import streamlit as st
from PIL import Image

import pandas as pd
import numpy as np
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from datetime import datetime, timedelta

from typing import List, Tuple
from datetime import date, time, datetime
from numerize.numerize import numerize
from streamlit_elements import elements, mui, html
import altair as alt

import requests

import joblib
# from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score
# from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# from sklearn.svm import SVC
# from streamlit_option_menu import option_menu
from streamlit_navigation_bar import st_navbar
import pages as pg

import random


#### Page Configuration ####
st.set_page_config(
    page_title="AQI Suivi",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

alt.themes.enable("dark")

# Add custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 0.5rem;
        padding: 1rem;
        box-shadow: 0 0.125rem 0.25rem rgba(0, 0, 0, 0.075);
    }
    .chart-container {
        background-color: white;
        border-radius: 0.5rem;
        padding: 1rem;
        box-shadow: 0 0.125rem 0.25rem rgba(0, 0, 0, 0.075);
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

#______________________________Code de la requete___________________________________________

## Requete vers l'api pour recuperer les datas

url = "http://192.168.111.73:8000/api/dht/"
response = requests.get(url)

if response.status_code != 200:
    st.markdown("Chargement de données...")

    print(f"Error of connection in API: {response.status_code}")
    
else:
    data = response.json()
    data_bd = pd.DataFrame(data)
#data_bd = pd.read_csv("data_dht.csv")
        #print(data_bd.columns) 
        
        # columns = ['id', 'temperature', 'humidite', 'created_at', 'updated_at']

    ## Fin de la requete  

    ## Lecture des données passées pour la ville de Abidjan

    data_abidj = pd.read_csv("Abidjan_suivi.csv", sep=";")
    print(data_abidj.isna().sum())
    print(data_abidj.drop_duplicates())

    #_____________________________________Fin 1er code____________________________________

    ### Methode transform date for dataset_abidjan

    def transform_into_date(row):
        year = row['Year']
        month = row['Month']
        day = row['Day']
        return datetime(year=year, month=month, day=day)

    data_abidj['Date'] = data_abidj.apply(transform_into_date, axis=1)

    ### Creation de colonnes détaillées pour la visualisation

    data_bd['created_at'] = pd.to_datetime(data_bd['created_at'], utc=True)
    data_bd["Date"] = data_bd["created_at"].dt.strftime('%Y-%m-%d')
    data_bd["Jour"] = data_bd["created_at"].dt.strftime('%A')
    data_bd["Heure"] = data_bd["created_at"].dt.strftime(' %H:%M')
    data_bd["Hour_brute"] = data_bd["created_at"].dt.strftime(' %H')
    #print(data_bd)

    # Ajout de la Sidebar
    with st.sidebar:
        st.image("./images/log.png", width=200)
        st.title("AQI Dashboard Controls")
        
        st.divider()
        
        # Date range selector
        st.subheader("Date Range")
        today = datetime.now().date()
        default_start = today - timedelta(days=30)
        start_date = st.date_input("Start Date", default_start)
        end_date = st.date_input("End Date", today)
        st.divider()
        
        # Filters
        st.subheader("Filters")
        polluant_filter = st.multiselect(
            "Polluant Category",
            options=["temperature", "humidite", "NO2", "SO2", "Population_Density"],
            default=["temperature", "humidite"]
        )
        
        region_filter = st.selectbox(
            "Region",
            options=["Abidjan", "Yamoussoukro", "Jacqueville", "Divo"]
        )
        
        # Advanced options
        st.subheader("Advanced Options")
        show_targets = st.checkbox("Show Targets", value=True)
        show_forecasts = st.checkbox("Show Forecasts", value=False)
        
        st.divider()
        st.markdown("© 2025 AQI Compagny")

    # Fonction Generate data_set

    def generate_sample_data(start_date, end_date, data_f):
        
        data_f['Date'] = [value.to_pydatetime().date() for value in data_f['Date'].tolist()]
        date_range  = data_f.loc[(data_f['Date'] >= start_date) & (data_f['Date'] <= end_date)]
        
        print('-'*50)
        print('DATE RANGE')
        print(date_range)
        print('-'*50)
        return date_range


    def date_generate(start_date, end_date, data_f):
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        
        date_range = pd.to_datetime(date_range)
        date_range=date_range.strftime('%Y-%m-%d')
        print(type(date_range))
        
        for date in date_range:
            if date in data_f["Date"]:
                data_f = data_f[data_f["Date"] == date]
                print(data_f)
                print("data_f")
                
            else:
                print(date)
                
                print("no no no no")

        return data_f

    # Generate data_set

    data_abidj = generate_sample_data(start_date, end_date, data_abidj)

    # data_bd = date_generate(start_date, end_date, data_bd)
    # print(data_bd)

    image_path="./images/dcb_aqi.jpg"
    footer = f"""
    <style>
    .footer {{
        # position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        color: white;
        text-align: center;
        padding: 10px;
    }}
    .footer img {{
        width: 50px;
        heigth:auto;
    }}

    </style>
    <div class="footer">
        <p> BON A SAVOIR <p>
        <img src="/images/new_p.jpeg" alt="bon_a_savoir">
        <p>Développé avec ❤️ par Didiane KOFFI</p>
    </div>
    """


    # Main dashboard

    # Charger l'image
    image = Image.open("./images/AQI.png")

    # Redimensionner l'image aux dimensions souhaitées
    image_resized = image.resize((1000, 250))

    # Afficher l'image redimensionnée
    st.image(image_resized, use_container_width=True)
    # st.image("./images/AQI.png", use_column_width=True)
    st.markdown('<h1 class="main-header">Analytics Dashboard</h1>', unsafe_allow_html=True)

    # KPIs
    st.subheader("Key Performance Indicators")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        temperature = data_bd["temperature"].tail(1).iloc[0]
        temp_pass = data_bd["temperature"].tail(2).iloc[0]
        if temp_pass <= temperature:
            porcent = (temperature - temp_pass)/temperature
            st.metric(
                label="Temperature Actuelle",
                value=f"{temperature:,.2f} °C",
                delta=f"{porcent*100}%",
                
            )
        elif temp_pass > temperature:
            porcent = (temp_pass - temperature)/temp_pass
            st.metric(
                label="Temperature Actuelle",
                value=f"{temperature:,.2f} °C",
                delta=f"{-porcent*100}%",
                delta_color="inverse"
            )
            
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        humidite = data_bd["humidite"].tail(1).iloc[0]
        hum_pass = data_bd["humidite"].tail(2).iloc[0]
        if hum_pass <= humidite:
            porcent = (humidite - hum_pass)/humidite
            st.metric(
                label="Humidite Actuelle",
                value=f"{humidite:,.2f} %",
                delta=f"{porcent*100}%",
                
            )
        elif temp_pass > temperature:
            porcent = (temp_pass - temperature)/temp_pass
            st.metric(
                label="Humidite Actuelle",
                value=f"{humidite:,.2f} %",
                delta=f"{-porcent*100}%",
                delta_color="inverse"
            )
            
        st.markdown('</div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        temp_moy = data_bd.groupby(["Date", "Hour_brute"]).agg(mean_temp=("temperature","mean")).reset_index()
        temp_moy_h = temp_moy['mean_temp'].tail(1).iloc[0]
        temp_moy_p = temp_moy['mean_temp'].tail(2).iloc[0]
        
        if temp_moy_p <= temp_moy_h:
            porcent = (temp_moy_h - temp_moy_p)/temp_moy_h
            st.metric(
                label="Temperature Moyenne par heure",
                value=f"{temp_moy_h:,.2f} °C",
                delta=f"{porcent*100}%",
            )
        elif temp_moy_p > temp_moy_h:
            porcent = (temp_moy_p - temp_moy_h)/temp_moy_p
            st.metric(
                label="Temperature Moyenne par heure",
                value=f"{temp_moy_h:,.2f} °C",
                delta=f"{-porcent*100}%",
                delta_color="inverse"   
            )

        st.markdown('</div>', unsafe_allow_html=True)

    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        hum_moy =  data_bd.groupby(["Date", "Hour_brute"]).agg(mean_hum=("humidite","mean")).reset_index()
        hum_moy_h = hum_moy['mean_hum'].tail(1).iloc[0]
        hum_moy_p = hum_moy['mean_hum'].tail(2).iloc[0]
        
        if hum_moy_p <= hum_moy_h:
            porcent = (hum_moy_h - hum_moy_p)/hum_moy_h
            st.metric(
                label="Humidite par heure",
                value=f"{hum_moy_h:,.2f} %",
                delta=f"{porcent*100}%",
            )
        elif hum_moy_p > hum_moy_h:
            porcent = (hum_moy_p - hum_moy_h)/hum_moy_p
            st.metric(
                label="Humidite par heure",
                value=f"{hum_moy_h:,.2f} %",
                delta=f"{-porcent*100}%",
                delta_color="inverse"     
            )
        
        # st.metric(
        #     label="Humidite Moyenne  par heure",
        #     value=f"{hum_moy_h:,.2f} %",
        #     delta=f"{3.2}%"
        # )
        st.markdown('</div>', unsafe_allow_html=True)

    ### Contenu de la page

    st.subheader("AQI performances")
    tab1, tab2, tab3 = st.tabs(["Current Data follow-up", "AQI Predictions", "Good To Know"])

    ### FONCTION PREDICT 

    def perdiction(X_val):
        liste = []
        std = StandardScaler()
        X_scalee = std.fit_transform(X_val)
        y_pred = model.predict(X_scalee)
        #print(y_pred)
        #'Good': 0, 'Hazardous': 1, 'Moderate': 2, 'Poor': 3
        #
        if type(y_pred).__name__ != "ndarray":
            if y_pred == 0:
                aqi_indice = "Good"
                
            elif y_pred == 1:
                aqi_indice = "Hazardous"
                
            elif y_pred == 2:
                aqi_indice = "Moderate"
                
            elif y_pred == 3:
                aqi_indice = "Poor" 
            
            liste.append(aqi_indice)
        else:
            for i in y_pred:
                if i == 0:
                    aqi_indice = "Good"
                
                elif i == 1:
                    aqi_indice = "Hazardous"
                    
                elif i == 2:
                    aqi_indice = "Moderate"
                    
                elif i == 3:
                    aqi_indice = "Poor"
                    
                liste.append(aqi_indice)

        return liste

    ### Affichage des plots
    with tab1:
        
        c1, c2 = st.columns(2)
        
        with c1:
            st.subheader(":green[**Evolution des polluants**]")
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=hum_moy["Date"],
                y=hum_moy["mean_hum"],
                mode='lines',
                name='Humidite',
                line=dict(color='#3366CC', width=2)
            ))
            
            if show_targets:
                # Add target line
                target = temp_moy["mean_temp"]
                fig.add_trace(go.Scatter(
                    x=temp_moy["Date"],
                    y=target,
                    mode='lines',
                    name='Temperature',
                    line=dict(color='red', width=1, dash='dash')
                ))
            
            fig.update_layout(
                height=400,
                margin=dict(l=20, r=20, t=20, b=20),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                xaxis_title="Date",
                yaxis_title="Polluants",
                hovermode="x unified"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            

        with c2:
            st.subheader(":green[**FLUX DE LA JOURNÉE**]")
            
            temp_day = temp_moy[temp_moy["Date"] == datetime.today().strftime('%Y-%m-%d')]
            fig1 = px.line(temp_day, x="Hour_brute", 
                    y="mean_temp")
            #fig3.update_traces(line_color=selected_color_theme)
            
            st.markdown("#### :blue[**Flux Temperature par heures**]")
            
            st.plotly_chart(fig1)
            
            #____________________________________________________________
            
            hum_day = hum_moy[hum_moy["Date"] == datetime.today().strftime('%Y-%m-%d')]
            fig2 = px.line(hum_day, x="Hour_brute", 
                    y="mean_hum")
            #fig3.update_traces(line_color=selected_color_theme)
            
            st.markdown("#### :blue[**Flux Humidite par heures**]")
            
            st.plotly_chart(fig2)

    with tab2:
        model = joblib.load('Support Vector Machine.joblib')
        data = data_bd.groupby(["Date", "Hour_brute"]).agg(temperature=("temperature","mean"), humidite=("humidite","mean")).reset_index()
        
        data_pres = data_bd.groupby(["Date"]).agg(temperature=("temperature","mean"), humidite=("humidite","mean")).reset_index()
        X_pres = data_pres.drop(["Date"], axis=1)
        
        X_t = data.drop(["Date", "Hour_brute"], axis=1)
        
        ### AQI prediction par seconde
        X_val = X_t.tail(1)
        aqi_indice = perdiction(X_val)
        
        ### AQI prediction par heure
        
        aqi_day = perdiction(X_t)
        data["AQI_Indice"] = aqi_day
        data_reverse = data.sort_values("Date", ascending=False)
        
        st.title(":green[**Tendances de AQI - Changements évolutives de la qualité de l'air**]")
        
        ct1, ct2 = st.columns([3,5])
            
        with ct1:
            st.markdown(f"# Statut AQI actuel : {aqi_indice[0]}")
            
            st.dataframe(data_reverse, use_container_width=True, hide_index=True)
            st.markdown("---")
                    
            st.subheader(":green[**PREDICTION DES DERNIERS JOURS**]")       
            y_pres = perdiction(X_pres)
            data_pres["AQI_Indice"] = y_pres
            st.dataframe(data_pres,use_container_width=True,
                hide_index=True)

            
        with ct2:
            image = Image.open("images/AQI_c.jpeg")
            image_resized = image.resize((1000, 500))
            #st.image(image_resized, use_column_width=True)
            
            aqi_perf = data[data["Date"] == datetime.today().strftime('%Y-%m-%d')]
            
            
            category={"Good":"#33ff36", "Moderate":"#f76a24", "Poor":"#f72424", "Hazardous":"#a13f07"}
            def plot_aqi(data_frame):
                color = []
                aqi_range = []
                for value in data_frame['AQI_Indice'].to_list():
                    color.append(category[value])
                
                
                for j in color:
                    if j == "#33ff36":
                        aqi_range.append(random.choice(range(0,50)))
                        
                    elif j == "#f76a24":
                        aqi_range.append(random.choice(range(50,100)))
                        
                    elif j == "#f72424":
                        aqi_range.append(random.choice(range(100,200)))
                        
                    elif j == "#a13f07":
                        aqi_range.append(random.choice(range(400,500)))
                        
                return aqi_range
            
            aqi_range = plot_aqi(aqi_perf)
            
            st.subheader(":blue[**Visualisation AQI dans la journée**]")
            
            fig = px.bar(
                aqi_perf,
                x="Hour_brute",
                y=aqi_range,
                color="AQI_Indice",
                color_discrete_map=category,
                text_auto='.2s',
            )
            fig.update_layout(
                height=400,
                margin=dict(l=20, r=20, t=50, b=20),
                xaxis_title="",
                yaxis_title="AQI"
            )
            st.plotly_chart(fig, use_container_width=True)

            
            st.markdown("---")
            
            st.subheader(":blue[**AQI des jours passés**]")
            
            aqi_range2 = plot_aqi(data_pres)
            print(data_pres)
            
            fig2 = px.scatter(
                data_pres,
                x="Date",
                y=aqi_range2,
                color="AQI_Indice",
                color_discrete_map=category,
                hover_name="AQI_Indice",
                size_max=600
                
            )
            fig2.update_layout(
                height=400,
                margin=dict(l=20, r=20, t=20, b=20),
                xaxis_title="Jours passés",
                yaxis_title="AQI Day"
            )
            st.plotly_chart(fig2, use_container_width=True)

        
    with tab3:
        st.subheader(f":green[**Conseils Utiles pour Tous**]")
        
        categories = ["Asthme", "Problèmes Cardiaques", "Allergies", "Sinus", "Rhume/Grippe", "Chronique (BPCO)"]
        

        # Affichage des boutons de catégorie
        cols = st.tabs(categories)
        with cols[0]:
            categor = "Asthme"

            # Titre de la section
            st.markdown(f"## {categor}")
            
            # Information sur le risque
            st.markdown("Le risque de symptômes de Asthme est Bas lorsque l'AQI est Good (0-50)")
            
            # Recommandations dans deux colonnes
            c1, c2 = st.columns(2)
            
            with c1:
                st.markdown("<p class='section-title'>À faire :</p>", unsafe_allow_html=True)
                st.markdown("""
                <p><span class='check-icon'>✓</span> Surveillez la qualité de l'air pour planifier les activités en extérieur en toute sécurité.</p>
                <p><span class='check-icon'>✓</span> Buvez de l'eau pour garder les voies respiratoires humides et réduire l'irritation.</p>
                <p><span class='check-icon'>✓</span> Incluez des fruits et légumes non allergènes qui soutiennent la santé pulmonaire, comme des pommes, des poires et des épinards.</p>
                """, unsafe_allow_html=True)
            
            with c2:
                st.markdown("<p class='section-title'>À ne pas faire :</p>", unsafe_allow_html=True)
                st.markdown("""
                <p><span class='x-icon'>✗</span> Ne sautez pas les médicaments prescrits par votre médecin.</p>
                <p><span class='x-icon'>✗</span> Ne participez pas à des activités en extérieur tôt le matin pendant les heures de forte pollinisation.</p>
                """, unsafe_allow_html=True)
                
            # Symptômes légers
            st.markdown("<p>Symptômes légers tels que léger sifflement, toux occasionnelle et légère difficulté à respirer</p>", unsafe_allow_html=True)
            
        with cols[1]:
            categor = "Problèmes Cardiaques"

            # Titre de la section
            st.markdown(f"## {categor}")
            
            # Information sur le risque
            st.markdown("Le risque de symptômes de Problèmes Cardiaques est Bas lorsque l'AQI est Good (0-50)")
            
            # Recommandations dans deux colonnes
            c1, c2 = st.columns(2)
            
            with c1:
                st.markdown("<p class='section-title'>À faire :</p>", unsafe_allow_html=True)
                st.markdown("""
                <p><span class='check-icon'>✓</span> Vérifiez régulièrement l'indice de qualité de l'air et planifiez vos activités en conséquence.</p>
                <p><span class='check-icon'>✓</span> Maintenez une alimentation saine riche en fruits, légumes et acides gras oméga-3.</p>
                <p><span class='check-icon'>✓</span> Participez à des exercices modérés réguliers comme la marche pour une forme cardiovasculaire.</p>
                """, unsafe_allow_html=True)
            
            with c2:
                st.markdown("<p class='section-title'>À ne pas faire :</p>", unsafe_allow_html=True)
                st.markdown("""
                <p><span class='x-icon'>✗</span> Ignorez tout symptôme cardiaque inhabituel.</p>
                <p><span class='x-icon'>✗</span> Participez à des exercices de haute intensité sans échauffement approprié.</p>
                <p><span class='x-icon'>✗</span> Consommez des quantités excessives de caféine ou de boissons énergétiques.</p>
                """, unsafe_allow_html=True)
                
            # Symptômes légers
            st.markdown("<p>Symptômes légers tels que de légères palpitations cardiaques, une fatigue légère, un léger inconfort thoracique, etc</p>", unsafe_allow_html=True)
            
                # st.markdown(f'<div class="category-button active-button">{categories[i]}</div>', unsafe_allow_html=True)

        with cols[2]:
            categor = "Allergies"

            # Titre de la section
            st.markdown(f"## {categor}")
            
            # Information sur le risque
            st.markdown("Le risque de symptômes de Allergies est Bas lorsque l'AQI est Good (0-50)")
            
            # Recommandations dans deux colonnes
            c1, c2 = st.columns(2)
            
            with c1:
                st.markdown("<p class='section-title'>À faire :</p>", unsafe_allow_html=True)
                st.markdown("""
                <p><span class='check-icon'>✓</span> Vérifiez régulièrement la qualité de l'air et minimisez l'exposition aux allergènes.</p>
                <p><span class='check-icon'>✓</span> Buvez beaucoup d'eau et de liquides pour garder les voies respiratoires humides et réduire l'irritation.</p>
                <p><span class='check-icon'>✓</span> Consommez des aliments riches en acides gras oméga-3 pour réduire l'inflammation.</p>
                """, unsafe_allow_html=True)
            
            with c2:
                st.markdown("<p class='section-title'>À ne pas faire :</p>", unsafe_allow_html=True)
                st.markdown("""
                <p><span class='x-icon'>✗</span> Ignorez le nettoyage régulier.</p>
                <p><span class='x-icon'>✗</span> Exercez-vous à l'extérieur pendant les heures de pointe de pollen.</p>
                """, unsafe_allow_html=True)
                
            # Symptômes légers
            st.markdown("<p>Symptômes légers tels que des éternuements occasionnels, une légère toux, une légère congestion nasale et une irritation mineure de la gorge.</p>", unsafe_allow_html=True)
            
        with cols[3]:
            categor = "Sinus"

            # Titre de la section
            st.markdown(f"## {categor}")
            
            # Information sur le risque
            st.markdown("Le risque de symptômes de Sinus est Bas lorsque l'AQI est Good (0-50)")
            
            # Recommandations dans deux colonnes
            c1, c2 = st.columns(2)
            
            with c1:
                st.markdown("<p class='section-title'>À faire :</p>", unsafe_allow_html=True)
                st.markdown("""
                <p><span class='check-icon'>✓</span> Vérifiez régulièrement l'indice de qualité de l'air pour prendre des décisions.</p>
                <p><span class='check-icon'>✓</span> Évitez les déclencheurs d'allergies nasales.</p>
                <p><span class='check-icon'>✓</span> Restez hydraté et consommez des liquides chauds.</p>
                """, unsafe_allow_html=True)
            
            with c2:
                st.markdown("<p class='section-title'>À ne pas faire :</p>", unsafe_allow_html=True)
                st.markdown("""
                <p><span class='x-icon'>✗</span> Fumer ou vous exposer à la fumée extérieure.</p>
                <p><span class='x-icon'>✗</span> Nager dans une piscine ou pratiquer des activités connexes.</p>
                """, unsafe_allow_html=True)
                
            # Symptômes légers
            st.markdown("<p>Symptômes légers, tels qu'un nez qui coule, des douleurs dentaires, de la toux, des maux de tête et de la fatigue.</p>", unsafe_allow_html=True)
            
                # st.markdown(f'<div class="category-button active-button">{categories[i]}</div>', unsafe_allow_html=True)

        with cols[4]:
            categor = "Rhume/Grippe"

            # Titre de la section
            st.markdown(f"## {categor}")
            
            # Information sur le risque
            st.markdown("Le risque de symptômes de Rhume/Grippe est Bas lorsque l'AQI est Good (0-50)")
            
            # Recommandations dans deux colonnes
            c1, c2 = st.columns(2)
            
            with c1:
                st.markdown("<p class='section-title'>À faire :</p>", unsafe_allow_html=True)
                st.markdown("""
                <p><span class='check-icon'>✓</span> Surveillez l'AQI de votre région et de votre domicile régulièrement pour rester informé.</p>
                <p><span class='check-icon'>✓</span> Restez hydraté avec de l'eau et des tisanes pour aider à fluidifier le mucus.</p>
                <p><span class='check-icon'>✓</span> Maintenez une bonne hygiène des mains en vous lavant régulièrement les mains.</p>
                """, unsafe_allow_html=True)
            
            with c2:
                st.markdown("<p class='section-title'>À ne pas faire :</p>", unsafe_allow_html=True)
                st.markdown("""
                <p><span class='x-icon'>✗</span> N'ignorez pas le port d'un masque dans des lieux bondés.</p>
                <p><span class='x-icon'>✗</span> Négligez le nettoyage régulier des surfaces couramment touchées.</p>
                <p><span class='x-icon'>✗</span> Utilisez excessivement des sprays décongestionnants.</p>
                """, unsafe_allow_html=True)
                
            # Symptômes légers
            st.markdown("<p>Symptômes légers, tels que des éternuements, un nez qui coule, une toux, etc.</p>", unsafe_allow_html=True)
            
        with cols[5]:
            categor = "Chronique (BPCO)"

            # Titre de la section
            st.markdown(f"## {categor}")
            
            # Information sur le risque
            st.markdown("Le risque de symptômes de Chronique (BPCO) est Bas lorsque l'AQI est Good (0-50)")
            
            # Recommandations dans deux colonnes
            c1, c2 = st.columns(2)
            
            with c1:
                st.markdown("<p class='section-title'>À faire :</p>", unsafe_allow_html=True)
                st.markdown("""
                <p><span class='check-icon'>✓</span> Restez informé des niveaux d'AQI pour éviter l'exposition.</p>
                <p><span class='check-icon'>✓</span> Restez hydraté pour garder les voies respiratoires dégagées et réduire la production de mucus.</p>
                <p><span class='check-icon'>✓</span> Incorporez des aliments riches en antioxydants dans votre alimentation pour soutenir la santé pulmonaire.</p>
                """, unsafe_allow_html=True)
            
            with c2:
                st.markdown("<p class='section-title'>À ne pas faire :</p>", unsafe_allow_html=True)
                st.markdown("""
                <p><span class='x-icon'>✗</span> Négligez l'exercice régulier.</p>
                <p><span class='x-icon'>✗</span> Exposez-vous à la fumée en étant près de fumeurs.</p>
                """, unsafe_allow_html=True)
                
            # Symptômes légers
            st.markdown("<p>Symptômes légers tels que léger sifflement, toux occasionnelle et légère difficulté à respirer</p>", unsafe_allow_html=True)
            
                # st.markdown(f'<div class="category-button active-button">{categories[i]}</div>', unsafe_allow_html=True)
   
        
    print("ok")

    ### Tableau de fin

    st.subheader(f":green[**Données Historique sur la Qualité de l'Air**] \n :blue[**{region_filter}**]")
    if data_abidj.empty:
        st.markdown("Pas de données pour la sélection -----")
    else:

        data_abidj = data_abidj.replace(regex=',', value='.')
        data_abidj[["tmmn(degC)", "tmmx(degC)", "rmax(%)", "rmin(%)"]] = data_abidj[["tmmn(degC)", "tmmx(degC)", "rmax(%)", "rmin(%)"]].apply(pd.to_numeric)
        data_abidj["temp_moy"] =data_abidj["tmmx(degC)"]
        #(data_abidj["tmmn(degC)"] + data_abidj["tmmx(degC)"])/2
        data_abidj["hum_moy"] = (data_abidj["rmax(%)"] + data_abidj["rmin(%)"])/2
        
        data_passe = data_abidj.drop(["Year", "Month", "Day", "tmmn(degC)", "tmmx(degC)", "rmax(%)", "rmin(%)"], axis=1)
        
        data_pred = data_passe.drop(["Date"], axis=1)
        y_pred = perdiction(data_pred)
        data_passe["AQI_Indice"] = y_pred
        st.dataframe(data_passe, use_container_width=True, hide_index=True)


    # Ajouter le pied de page

    st.markdown(footer, unsafe_allow_html=True)
