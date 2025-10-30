import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
import os
import tempfile
import zipfile
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from datetime import datetime
import requests
from PIL import Image
import pickle

# -------------------------------
# Configuration de la page
# -------------------------------
st.set_page_config(
    page_title="Système de Détection de Pannes d'Avion",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------
# CSS personnalisé - Thème CCAA
# -------------------------------
st.markdown(f"""
<style>
    .stApp {{
        background: linear-gradient(to bottom, #00b0ef, #ffffff);
        color: #020202;
    }}
    .main-header {{
        color: #020202;
        text-shadow: 1px 1px 2px rgba(255,255,255,0.7);
        font-family: 'Arial Black', sans-serif;
        text-align: center;
        padding: 10px;
        background-color: #00b0ef;
        border-radius: 10px;
        border: 2px solid #ffffff;
        margin-bottom: 20px;
    }}
    .stButton>button {{
        background-color: #00b0ef;
        color: #ffffff;
        border-radius: 10px;
        border: 2px solid #ffffff;
        font-weight: bold;
        width: 100%;
    }}
    .stButton>button:hover {{
        background-color: #ffffff;
        color: #00b0ef;
        border: 2px solid #00b0ef;
    }}
    .alert-box {{
        background-color: #FF4B4B;
        color: white;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        border: 2px solid #ffffff;
        animation: pulse 2s infinite;
    }}
    .normal-box {{
        background-color: #00D26A;
        color: white;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        border: 2px solid #ffffff;
    }}
    .component-card {{
        background-color: rgba(255, 255, 255, 0.8);
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
        border: 1px solid #00b0ef;
        color: #020202;
    }}
    @keyframes pulse {{
        0% {{ opacity: 1; }}
        50% {{ opacity: 0.7; }}
        100% {{ opacity: 1; }}
    }}
    .sidebar .sidebar-content {{
        background: linear-gradient(to bottom, #00b0ef, #ffffff);
        color: #020202;
    }}
    h1, h2, h3, h4, h5, h6 {{
        color: #020202;
        text-shadow: 1px 1px 2px rgba(255,255,255,0.7);
    }}
    .metric-card {{
        background-color: rgba(255, 255, 255, 0.8);
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #00b0ef;
        text-align: center;
        color: #020202;
    }}
    .css-1d391kg, .css-1d391kg p, .css-1d391kg div {{
        color: #020202;
    }}
    .css-1d391kg h1, .css-1d391kg h2, .css-1d391kg h3, .css-1d391kg h4, .css-1d391kg h5, .css-1d391kg h6 {{
        color: #020202;
    }}
    .stRadio > div {{
        background-color: rgba(255, 255, 255, 0.8);
        padding: 10px;
        border-radius: 10px;
    }}
    .stRadio label {{
        color: #020202;
        font-weight: bold;
    }}
    .stRadio div[role="radiogroup"] {{
        background-color: rgba(255, 255, 255, 0.8);
    }}
    .chat-message {{
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
    }}
    .chat-message.user {{
        background-color: #00b0ef;
        color: white;
    }}
    .chat-message.assistant {{
        background-color: #f0f0f0;
        color: #020202;
    }}
</style>
""", unsafe_allow_html=True)

# -------------------------------
# Fonctions utilitaires
# -------------------------------
def load_ccaa_logo():
    try:
        # Charger le logo depuis le dossier assets
        return Image.open("assets/ccaa_logo.png")
    except:
        # Fallback vers un logo d'avion si le logo CCAA n'est pas trouvé
        return "https://img.icons8.com/color/96/000000/airplane-tail.png"

def prepare_data(df, drop_cols=True):
    index_columns_names = ["UnitNumber","Cycle"]
    operational_settings_columns_names = ["OpSet"+str(i) for i in range(1,4)]
    sensor_measure_columns_names = ["SensorMeasure"+str(i) for i in range(1,22)]
    cols_to_drop = ['OpSet3', 'SensorMeasure1', 'SensorMeasure5', 'SensorMeasure6', 
                    'SensorMeasure10','SensorMeasure14', 'SensorMeasure16', 'SensorMeasure18', 'SensorMeasure19']
    
    if drop_cols:
        df = df.drop(cols_to_drop, axis=1, errors='ignore')
    
    if 'RUL' not in df.columns:
        try:
            rul = pd.DataFrame(df.groupby('UnitNumber')['Cycle'].max()).reset_index()
            rul.columns = ['UnitNumber', 'max']
            df = df.merge(rul, on=['UnitNumber'], how='left')
            df['RUL'] = df['max'] - df['Cycle']
            df.drop('max', axis=1, inplace=True)
        except:
            df['RUL'] = np.nan
    return df

def gen_test(id_df, seq_length, seq_cols, mask_value):
    df_mask = pd.DataFrame(np.zeros((seq_length-1, id_df.shape[1])), columns=id_df.columns)
    df_mask[:] = mask_value    
    id_df1 = pd.concat([df_mask, id_df], ignore_index=True)    
    data_array = id_df1[seq_cols].values
    lstm_array = []
    start = len(data_array)-seq_length
    stop = len(data_array)
    lstm_array.append(data_array[start:stop, :])    
    return np.array(lstm_array)

def _row_features(arr: np.ndarray) -> dict:
    if arr is None or len(arr)==0:
        return {f: np.nan for f in ['mean','std','min','max','median','p10','p90','iqr','skew','kurt','rms','energy','zcr']}
    mean_v = float(np.mean(arr))
    std_v  = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
    min_v  = float(np.min(arr))
    max_v  = float(np.max(arr))
    med_v  = float(np.median(arr))
    p10_v  = float(np.percentile(arr,10))
    p90_v  = float(np.percentile(arr,90))
    iqr_v  = float(np.percentile(arr,75)-np.percentile(arr,25))
    skew_v = float(pd.Series(arr).skew())
    kurt_v = float(pd.Series(arr).kurtosis())
    rms_v  = float(np.sqrt(np.mean(arr**2)))
    energy_v = float(np.sum(arr**2))
    zcr_v = float(np.sum(np.sign(arr-mean_v)[:-1]*np.sign(arr-mean_v)[1:]<0)/max(1,len(arr)-1))
    return {'mean':mean_v,'std':std_v,'min':min_v,'max':max_v,'median':med_v,
            'p10':p10_v,'p90':p90_v,'iqr':iqr_v,'skew':skew_v,'kurt':kurt_v,
            'rms':rms_v,'energy':energy_v,'zcr':zcr_v}

def build_features(df_long: pd.DataFrame) -> pd.DataFrame:
    """Construit les features par cycle/sensor puis pivot large"""
    # Optimisation pour éviter les problèmes de mémoire
    time_cols = [c for c in df_long.columns if c not in ['cycle','sensor']]
    feats = []
    
    # Échantillonnage pour les grands datasets
    sample_size = min(1000, len(df_long))
    if len(df_long) > 1000:
        df_sample = df_long.sample(sample_size, random_state=42)
    else:
        df_sample = df_long
    
    for _, row in df_sample.iterrows():
        arr = pd.to_numeric(row[time_cols], errors='coerce').fillna(0).values
        f = _row_features(arr)
        f['cycle'] = int(row['cycle'])
        f['sensor'] = str(row['sensor'])
        feats.append(f)
    
    feats_df = pd.DataFrame(feats).set_index(['cycle','sensor'])
    wide = feats_df.unstack('sensor')
    wide.columns = [f"{col[1]}__{col[0]}" for col in wide.columns]
    wide = wide.reset_index()
    return wide

def read_txt_gz(fp):
    try:
        data = np.loadtxt(fp, delimiter='\t')
        df = pd.DataFrame(data)
        df['sensor'] = os.path.basename(fp).split('.')[0]
        df['cycle'] = df.index.values
        return df
    except Exception as e:
        st.warning(f"Impossible de lire {fp}: {e}")
        return None

def reconstruct_dataset_from_folder(folder_path):
    dfs = []
    for f in os.listdir(folder_path):
        if f.endswith('.txt') or f.endswith('.txt.gz'):
            path = os.path.join(folder_path, f)
            df = read_txt_gz(path)
            if df is not None:
                dfs.append(df)
    if len(dfs) == 0:
        st.error("Aucun fichier valide trouvé dans le dossier.")
        return None
    return pd.concat(dfs, ignore_index=True)

def create_sequences(data, seq_len=10):
    """Crée des séquences glissantes pour LSTM"""
    sequences = []
    for i in range(len(data) - seq_len + 1):
        sequences.append(data[i:i+seq_len])
    return np.array(sequences)

# -------------------------------
# Chargement des modèles
# -------------------------------
@st.cache_resource
def load_engine_models():
    try:
        scaler = joblib.load('models/scaler.pkl')
        feature_extractor = load_model('models/feature_extractor.h5')
        svm_classifier = joblib.load('models/svm_classifier.pkl')
        return scaler, feature_extractor, svm_classifier
    except Exception as e:
        st.error(f"Erreur lors du chargement des modèles moteur: {str(e)}")
        return None, None, None

@st.cache_resource
def load_hydraulic_models():
    try:
        models = {target: joblib.load(f'models/rf_{target}.joblib')
                  for target in ['cooler_condition','valve_condition',
                                 'internal_pump_leakage','hydraulic_accumulator',
                                 'stable_flag']}
        feature_cols = joblib.load('models/feature_columns.joblib')
        median_values = joblib.load('models/median_values.joblib')
        rare_class_mapping = joblib.load('models/rare_class_mapping.joblib')
        return models, feature_cols, median_values, rare_class_mapping
    except Exception as e:
        st.error(f"Erreur lors du chargement des modèles hydrauliques: {str(e)}")
        return None, None, None, None

@st.cache_resource
def load_apu_models():
    try:
        # KMeans PySpark (params)
        with open("apu_models/kmeans_model.pkl", "rb") as f:
            kmeans_params = pickle.load(f)
        
        # Scaler PySpark
        with open("apu_models/spark_scaler.pkl", "rb") as f:
            spark_scaler_params = pickle.load(f)
        
        # Colonnes ANALOG_SENSORS
        with open("apu_models/analog_sensors.pkl", "rb") as f:
            analog_sensors = pickle.load(f)
        
        # LSTM Autoencoder
        lstm_model = load_model("apu_models/lstm_autoencoder.h5")
        
        # Scaler scikit-learn
        sklearn_scaler = joblib.load("apu_models/sklearn_scaler.joblib")
        
        # Threshold
        with open("apu_models/threshold.pkl", "rb") as f:
            threshold = pickle.load(f)
        
        return kmeans_params, spark_scaler_params, analog_sensors, lstm_model, sklearn_scaler, threshold
    except Exception as e:
        st.error(f"Erreur lors du chargement des modèles APU: {str(e)}")
        return None, None, None, None, None, None

# -------------------------------
# Navigation
# -------------------------------
def main():
    # Initialisation de l'état de la page
    if "selected_page" not in st.session_state:
        st.session_state.selected_page = "Accueil"

    # Sidebar avec navigation
    with st.sidebar:
        logo = load_ccaa_logo()
        if isinstance(logo, Image.Image):
            st.image(logo, width=300)  # Agrandir le logo
        else:
            st.image(logo, width=300)
        
        st.markdown("---")
        st.markdown("<h2 style='text-align: center; color: #020202;'>Navigation</h2>", unsafe_allow_html=True)
        
        # Menu de navigation stylisé
        page_options = ["Accueil", "Moteurs", "Système Hydraulique", "APU", "Chatbot"]
        page_icons = ["🏠", "🔧", "💧", "✈️", "🤖"]
        
        selected_page = st.radio(
            "Sélectionnez une page:",
            page_options,
            index=page_options.index(st.session_state.selected_page),
            format_func=lambda x: f"{page_icons[page_options.index(x)]} {x}"
        )
        
        # Mettre à jour la page sélectionnée
        st.session_state.selected_page = selected_page
        
        st.markdown("---")
        st.markdown("### Statut du système")
        st.info("Système défaillant ⚠️")
        st.markdown(f"Dernière mise à jour: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Informations supplémentaires
        st.markdown("---")
        st.markdown("### 📞 Support technique")
        st.markdown("Contactez-nous en cas de problème:")
        st.markdown("- Email: support@ccaa.cm")
        st.markdown("- Téléphone: +237 658 578 534")
    
    # Affichage de la page sélectionnée
    if st.session_state.selected_page == "Accueil":
        show_home()
    elif st.session_state.selected_page == "Moteurs":
        show_engines()
    elif st.session_state.selected_page == "Système Hydraulique":
        show_hydraulic()
    elif st.session_state.selected_page == "APU":
        show_apu()
    elif st.session_state.selected_page == "Chatbot":
        show_chatbot()

# -------------------------------
# Page d'accueil
# -------------------------------
def show_home():
    
    st.markdown("<h1 class='main-header'>✈️ Système de Détection d'Anomalies pour Avions</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center; color: #020202;'>Base Aérienne Virtuelle - Surveillance en Temps Réel</h3>", unsafe_allow_html=True)
    
    # Section d'introduction
    col1, col2, col3 = st.columns([2, 1, 2])
    with col1:
        st.markdown("""
        <div style='background-color: rgba(255, 255, 255, 0.8); padding: 20px; border-radius: 10px; color: #020202;'>
            <h4 style='color: #020202;'>Bienvenue dans le système de détection d'anomalies</h4>
            <p style='color: #020202;'>Cette application permet de surveiller en temps réel l'état des composants critiques de l'avion:</p>
            <ul style='color: #020202;'>
                <li>Moteurs</li>
                <li>Système hydraulique</li>
                <li>Unités de puissance auxiliaire (APU)</li>
            </ul>
            <p style='color: #020202;'>Utilisez le menu de navigation pour accéder aux différentes sections.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        # Logo ou image d'avion
        st.image("https://www.agenceafrique.com/wp-content/uploads/2019/12/camair.jpg", width=650)
    
    # Section d'état global
    st.markdown("---")
    st.markdown("<h2 style='text-align: center; color: #020202;'>État Global de l'Avion</h2>", unsafe_allow_html=True)
    
    # Charger les données d'exemple pour l'accueil
    engine_status, hydraulic_status, apu_status = load_demo_data()
    
    # Cartes d'état des composants
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("<h4 style='text-align: center; color: #020202;'>Moteurs</h4>", unsafe_allow_html=True)
        if engine_status["alert"]:
            st.markdown(f"<div class='alert-box'><h5>⚠️ ALERTE</h5><p>{engine_status['message']}</p></div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='normal-box'><h5>✅ NORMAL</h5><p>{engine_status['message']}</p></div>", unsafe_allow_html=True)
        
        # Métriques moteur
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.metric("RUL Moyen", f"{engine_status['rul_avg']} cycles")
        st.metric("Anomalies", f"{engine_status['anomaly_count']}")
        st.markdown("</div>", unsafe_allow_html=True)
        
        if engine_status["alert"]:
            if st.button("🔍 Voir détails des moteurs", key="engine_btn"):
                st.session_state.selected_page = "Moteurs"
                st.experimental_rerun()
    
    with col2:
        st.markdown("<h4 style='text-align: center; color: #020202;'>Système Hydraulique</h4>", unsafe_allow_html=True)
        if hydraulic_status["alert"]:
            st.markdown(f"<div class='alert-box'><h5>⚠️ ALERTE</h5><p>{hydraulic_status['message']}</p></div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='normal-box'><h5>✅ NORMAL</h5><p>{hydraulic_status['message']}</p></div>", unsafe_allow_html=True)
        
        # Métriques hydrauliques
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.metric("Condition du refroidisseur", f"{hydraulic_status['cooler_condition']}%")
        st.metric("Condition de la valve", f"{hydraulic_status['valve_condition']}%")
        st.markdown("</div>", unsafe_allow_html=True)
        
        if hydraulic_status["alert"]:
            if st.button("🔍 Voir détails du système hydraulique", key="hydraulic_btn"):
                st.session_state.selected_page = "Système Hydraulique"
                st.experimental_rerun()
    
    with col3:
        st.markdown("<h4 style='text-align: center; color: #020202;'>APU</h4>", unsafe_allow_html=True)
        if apu_status["alert"]:
            st.markdown(f"<div class='alert-box'><h5>⚠️ ALERTE</h5><p>{apu_status['message']}</p></div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='normal-box'><h5>✅ NORMAL</h5><p>{apu_status['message']}</p></div>", unsafe_allow_html=True)
        
        # Métriques APU
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.metric("Score d'anomalie", f"{apu_status['anomaly_score']:.2f}")
        st.metric("Clusters anormaux", f"{apu_status['abnormal_clusters']}")
        st.markdown("</div>", unsafe_allow_html=True)
        
        if apu_status["alert"]:
            if st.button("🔍 Voir détails des APU", key="apu_btn"):
                st.session_state.selected_page = "APU"
                st.experimental_rerun()
    
    # Graphique de synthèse
    st.markdown("---")
    st.markdown("<h3 style='text-align: center; color: #020202;'>Synthèse des Anomalies Détectées</h3>", unsafe_allow_html=True)
    
    # Données pour le graphique
    components = ['Moteurs', 'Système Hydraulique', 'APU']
    anomaly_counts = [engine_status['anomaly_count'], hydraulic_status['anomaly_count'], apu_status['anomaly_count']]
    alert_status = [engine_status['alert'], hydraulic_status['alert'], apu_status['alert']]
    
    colors = ['#FF4B4B' if alert else '#00D26A' for alert in alert_status]
    
    fig = go.Figure(data=[
        go.Bar(x=components, y=anomaly_counts, marker_color=colors)
    ])
    
    fig.update_layout(
        title="Nombre d'anomalies par composant",
        xaxis_title="Composants",
        yaxis_title="Nombre d'anomalies",
        template="plotly_white",
        font=dict(color="#020202")
    )
    
    st.plotly_chart(fig, use_container_width=True)

def load_demo_data():
    # Données d'exemple pour l'accueil
    engine_status = {
        "alert": True,
        "message": "2 moteurs présentent des anomalies critiques",
        "rul_avg": 45,
        "anomaly_count": 2
    }
    
    hydraulic_status = {
        "alert": False,
        "message": "Tous les composants hydrauliques fonctionnent normalement",
        "cooler_condition": 92,
        "valve_condition": 88,
        "anomaly_count": 0
    }
    
    apu_status = {
        "alert": True,
        "message": "Détection d'anomalies sur l'APU principal",
        "anomaly_score": 0.87,
        "abnormal_clusters": 3,
        "anomaly_count": 1
    }
    
    return engine_status, hydraulic_status, apu_status

# -------------------------------
# Page Moteurs
# -------------------------------
def show_engines():
            
    st.markdown("<h1 class='main-header'>🔧 Analyse des Moteurs</h1>", unsafe_allow_html=True)
    
    # Charger les modèles
    scaler, feature_extractor, svm_classifier = load_engine_models()
    
    if scaler is None or feature_extractor is None or svm_classifier is None:
        st.error("Impossible de charger les modèles de moteurs. Vérifiez les fichiers de modèles.")
        return
    
    # Section d'upload
    st.markdown("### 📤 Téléversement des données moteur")
    uploaded_file = st.file_uploader("Chargez les données moteur (TXT ou CSV)", type=['txt', 'csv'], key="engine_upload")
    
    if uploaded_file:
        try:
            with st.spinner("Traitement des données..."):
                df_new = pd.read_csv(uploaded_file, delim_whitespace=True, 
                                     names=["UnitNumber","Cycle"] + ["OpSet"+str(i) for i in range(1,4)] + ["SensorMeasure"+str(i) for i in range(1,22)])
                
                # Prétraitement
                df_new = prepare_data(df_new)
                feats = [col for col in df_new.columns if col not in ['UnitNumber', 'Cycle', 'RUL']]
                df_new[feats] = scaler.transform(df_new[feats])
                sequence_length = 50
                mask_value = 0
                x_new = np.concatenate([gen_test(df_new[df_new['UnitNumber']==unit], sequence_length, feats, mask_value) for unit in df_new['UnitNumber'].unique()])
            
            # Prédiction
            with st.spinner("Analyse en cours..."):
                cnn_features = feature_extractor.predict(x_new)
                y_pred = svm_classifier.predict(cnn_features)
                y_prob = svm_classifier.predict_proba(cnn_features)[:,1] if hasattr(svm_classifier, "predict_proba") else np.array([0.5]*len(y_pred))
            
            # Résultats
            st.markdown("---")
            st.markdown("## 📊 Résultats de l'analyse")
            
            # Compteur d'alertes
            alert_count = sum(y_pred)
            total_units = len(y_pred)
            
            if alert_count > 0:
                st.markdown(f"<div class='alert-box'><h3>🚨 {alert_count} ALERTE(S) CRITIQUE(S) DÉTECTÉE(S)</h3></div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='normal-box'><h3>✅ AUCUNE ALERTE CRITIQUE DÉTECTÉE</h3></div>", unsafe_allow_html=True)
            
            # Métriques
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Unités analysées", total_units)
            with col2:
                st.metric("Alertes critiques", alert_count)
            with col3:
                st.metric("Taux d'anomalie", f"{(alert_count/total_units)*100:.1f}%")
            
            # Détails des alertes
            st.markdown("### 📋 Détails des unités")
            results_df = pd.DataFrame({
                'Unité': df_new['UnitNumber'].unique(),
                'Statut': ['CRITIQUE' if p == 1 else 'NORMAL' for p in y_pred],
                'Probabilité Anomalie': [f"{p:.2%}" for p in y_prob],
                'RUL Estimé Min': [df_new[df_new['UnitNumber']==u]['RUL'].min() if 'RUL' in df_new.columns and not pd.isna(df_new['RUL']).all() else "N/A" for u in df_new['UnitNumber'].unique()]
            })
            
            # Style the dataframe
            def color_critical(val):
                color = 'red' if val == 'CRITIQUE' else 'green'
                return f'color: {color}; font-weight: bold'
            
            styled_df = results_df.style.applymap(color_critical, subset=['Statut'])
            st.dataframe(styled_df)
            
            # Export des résultats
            st.markdown("### 💾 Export des résultats")
            csv = results_df.to_csv(index=False)
            st.download_button(
                label="📥 Télécharger les résultats en CSV",
                data=csv,
                file_name="resultats_moteurs.csv",
                mime="text/csv"
            )
            
            # Visualisations
            st.markdown("---")
            st.markdown("## 📈 Visualisations")
            
            # Graphique 1: Distribution des probabilités d'anomalie
            fig1 = px.histogram(
                x=y_prob, 
                nbins=20,
                title="Distribution des probabilités d'anomalie",
                labels={'x': 'Probabilité d\'anomalie', 'y': 'Nombre d\'unités'}
            )
            fig1.update_layout(template="plotly_white", font=dict(color="#020202"))
            st.plotly_chart(fig1, use_container_width=True)
            
            # Graphique 2: RUL vs Probabilité d'anomalie
            if 'RUL' in df_new.columns and not pd.isna(df_new['RUL']).all():
                rul_values = [df_new[df_new['UnitNumber']==u]['RUL'].min() for u in df_new['UnitNumber'].unique()]
                fig2 = px.scatter(
                    x=rul_values, 
                    y=y_prob,
                    color=y_pred,
                    title="RUL vs Probabilité d'anomalie",
                    labels={'x': 'RUL (Cycles restants)', 'y': 'Probabilité d\'anomalie'},
                    color_discrete_map={0: 'green', 1: 'red'}
                )
                fig2.update_layout(template="plotly_white", font=dict(color="#020202"))
                st.plotly_chart(fig2, use_container_width=True)
            
            # Graphique 3: Évolution des capteurs pour une unité critique
            if alert_count > 0:
                critical_unit = df_new['UnitNumber'].unique()[np.where(y_pred == 1)[0][0]]
                st.markdown(f"### 🔍 Évolution des capteurs - Unité {critical_unit} (CRITIQUE)")
                
                # Sélectionner un capteur à visualiser
                sensor_options = [col for col in feats if col.startswith('SensorMeasure')]
                selected_sensor = st.selectbox("Sélectionnez un capteur à visualiser", sensor_options)
                
                if selected_sensor:
                    fig3 = px.line(
                        df_new[df_new['UnitNumber'] == critical_unit],
                        x='Cycle',
                        y=selected_sensor,
                        title=f"Évolution du {selected_sensor} - Unité {critical_unit}",
                        labels={'Cycle': 'Cycle', selected_sensor: 'Valeur normalisée'}
                    )
                    fig3.update_layout(template="plotly_white", font=dict(color="#020202"))
                    st.plotly_chart(fig3, use_container_width=True)
        
        except Exception as e:
            st.error(f"Erreur lors du traitement : {str(e)}")
    else:
        st.info("Veuillez uploader un fichier TXT ou CSV pour commencer l'analyse.")

# -------------------------------
# Page Système Hydraulique
# -------------------------------
def show_hydraulic():
            
    st.markdown("<h1 class='main-header'>💧 Analyse du Système Hydraulique</h1>", unsafe_allow_html=True)
    
    # Charger les modèles
    models, feature_cols, median_values, rare_class_mapping = load_hydraulic_models()
    
    if models is None:
        st.error("Impossible de charger les modèles hydrauliques. Vérifiez les fichiers de modèles.")
        return
    
    # Section d'upload
    st.markdown("### 📤 Téléversement des données hydrauliques")
    uploaded_file = st.file_uploader("Upload ZIP folder with .txt.gz files", type=['zip'], key="hydraulic_upload")
    
    if uploaded_file is not None:
        with tempfile.TemporaryDirectory() as tmpdirname:
            # Décompression ZIP
            with zipfile.ZipFile(uploaded_file, 'r') as zip_ref:
                zip_ref.extractall(tmpdirname)
            st.success("✅ Fichiers extraits. Reconstruction du dataset...")
            
            df_long = reconstruct_dataset_from_folder(tmpdirname)
            if df_long is None:
                st.stop()
            
            st.info("✅ Dataset reconstruit. Calcul des features...")
            X_new = build_features(df_long)
            
            if X_new is None or len(X_new) == 0:
                st.error("Erreur lors de la construction des features.")
                return
            
            # Remplissage NaN et fusion classes rares
            X_new = X_new.reindex(columns=feature_cols, fill_value=0)
            
            st.info("✅ Prédictions en cours...")
            predictions = {target: clf.predict(X_new) for target, clf in models.items()}
            df_pred = pd.DataFrame(predictions)
            
            # Résultats
            st.markdown("---")
            st.markdown("## 📊 Résultats de l'analyse")
            
            # Vérification des alertes
            alerts = []
            if (df_pred['cooler_condition'] < 20).any():
                alerts.append("❌ Condition du refroidisseur critique")
            if (df_pred['valve_condition'] < 80).any():
                alerts.append("❌ Condition de la valve critique")
            if (df_pred['internal_pump_leakage'] > 1).any():
                alerts.append("❌ Fuite interne pompe élevée")
            if (df_pred['hydraulic_accumulator'] < 100).any():
                alerts.append("❌ Pression accumulateur hydraulique faible")
            if (df_pred['stable_flag'] == 1).any():
                alerts.append("⚠️ Conditions potentiellement instables")
            
            if alerts:
                st.markdown("<div class='alert-box'><h3>🚨 ALERTES DÉTECTÉES</h3></div>", unsafe_allow_html=True)
                for alert in alerts:
                    st.error(alert)
            else:
                st.markdown("<div class='normal-box'><h3>✅ AUCUNE ALERTE CRITIQUE DÉTECTÉE</h3></div>", unsafe_allow_html=True)
            
            # Affichage des prédictions
            st.markdown("### 📋 Détails des conditions des composants")
            st.dataframe(df_pred)
            
            # Export des résultats
            st.markdown("### 💾 Export des résultats")
            csv = df_pred.to_csv(index=False)
            st.download_button(
                label="📥 Télécharger les résultats en CSV",
                data=csv,
                file_name="resultats_hydraulique.csv",
                mime="text/csv"
            )
            
            # Visualisations
            st.markdown("---")
            st.markdown("## 📈 Visualisations")
            
            # Graphique 1: Évolution des conditions
            fig = make_subplots(
                rows=len(df_pred.columns), 
                cols=1,
                subplot_titles=df_pred.columns.tolist()
            )
            
            for i, col in enumerate(df_pred.columns):
                fig.add_trace(
                    go.Scatter(
                        y=df_pred[col].values, 
                        mode='lines+markers',
                        name=col
                    ),
                    row=i+1, col=1
                )
            
            fig.update_layout(
                height=800,
                title_text="Évolution des conditions des composants",
                template="plotly_white",
                font=dict(color="#020202")
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Graphique 2: Matrice de corrélation des conditions
            corr = df_pred.corr()
            fig2 = px.imshow(
                corr,
                text_auto=True,
                aspect="auto",
                title="Corrélation entre les conditions des composants",
                color_continuous_scale='RdBu_r'
            )
            fig2.update_layout(template="plotly_white", font=dict(color="#020202"))
            st.plotly_chart(fig2, use_container_width=True)

# -------------------------------
# Page APU
# -------------------------------
def show_apu():
            
    st.markdown("<h1 class='main-header'>✈️ Analyse des Unités de Puissance Auxiliaire (APU)</h1>", unsafe_allow_html=True)
    
    # Charger les modèles
    kmeans_params, spark_scaler_params, ANALOG_SENSORS, lstm_model, sklearn_scaler, threshold = load_apu_models()
    
    if kmeans_params is None:
        st.error("Impossible de charger les modèles APU. Vérifiez les fichiers de modèles.")
        return
    
    # Section d'upload
    st.markdown("### 📤 Téléversement des données APU")
    uploaded_file = st.file_uploader("CSV contenant les colonnes APU", type="csv", key="apu_upload")
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        
        # Vérifier colonnes
        missing_cols = [c for c in ANALOG_SENSORS if c not in df.columns]
        if missing_cols:
            st.error(f"Colonnes manquantes dans le CSV : {missing_cols}")
        else:
            st.success("✅ Colonnes vérifiées !")
            
            # Normalisation PySpark pour KMeans
            X_kmeans = df[ANALOG_SENSORS].copy()
            mean = np.array(spark_scaler_params["mean"])
            std = np.array(spark_scaler_params["std"])
            X_scaled_kmeans = (X_kmeans - mean) / std
            
            # KMeans prediction (via centres)
            centers = np.array(kmeans_params["centers"])
            def predict_kmeans(X):
                distances = np.linalg.norm(X[:, None] - centers[None, :], axis=2)
                return np.argmin(distances, axis=1)
            kmeans_labels = predict_kmeans(X_scaled_kmeans.values)
            df["Cluster_KMeans"] = kmeans_labels
            
            # Préparation pour LSTM
            X_scaled = sklearn_scaler.transform(df[ANALOG_SENSORS])
            sequence_length = 10
            X_lstm = create_sequences(X_scaled, sequence_length)
            
            # Prédiction LSTM
            X_pred = lstm_model.predict(X_lstm, verbose=0)
            mse = np.mean(np.power(X_lstm - X_pred, 2), axis=(1, 2))
            
            # Ajouter MSE et anomalie à df original
            mse_full = np.concatenate([np.full(sequence_length-1, np.nan), mse])
            df["MSE_LSTM"] = mse_full
            df["Anomaly_LSTM"] = df["MSE_LSTM"] > threshold
            
            # Résultats
            st.markdown("---")
            st.markdown("## 📊 Résultats de l'analyse")
            
            total = len(df)
            anomalies = df["Anomaly_LSTM"].sum()
            pct_anomalies = anomalies / total * 100
            
            if anomalies > 0:
                st.markdown(f"<div class='alert-box'><h3>🚨 {anomalies} ANOMALIE(S) DÉTECTÉE(S)</h3></div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='normal-box'><h3>✅ AUCUNE ANOMALIE DÉTECTÉE</h3></div>", unsafe_allow_html=True)
            
            # Métriques
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Enregistrements analysés", total)
            with col2:
                st.metric("Anomalies détectées", anomalies)
            with col3:
                st.metric("Taux d'anomalie", f"{pct_anomalies:.2f}%")
            
            # Détails des anomalies
            st.markdown("### 📋 Détails des anomalies")
            st.dataframe(df[df["Anomaly_LSTM"]].reset_index(drop=True))
            
            # Export des résultats
            st.markdown("### 💾 Export des résultats")
            csv = df[df["Anomaly_LSTM"]].to_csv(index=False)
            st.download_button(
                label="📥 Télécharger les anomalies en CSV",
                data=csv,
                file_name="anomalies_apu.csv",
                mime="text/csv"
            )
            
            # Visualisations
            st.markdown("---")
            st.markdown("## 📈 Visualisations")
            
            # Graphique 1: Erreur de reconstruction LSTM
            fig1 = px.line(
                df, 
                y="MSE_LSTM", 
                title="Erreur de reconstruction LSTM",
                labels={"index": "Index", "MSE_LSTM": "Erreur quadratique moyenne"}
            )
            fig1.add_hline(y=threshold, line_dash="dash", line_color="red", annotation_text="Seuil d'anomalie")
            fig1.update_layout(template="plotly_white", font=dict(color="#020202"))
            st.plotly_chart(fig1, use_container_width=True)
            
            # Graphique 2: Répartition des clusters KMeans
            fig2 = px.histogram(
                df, 
                x="Cluster_KMeans", 
                title="Répartition des clusters KMeans",
                labels={"Cluster_KMeans": "Cluster", "count": "Nombre d'occurrences"}
            )
            fig2.update_layout(template="plotly_white", font=dict(color="#020202"))
            st.plotly_chart(fig2, use_container_width=True)
            
            # Graphique 3: Visualisation des clusters par capteur
            st.markdown("### 🔍 Visualisation des clusters par capteur")
            sensor_to_plot = st.selectbox("Choisir un capteur", ANALOG_SENSORS, index=0)
            fig3 = px.scatter(
                df, 
                x=df.index, 
                y=sensor_to_plot, 
                color="Cluster_KMeans",
                title=f"{sensor_to_plot} - Clusters KMeans",
                labels={"index": "Index", sensor_to_plot: "Valeur du capteur", "Cluster_KMeans": "Cluster"}
            )
            fig3.update_layout(template="plotly_white", font=dict(color="#020202"))
            st.plotly_chart(fig3, use_container_width=True)
            
            st.success("Analyse terminée ! ✅")

# -------------------------------
# Page Chatbot
# -------------------------------
def show_chatbot():
    st.markdown("<h1 class='main-header'>🤖 Assistant Virtuel - Détection de Pannes</h1>", unsafe_allow_html=True)
    
    st.markdown("""
    <div style='background-color: rgba(255, 255, 255, 0.8); padding: 20px; border-radius: 10px; margin-bottom: 20px; color: #020202;'>
        <p>Bienvenue dans l'assistant virtuel pour la détection de pannes d'avion.</p>
        <p>Je peux vous aider à:</p>
        <ul>
            <li>Comprendre les alertes et anomalies détectées</li>
            <li>Interpréter les résultats d'analyse</li>
            <li>Expliquer le fonctionnement des différents composants</li>
            <li>Vous guider dans l'utilisation de cette application</li>
        </ul>
        <p><strong>Note:</strong> Cette version utilise une simulation. L'intégration avec l'API Grok sera ajoutée ultérieurement.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialisation de l'historique de chat
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []
    
    # Affichage de l'historique des messages
    for message in st.session_state.chat_messages:
        if message["role"] == "user":
            st.markdown(f"""
            <div style='display: flex; justify-content: flex-end; margin: 10px 0;'>
                <div style='background-color: #00b0ef; color: white; padding: 10px 15px; border-radius: 15px; max-width: 70%;'>
                    {message["content"]}
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style='display: flex; justify-content: flex-start; margin: 10px 0;'>
                <div style='background-color: #f0f0f0; color: #020202; padding: 15px; border-radius: 15px; max-width: 70%; border-left: 4px solid #00b0ef;'>
                    {message["content"]}
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    # Entrée de l'utilisateur
    user_input = st.text_input("Posez votre question sur la détection de pannes...", key="chat_input")
    
    # Bouton d'envoi
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        send_button = st.button("Envoyer", key="send_button", use_container_width=True)
    
    # Traitement de la question
    if send_button and user_input:
        # Ajouter le message de l'utilisateur à l'historique
        st.session_state.chat_messages.append({"role": "user", "content": user_input})
        
        # Simulation de réponse
        with st.spinner("Réflexion..."):
            # Simulation de délai
            time.sleep(1)
            
            # Réponses prédéfinies selon le contexte
            if "moteur" in user_input.lower():
                response = """
                **Les moteurs d'avion** sont surveillés en analysant les données de capteurs qui mesurent divers paramètres comme la température, la pression et les vibrations. 
                
                **Indicateurs clés:**
                - RUL (Remaining Useful Life): Estimation du nombre de cycles restants avant maintenance
                - Probabilité d'anomalie: Score indiquant le risque de défaillance
                
                **Alertes courantes:**
                - Baisse anormale de performance
                - Augmentation des vibrations
                - Température anormale
                
                Pour plus de détails, consultez la page **Moteurs** de cette application.
                """
            elif "hydraulique" in user_input.lower():
                response = """
                **Le système hydraulique** est crucial pour le fonctionnement des commandes de vol. Il est composé de:
                
                **Composants surveillés:**
                - Refroidisseur (cooler)
                - Valve
                - Pompe interne
                - Accumulateur hydraulique
                
                **Indicateurs de santé:**
                - Condition du refroidisseur (%)
                - Condition de la valve (%)
                - Niveau de fuite de la pompe
                - Pression de l'accumulateur
                
                Pour plus de détails, consultez la page **Système Hydraulique** de cette application.
                """
            elif "apu" in user_input.lower() or "unités de puissance" in user_input.lower():
                response = """
                **L'APU (Auxiliary Power Unit)** fournit l'énergie auxiliaire lorsque les moteurs principaux sont arrêtés.
                
                **Surveillance:**
                - Analyse des capteurs analogiques
                - Détection d'anomalies par apprentissage automatique
                - Clustering des états de fonctionnement
                
                **Alertes courantes:**
                - Surchauffe
                - Baisse de performance
                - Variations anormales des paramètres
                
                Pour plus de détails, consultez la page **APU** de cette application.
                """
            else:
                response = """
                Je suis un assistant spécialisé dans la détection de pannes d'avion. Je peux vous aider avec:
                
                - **Moteurs**: Analyse RUL, détection d'anomalies
                - **Système hydraulique**: Surveillance des composants
                - **APU**: Détection d'anomalies sur les unités de puissance auxiliaire
                
                Posez-moi une question spécifique sur l'un de ces sujets!
                """
            
            # Ajouter la réponse à l'historique
            st.session_state.chat_messages.append({"role": "assistant", "content": response})
            
            # Recharger la page pour afficher le nouveau message
            st.experimental_rerun()
    
    # Bouton pour effacer l'historique
    if st.session_state.chat_messages:
        if st.button("Effacer l'historique", key="clear_chat"):
            st.session_state.chat_messages = []
            st.experimental_rerun()

# -------------------------------
# Point d'entrée
# -------------------------------
if __name__ == "__main__":
    main()