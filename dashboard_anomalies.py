import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from scipy import stats
from datetime import datetime

# Configuration de la page
st.set_page_config(layout="wide")
st.title("DASHBOARD DE DETECTION D'ANOMALIES DANS LES VENTES")

# MENU NAVIGATION
st.sidebar.title("Navigation")
section = st.sidebar.radio("Aller à :", ["Accueil", "Détection d’anomalies", "Alertes & recommandations", "Prévision des ventes"])

# Génération des données
@st.cache_data
def generate_data():
    np.random.seed(42)
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2024, 6, 30)
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')

    def generate_sales_data(date_range):
        data = []
        for i, date in enumerate(date_range):
            base_sales = 1000 + (i * 2)
            weekly_multiplier = [0.8, 0.9, 1.0, 1.1, 1.2, 1.5, 1.3][date.weekday()]
            month_multiplier = [0.8, 0.9, 1.0, 1.1, 1.2, 1.0, 0.9, 0.8, 1.1, 1.3, 1.5, 1.8][date.month-1]
            noise = np.random.normal(0, 100)
            sales = base_sales * weekly_multiplier * month_multiplier + noise
            data.append({
                'date': date,
                'sales': max(0, sales),
                'day_of_week': date.weekday(),
                'month': date.month,
                'is_weekend': date.weekday() >= 5
            })
        return pd.DataFrame(data)

    def inject_anomalies(df):
        df_copy = df.copy()
        df_copy.loc[df_copy['date'] == pd.to_datetime('2023-11-24'), 'sales'] *= 3
        df_copy.loc[df_copy['date'] == pd.to_datetime('2023-11-24'), 'anomaly_type'] = 'black_friday'

        for d in ['2023-03-15', '2023-08-22', '2024-01-10']:
            df_copy.loc[df_copy['date'] == pd.to_datetime(d), 'sales'] *= 0.2
            df_copy.loc[df_copy['date'] == pd.to_datetime(d), 'anomaly_type'] = 'stock_out'

        for d in ['2023-07-14', '2023-12-26', '2024-02-14']:
            df_copy.loc[df_copy['date'] == pd.to_datetime(d), 'sales'] *= 2.5
            df_copy.loc[df_copy['date'] == pd.to_datetime(d), 'anomaly_type'] = 'promotion'

        for d in ['2023-05-10', '2023-09-18']:
            df_copy.loc[df_copy['date'] == pd.to_datetime(d), 'sales'] *= 0.1
            df_copy.loc[df_copy['date'] == pd.to_datetime(d), 'anomaly_type'] = 'tech_issue'

        df_copy['anomaly_type'].fillna('normal', inplace=True)
        return df_copy

    df = generate_sales_data(date_range)
    return inject_anomalies(df)

df = generate_data()

# Feature engineering
def create_features(df):
    df['ma_7'] = df['sales'].rolling(window=7, min_periods=1).mean()
    df['ma_30'] = df['sales'].rolling(window=30, min_periods=1).mean()
    df['deviation_ma_7'] = df['sales'] - df['ma_7']
    df['deviation_ma_30'] = df['sales'] - df['ma_30']
    df['ratio_ma_7'] = df['sales'] / df['ma_7']
    df['ratio_ma_30'] = df['sales'] / df['ma_30']
    return df.dropna()

df = create_features(df)

# Détection des anomalies
features = ['sales', 'day_of_week', 'month', 'deviation_ma_7', 'deviation_ma_30', 'ratio_ma_7', 'ratio_ma_30']
model = IsolationForest(contamination=0.05, random_state=42)
df['anomaly'] = model.fit_predict(df[features])
df['anomaly'] = df['anomaly'] == -1

# Catégorisation
median_sales = df['sales'].median()
df['anomaly_category'] = np.select([
    df['sales'] > median_sales * 2,
    df['sales'] < median_sales * 0.5
], ['spike', 'drop'], default='moderate')


# === ACCUEIL ===
if section == "Accueil":
    st.header("Bienvenue sur le Dashboard d’Analyse des Ventes !")
    st.markdown("""
    Ce tableau de bord utilise l’intelligence artificielle pour détecter automatiquement des anomalies dans les ventes.

    ### Fonctions principales :
    - Surveillance des ventes journalières
    - Détection de hausses et chutes suspectes
    - Recommandations business
    - Visualisation des KPI

    NB: Utilisez le menu à gauche pour naviguer dans les différentes sections.
    """)


# === DÉTECTION ===
elif section == " Détection d’anomalies":
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Nombre de jours", len(df))
    col2.metric("Ventes moyennes", f"{df['sales'].mean():.0f} €")
    col3.metric("Anomalies détectées", df['anomaly'].sum())
    col4.metric("% d'anomalies", f"{df['anomaly'].mean()*100:.1f} %")

    with st.sidebar:
        st.header("Filtres")
        selected_months = st.multiselect("Mois", sorted(df['month'].unique()), default=sorted(df['month'].unique()))
        selected_categories = st.multiselect("Catégorie d'anomalie", df['anomaly_category'].unique(), default=df['anomaly_category'].unique())

    filtered = df[
        df['month'].isin(selected_months) &
        ((df['anomaly'] == False) | (df['anomaly_category'].isin(selected_categories)))
    ]

    # Série temporelle
    st.subheader("Ventes journalières avec anomalies")
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(filtered['date'], filtered['sales'], label='Ventes')
    ax.scatter(filtered[filtered['anomaly']]['date'], filtered[filtered['anomaly']]['sales'], color='red', label='Anomalies', s=50)
    ax.set_xlabel("Date")
    ax.set_ylabel("Ventes (€)")
    ax.legend()
    st.pyplot(fig)

    # Histogramme
    st.subheader("Répartition des ventes")
    fig2, ax2 = plt.subplots()
    ax2.hist(filtered[~filtered['anomaly']]['sales'], bins=50, alpha=0.7, label='Normales')
    ax2.hist(filtered[filtered['anomaly']]['sales'], bins=20, alpha=0.7, color='red', label='Anomalies')
    ax2.legend()
    st.pyplot(fig2)

    # Anomalies par catégorie
    st.subheader("Anomalies par catégorie")
    st.bar_chart(filtered[filtered['anomaly']]['anomaly_category'].value_counts())

    # Détails des anomalies
    st.subheader("Détails des anomalies détectées")
    st.dataframe(filtered[filtered['anomaly']][['date', 'sales', 'anomaly_type', 'anomaly_category']].reset_index(drop=True))


# === ALERTES ===
elif section == "Alertes & recommandations":
    anomalies_detected = df[df['anomaly']]
    st.subheader("Anomalies critiques détectées")

    for _, row in anomalies_detected.iterrows():
        st.markdown(f"""
        #### {row['date'].strftime('%d %B %Y')}
        - Ventes : **{int(row['sales'])}€**
        - Type : `{row['anomaly_type']}` / Catégorie : `{row['anomaly_category']}`
        - Action recommandée :
        {"• Vérifier les stocks, bugs ou promotions imprévues" if row['anomaly_category'] == 'drop' else "• Analyser si une promo ou un événement explique cette hausse"}
        ---
        """)


# Prévision des ventes
elif section == " Prévision des ventes":
    st.subheader("Prévision des ventes sur 30 jours")

    # Préparation des données
    last_date = df['date'].max()
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=30)

    # Utilisation de la moyenne mobile et de la tendance
    forecast_base = df['sales'].rolling(window=30, min_periods=1).mean().iloc[-1]
    trend = df['sales'].diff().mean()  # petite tendance moyenne

    future_sales = [forecast_base + trend * i for i in range(1, 31)]

    df_future = pd.DataFrame({
        'date': future_dates,
        'sales': future_sales
    })

    # Affichage du graphe combiné
    st.markdown("Prévision basée sur la tendance moyenne et la moyenne mobile des derniers jours.")

    fig3, ax3 = plt.subplots(figsize=(14, 4))
    ax3.plot(df['date'], df['sales'], label='Historique des ventes')
    ax3.plot(df_future['date'], df_future['sales'], linestyle='--', color='green', label='Prévision (30j)')
    ax3.set_xlabel("Date")
    ax3.set_ylabel("Ventes (€)")
    ax3.legend()
    st.pyplot(fig3)
