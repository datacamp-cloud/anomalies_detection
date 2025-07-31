import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from scipy import stats
from datetime import datetime

st.set_page_config(layout="wide")
st.title("📊 Dashboard de Détection d'Anomalies dans les Ventes")

# Load data
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


# Create features for anomaly detection
def create_features(df):
    df['ma_7'] = df['sales'].rolling(window=7, min_periods=1).mean()
    df['ma_30'] = df['sales'].rolling(window=30, min_periods=1).mean()
    df['deviation_ma_7'] = df['sales'] - df['ma_7']
    df['deviation_ma_30'] = df['sales'] - df['ma_30']
    df['ratio_ma_7'] = df['sales'] / df['ma_7']
    df['ratio_ma_30'] = df['sales'] / df['ma_30']
    return df.dropna()

df = create_features(df)

features = ['sales', 'day_of_week', 'month', 'deviation_ma_7', 'deviation_ma_30', 'ratio_ma_7', 'ratio_ma_30']
model = IsolationForest(contamination=0.05, random_state=42)
df['anomaly'] = model.fit_predict(df[features])
df['anomaly'] = df['anomaly'] == -1

median_sales = df['sales'].median()
df['anomaly_category'] = np.select([
    df['sales'] > median_sales * 2,
    df['sales'] < median_sales * 0.5
], ['spike', 'drop'], default='moderate')


# Dashboard Layout

col1, col2, col3, col4 = st.columns(4)
col1.metric("📅 Nombre de jours", len(df))
col2.metric("💰 Ventes moyennes", f"{df['sales'].mean():.0f} €")
col3.metric("🚨 Anomalies détectées", df['anomaly'].sum())
col4.metric("📈 % d'anomalies", f"{df['anomaly'].mean()*100:.1f} %")

with st.sidebar:
    st.header("🔍 Filtres")
    selected_months = st.multiselect("Mois", sorted(df['month'].unique()), default=sorted(df['month'].unique()))
    selected_categories = st.multiselect("Catégorie d'anomalie", df['anomaly_category'].unique(), default=df['anomaly_category'].unique())

filtered = df[
    df['month'].isin(selected_months) &
    ((df['anomaly'] == False) | (df['anomaly_category'].isin(selected_categories)))
]


# Visualization
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
