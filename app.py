# =====================================================
# STREAMLIT DATA ANALYTICS APP
# =====================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(page_title="Health Data Dashboard", layout="wide")

st.title("📊 Health Data Analytics Dashboard")

# =====================================================
# LOAD DATA
# =====================================================
@st.cache_data
def load_data():
    df = pd.read_csv("covid_19.csv")
    return df

df = load_data()

# =====================================================
# DATA CLEANING
# =====================================================
df['Deaths'].fillna(0, inplace=True)
df['Tests'].fillna(df['Tests'].mean(), inplace=True)

# Feature Engineering
df['Death_Rate'] = df['Deaths'] / df['Cases']
df['Recovery_Rate'] = df['Recovered'] / df['Cases']

df.replace([np.inf, -np.inf], 0, inplace=True)

# =====================================================
# SIDEBAR
# =====================================================
st.sidebar.header("Filter Data")

country = st.sidebar.selectbox("Select Country", df['country'].unique())

filtered_df = df[df['country'] == country]

# =====================================================
# SHOW DATA
# =====================================================
st.subheader("📄 Dataset Preview")
st.dataframe(filtered_df.head())

# =====================================================
# KPI METRICS
# =====================================================
st.subheader("📌 Key Metrics")

col1, col2, col3 = st.columns(3)

col1.metric("Total Cases", int(filtered_df['Cases'].sum()))
col2.metric("Total Deaths", int(filtered_df['Deaths'].sum()))
col3.metric("Total Recovered", int(filtered_df['Recovered'].sum()))

# =====================================================
# VISUALIZATION 1: TOP COUNTRIES
# =====================================================
st.subheader("🌍 Top 10 Countries by Cases")

top10 = df.groupby('country')['Cases'].max().sort_values(ascending=False).head(10)

fig1, ax1 = plt.subplots()
top10.plot(kind='bar', ax=ax1)
plt.xticks(rotation=45)

st.pyplot(fig1)

# =====================================================
# VISUALIZATION 2: CASES VS DEATHS
# =====================================================
st.subheader("📉 Cases vs Deaths")

fig2, ax2 = plt.subplots()
sns.scatterplot(x='Cases', y='Deaths', data=df, ax=ax2)

st.pyplot(fig2)

# =====================================================
# VISUALIZATION 3: DEATH RATE DISTRIBUTION
# =====================================================
st.subheader("📊 Death Rate Distribution")

fig3, ax3 = plt.subplots()
sns.histplot(df['Death_Rate'], bins=30, kde=True, ax=ax3)

st.pyplot(fig3)

# =====================================================
# SIMPLE PREDICTION (MANUAL INPUT)
# =====================================================
st.subheader("🤖 Predict Deaths")

cases = st.number_input("Enter Cases", value=1000)
population = st.number_input("Enter Population", value=100000)
tests = st.number_input("Enter Tests", value=5000)

# Simple formula-based prediction (beginner friendly)
predicted_deaths = (cases * 0.02)

st.write("### Predicted Deaths:", int(predicted_deaths))

# =====================================================
# END
# =====================================================