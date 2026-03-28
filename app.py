# =====================================================
# FINAL STREAMLIT HEALTH DATA ANALYTICS DASHBOARD
# =====================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression

# =====================================================
# PAGE CONFIG (FRONTEND SETUP)
# =====================================================
st.set_page_config(
    page_title="Health Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Health Data Analytics Dashboard")
st.markdown("### Interactive Insights & Prediction System")

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
df.drop_duplicates(inplace=True)

# Feature Engineering
df['Death_Rate'] = df['Deaths'] / df['Cases']
df['Recovery_Rate'] = df['Recovered'] / df['Cases']
df.replace([np.inf, -np.inf], 0, inplace=True)

# =====================================================
# KPI METRICS (DASHBOARD)
# =====================================================
st.subheader("📌 Key Metrics")

col1, col2, col3 = st.columns(3)

col1.metric("Total Cases", int(df['Cases'].sum()))
col2.metric("Total Deaths", int(df['Deaths'].sum()))
col3.metric("Total Recovered", int(df['Recovered'].sum()))

# =====================================================
# SIDEBAR FILTER
# =====================================================
st.sidebar.header("🔍 Filter Data")

country = st.sidebar.selectbox("Select Country", df['country'].unique())
filtered_df = df[df['country'] == country]

# =====================================================
# TABS (FRONTEND STRUCTURE)
# =====================================================
tab1, tab2, tab3 = st.tabs(["📄 Data", "📊 Charts", "🤖 Prediction"])

# =====================================================
# TAB 1: DATA
# =====================================================
with tab1:
    st.subheader("📄 Dataset Preview")
    st.dataframe(filtered_df.head())

    # Download Button
    csv = filtered_df.to_csv(index=False)
    st.download_button("📥 Download Data", csv, "data.csv")

# =====================================================
# TAB 2: CHARTS
# =====================================================
with tab2:
    st.subheader("🌍 Top 10 Countries by Cases")

    top10 = df.groupby('country')['Cases'].max().sort_values(ascending=False).head(10)

    fig1, ax1 = plt.subplots()
    top10.plot(kind='bar', ax=ax1)
    plt.xticks(rotation=45)
    st.pyplot(fig1)

    st.subheader("📉 Cases vs Deaths")

    fig2, ax2 = plt.subplots()
    sns.scatterplot(x='Cases', y='Deaths', data=df, ax=ax2)
    st.pyplot(fig2)

    st.subheader("📊 Death Rate Distribution")

    fig3, ax3 = plt.subplots()
    sns.histplot(df['Death_Rate'], bins=30, kde=True, ax=ax3)
    st.pyplot(fig3)

# =====================================================
# TAB 3: MACHINE LEARNING PREDICTION
# =====================================================
with tab3:
    st.subheader("🤖 Predict Deaths using ML")

    # Train Model
    X = df[['Cases', 'population', 'Tests']]
    y = df['Deaths']

    X = X.fillna(0)
    y = y.fillna(0)

    model = LinearRegression()
    model.fit(X, y)

    # User Input
    cases = st.number_input("Enter Cases", value=1000)
    population = st.number_input("Enter Population", value=100000)
    tests = st.number_input("Enter Tests", value=5000)

    if st.button("Predict"):
        prediction = model.predict([[cases, population, tests]])
        st.success(f"Predicted Deaths: {int(prediction[0])}")

# =====================================================
# END
# =====================================================