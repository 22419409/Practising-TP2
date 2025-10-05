import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score

@st.cache_data
def load_data():
    df = pd.read_csv("Slum_Population_India_Unclean.csv")
    df.columns = [c.strip() for c in df.columns]
    df['State'] = df['State'].astype(str).str.strip().str.title()
    numeric_cols = ['Slums', 'Slum HH', 'Slum Pop', 'Literacy']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col].astype(str).str.replace(',', '').str.replace('%', '').str.strip(),
                errors='coerce'
            )
    df.fillna(df.median(numeric_only=True), inplace=True)
    return df

df = load_data()
st.set_page_config(page_title="Human Settlement Dashboard", layout="wide")
st.title("🏙️ Human Settlement for Sustainable Development")

st.sidebar.title("Navigation")
page = st.sidebar.radio("Select Dashboard", ["📊 Data Insights", "🤖 ML Prediction Tool"])

# --- Dashboard 1 ---
if page == "📊 Data Insights":
    st.header("📊 Data Insights Dashboard")
    st.write(df.head())
    col1, col2, col3 = st.columns(3)
    col1.metric("Total States", df['State'].nunique())
    col2.metric("Total Slum Population", f"{df['Slum Pop'].sum():,.0f}")
    col3.metric("Average Literacy Rate", f"{df['Literacy'].mean():.2f}%")

    st.subheader("Top 10 States by Slum Population")
    top10 = df.nlargest(10, 'Slum Pop')
    fig, ax = plt.subplots(figsize=(10,5))
    sns.barplot(data=top10, x='State', y='Slum Pop', palette='Blues_d')
    plt.xticks(rotation=45)
    st.pyplot(fig)

    st.subheader("Literacy vs Slum Population")
    fig2, ax2 = plt.subplots(figsize=(8,6))
    sns.regplot(data=df, x='Literacy', y='Slum Pop', scatter_kws={'s':40}, line_kws={'color':'red'})
    st.pyplot(fig2)

    st.subheader("Correlation Heatmap")
    fig3, ax3 = plt.subplots(figsize=(7,5))
    sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
    st.pyplot(fig3)

# --- Dashboard 2 ---
if page == "🤖 ML Prediction Tool":
    st.header("🤖 Machine Learning Prediction Dashboard")
    df['HighSlum'] = (df['Slum Pop'] >= df['Slum Pop'].median()).astype(int)
    features = ['Literacy', 'Slums', 'Slum HH']
    X = df[features]
    y = df['HighSlum']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = DecisionTreeClassifier(max_depth=3, random_state=42)
    model.fit(X_train, y_train)
    acc = model.score(X_test, y_test)

    literacy = st.slider("Literacy Rate (%)", float(df['Literacy'].min()), float(df['Literacy'].max()), float(df['Literacy'].mean()))
    slums = st.slider("Slum Percentage (%)", float(df['Slums'].min()), float(df['Slums'].max()), float(df['Slums'].mean()))
    hh = st.number_input("Number of Slum Households", min_value=0, max_value=int(df['Slum HH'].max()), value=int(df['Slum HH'].mean()))

    if st.button("Predict"):
        input_data = pd.DataFrame({'Literacy':[literacy], 'Slums':[slums], 'Slum HH':[hh]})
        pred = model.predict(input_data)[0]
        result = "🌆 High Slum Population Area" if pred==1 else "🏠 Low Slum Population Area"
        st.success(f"Prediction: {result}")
        st.info(f"Model Accuracy: {acc*100:.2f}%")

        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        st.subheader("Confusion Matrix")
        st.write(pd.DataFrame(cm, index=["Actual Low", "Actual High"], columns=["Pred Low", "Pred High"]))
