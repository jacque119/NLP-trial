import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

@st.cache_data
def load_data():
    return pd.read_csv("IMDB Dataset.csv")

def show_eda():
    st.subheader("Sample Data")
    df = load_data()
    st.write(df.sample(5))

    st.subheader("Sentiment Distribution")
    fig, ax = plt.subplots()
    sns.countplot(data=df, x='sentiment', ax=ax)
    st.pyplot(fig)
