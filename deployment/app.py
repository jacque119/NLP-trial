import streamlit as st
from predictions import predict_sentiment
from eda import show_eda

st.set_page_config(page_title="IMDB Sentiment Analyzer", layout="centered")

st.title("IMDB Movie Reviews Sentiment Analyzer")

menu = st.sidebar.selectbox("Menu", ["Predict Sentiment", "EDA"])

if menu == "Predict Sentiment":
    st.subheader("Enter a Movie Reviews:")
    user_input = st.text_area("Write your review here...")

    if st.button("Predict"):
        if not user_input.strip():
            st.warning("Please enter a review before predicting.")
        else:
            label, score = predict_sentiment(user_input)
            st.success(f"Predicted Sentiment: {label} (Confidence: {score:.4f})")

elif menu == "EDA":
    show_eda()
