
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Coffee Quality Predictor", page_icon="☕")

st.title("☕ Coffee Quality Predictor")
st.write("Enter coffee details to predict its quality category.")

@st.cache_resource
def load_model():
    return joblib.load("coffee_random_forest.pkl")

model = load_model()

country = st.text_input("Country of Origin", "Colombia")
variety = st.text_input("Coffee Variety", "Bourbon")
processing = st.text_input("Processing Method", "Washed")

altitude = st.number_input("Altitude (meters)", value=1500.0)
moisture = st.number_input("Moisture", min_value=0.0, value=10.0)
defects_one = st.number_input("Category One Defects", min_value=0, value=0)
quakers = st.number_input("Quakers", min_value=0, value=0)
defects_two = st.number_input("Category Two Defects", min_value=0, value=5)

if st.button("Predict Coffee Quality"):
    sample = pd.DataFrame([{
        "Country.of.Origin": country,
        "Variety": variety,
        "Processing.Method": processing,
        "altitude_mean_meters": altitude,
        "Moisture": moisture,
        "Category.One.Defects": defects_one,
        "Quakers": quakers,
        "Category.Two.Defects": defects_two
    }])

    prediction = model.predict(sample)[0]

    if prediction == 1:
        st.success("Predicted category: 85 points or above")
    else:
        st.info("Predicted category: Below 85 points")

st.caption("Educational machine learning project.")
