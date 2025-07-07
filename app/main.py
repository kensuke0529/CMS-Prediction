import joblib
import pandas as pd
import streamlit as st
from pathlib import Path

base = Path().resolve().parent 
model_path = base / 'model' / 'xgb_pipeline.joblib'  

model = joblib.load(model_path)

# Streamlit form
st.title("Medicare Payment Prediction")

# User inputs
Rndrng_Prvdr_CCN = st.text_input("Provider CCN (e.g. 010001)", value="010001")
Rndrng_Prvdr_State_FIPS = st.number_input("Provider State FIPS", 0, 99, 1)
Rndrng_Prvdr_Zip5 = st.number_input("Provider Zip5", 10000, 99999, 36301)
DRG_Cd = st.text_input("DRG Code (e.g. 023)", value="023")
year = st.number_input("Year", 2000, 2030, 2021)
RUCA_category = st.selectbox(
    "RUCA Category", ["metro_core", "metro_other", "rural", "other"])
drg_grouped = st.text_input("DRG Grouped", value="023")

# DataFrame for prediction
input_dict = {
    "Rndrng_Prvdr_CCN": [Rndrng_Prvdr_CCN],
    "Rndrng_Prvdr_State_FIPS": [float(Rndrng_Prvdr_State_FIPS)],
    "Rndrng_Prvdr_Zip5": [float(Rndrng_Prvdr_Zip5)],
    "DRG_Cd": [DRG_Cd],
    "year": [int(year)],
    "RUCA_category": [RUCA_category],
    "drg_grouped": [drg_grouped],
}

input_df = pd.DataFrame(input_dict)

if st.button("Predict"):
    try:
        prediction = model.predict(input_df)
        st.success(
            f"Predicted Average Medicare Payment: ${prediction[0]:,.2f}")
    except Exception as e:
        st.error(f"Prediction error: {e}")


# http://localhost:8501/