"""Streamlit app that serves the trained model.

Run:  streamlit run app.py   (after running train.py)
"""
import os
import subprocess
import sys

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Saudi Property Price Predictor", page_icon="🏠")
st.title("🏠 Saudi Property Price Predictor")

@st.cache_resource(show_spinner="Training the model (first launch only, about a minute)...")
def load_bundle():
    # On a fresh deployment there is no saved model yet, so train one.
    if not os.path.exists("model.joblib"):
        subprocess.run([sys.executable, "train.py", "--data", "data/SA_Aqar.csv",
                        "--target", "price"], check=True)
    return joblib.load("model.joblib")


try:
    bundle = load_bundle()
except Exception as e:
    st.error(f"Could not load or train the model: {e}")
    st.stop()

pipeline, columns = bundle["pipeline"], bundle["columns"]
categories, defaults = bundle["categories"], bundle["defaults"]

st.write("Enter the property details:")
values = {}
left, right = st.columns(2)
for i, col in enumerate(columns):
    box = left if i % 2 == 0 else right
    if col in categories:
        values[col] = box.selectbox(col, categories[col])
    else:
        d = float(defaults[col])
        # 0/1 features (garage, pool, ...) become checkboxes
        if set(defaults) and d in (0.0, 1.0) and col not in ("size", "property_age"):
            values[col] = int(box.checkbox(col, value=bool(d)))
        else:
            values[col] = box.number_input(col, value=d)

if st.button("Predict price"):
    row = pd.DataFrame([values], columns=columns)
    price = pipeline.predict(row)[0]
    st.success(f"Estimated price: {price:,.0f} SAR")
    st.caption("Estimate from a statistical model trained on listings; not a valuation.")
