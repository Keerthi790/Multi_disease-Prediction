# app.py

import streamlit as st
import multi_disease_model as mdm

st.set_page_config(page_title="Multi Disease Prediction", layout="centered")
st.title("🩺 Multiple Disease Prediction System")

# Sidebar
st.sidebar.title("Select Disease")
choice = st.sidebar.radio("", [
    "Diabetes Prediction",
    "Heart Disease Prediction",
    "Kidney Disease Prediction",
    "Liver Disease Prediction"
])

# Lazy‐load models once
if 'loaded' not in st.session_state:
    try:
        mdm.load_and_train_all()
        st.session_state.loaded = True
    except FileNotFoundError as e:
        st.sidebar.error(str(e))
        st.stop()

# Helper to render a form with feature ranges and display for normal and disease
def render_form(key_name, disease_key):
    st.header(f"{disease_key.capitalize()} Prediction")
    info = mdm.get_feature_info(disease_key)
    inputs = []
    with st.form(key_name):
        for feat, mn, mx in info:
            st.write(f"Range for {feat}: {mn} - {mx}")
            
            # Display ranges for normal and disease if available
            # You can replace the below values with those specific to each disease type.
            if disease_key == "diabetes":
                normal_range = f"Normal: {mn} - {mx}"  # Replace this with actual normal range values for diabetes
                disease_range = f"Disease: {mn + 0.1} - {mx + 0.2}"  # Replace with disease-specific ranges
            elif disease_key == "heart":
                normal_range = f"Normal: {mn} - {mx}"  # Replace with heart disease-specific range
                disease_range = f"Disease: {mn + 0.1} - {mx + 0.2}"
            elif disease_key == "kidney":
                normal_range = f"Normal: {mn} - {mx}"
                disease_range = f"Disease: {mn + 0.1} - {mx + 0.2}"
            elif disease_key == "liver":
                normal_range = f"Normal: {mn} - {mx}"
                disease_range = f"Disease: {mn + 0.1} - {mx + 0.2}"

            st.write(normal_range)  # Show normal range
            st.write(disease_range)  # Show disease range

            default = (mn + mx) / 2  # Default value is the middle of the range
            val = st.number_input(
                label=feat,
                min_value=mn,
                max_value=mx,
                value=default,
                step=(mx - mn) / 100 if mx > mn else 1.0
            )
            inputs.append(val)
        
        submit = st.form_submit_button("Predict")
    
    if submit:
        pred = mdm.predict_disease(disease_key, inputs)
        if pred == 1:
            st.success(f"🟢 **Disease Present** for {disease_key.capitalize()}")
        else:
            st.success(f"🔴 **Normal** for {disease_key.capitalize()}")

# Dispatch
if choice == "Diabetes Prediction":
    render_form("form_diabetes", "diabetes")
elif choice == "Heart Disease Prediction":
    render_form("form_heart", "heart")
elif choice == "Kidney Disease Prediction":
    render_form("form_kidney", "kidney")
elif choice == "Liver Disease Prediction":
    render_form("form_liver", "liver")
