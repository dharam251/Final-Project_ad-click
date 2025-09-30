import streamlit as st
import pandas as pd
import joblib
import warnings

# Suppress Streamlit warnings for a cleaner interface
warnings.filterwarnings('ignore')

# --- 1. Load the Saved ML Pipeline ---
MODEL_FILENAME = 'ad_click_bagging_pipeline.pkl'

try:
    # Load the entire pipeline (preprocessor + model)
    pipeline = joblib.load(MODEL_FILENAME)
    st.sidebar.success("Model pipeline loaded successfully.")
except FileNotFoundError:
    st.error(f"FATAL ERROR: Model file '{MODEL_FILENAME}' not found. Please ensure it is uploaded to your repo.")
    st.stop()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()


# --- 2. Input Form Configuration ---
st.set_page_config(page_title="Ad Click Prediction System", layout="centered")

st.title("💸 E-Commerce Ad Click Predictor")
st.markdown("---")
st.caption("Predicting a user's likelihood to click on a digital advertisement using **Random Forest**.")

# Define the options for the categorical dropdowns (based on training data)
# These values MUST match the categories seen during training for the OneHotEncoder to work correctly.
GENDER_OPTIONS = ['Male', 'Female', 'Non-Binary']
DEVICE_OPTIONS = ['Desktop', 'Mobile', 'Tablet']
POSITION_OPTIONS = ['Top', 'Bottom', 'Side']
HISTORY_OPTIONS = ['Shopping', 'Education', 'Entertainment', 'Social Media']
TIME_OPTIONS = ['Morning', 'Afternoon', 'Evening', 'Night']


# Create the form for user input
with st.form("prediction_form"):
    st.header("User Context and Ad Placement")

    # Numerical Input
    age = st.slider("Age", min_value=18, max_value=80, value=30, step=1)

    # Categorical Inputs
    col1, col2 = st.columns(2)
    with col1:
        gender = st.selectbox("Gender", options=GENDER_OPTIONS)
    with col2:
        device_type = st.selectbox("Device Type", options=DEVICE_OPTIONS)

    col3, col4 = st.columns(2)
    with col3:
        ad_position = st.selectbox("Ad Position", options=POSITION_OPTIONS)
    with col4:
        time_of_day = st.selectbox("Time of Day", options=TIME_OPTIONS)

    browsing_history = st.selectbox("Browsing History", options=HISTORY_OPTIONS, help="User's primary recent browsing category.")

    # Submit Button
    submitted = st.form_submit_button("Predict Click Likelihood")


# --- 3. Prediction Logic ---
if submitted:
    # 1. Create a raw input DataFrame
    raw_data = {
        'id': [None], # Placeholder for columns that were in the original CSV but dropped
        'full_name': [None], # Placeholder
        'age': [age],
        'gender': [gender],
        'device_type': [device_type],
        'ad_position': [ad_position],
        'browsing_history': [browsing_history],
        'time_of_day': [time_of_day]
    }
    input_df = pd.DataFrame(raw_data)

    # Reorder columns to match the EXPECTED input structure of the pipeline
    input_df = input_df[['id', 'full_name', 'age', 'gender', 'device_type', 'ad_position', 'browsing_history', 'time_of_day']]

    try:
        # The pipeline handles all preprocessing (dropping ID/name, scaling, encoding) automatically!
        prediction_proba = pipeline.predict_proba(input_df)[0]
        click_prob = prediction_proba[1] # Probability of class 1 (Clicked)
        prediction_class = pipeline.predict(input_df)[0] # Final class (0 or 1)

        # --- 4. Display Results ---
        st.markdown("## Outcome")

        if prediction_class == 1:
            st.success(f"**Predicted Outcome:** CLICK! 🎯")
            st.markdown(f"**Confidence Score (Click Probability):** **{click_prob:.2f}**")
            st.info("This is a high-opportunity scenario for ad display.")
        else:
            st.warning(f"**Predicted Outcome:** NO CLICK 🚫")
            st.markdown(f"**Confidence Score (Click Probability):** **{click_prob:.2f}**")
            st.info("This scenario is low-value. Consider saving ad budget.")

    except Exception as e:
        st.error(f"An error occurred during prediction. Check your console for details. Error: {e}")
