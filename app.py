import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import gdown

# =========================================
# PAGE CONFIG
# =========================================

st.set_page_config(
    page_title="Telecom Customer Churn Platform",
    page_icon="📡",
    layout="wide"
)

# =========================================
# BASE DIRECTORY
# =========================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# =========================================
# GOOGLE DRIVE FILE IDS
# =========================================

CHURN_MODEL_ID = "1M74k49wB6W_WSdYD5D6hEGxAl8TnaCur"

REVENUE_MODEL_ID = "1SyqLpgT4MhUgotP1K3CJCxhYxjUzAAns"

# =========================================
# MODEL PATHS
# =========================================

CHURN_MODEL_PATH = os.path.join(
    BASE_DIR,
    "churn_model.pkl"
)

REVENUE_MODEL_PATH = os.path.join(
    BASE_DIR,
    "revenue_model.pkl"
)

# =========================================
# DOWNLOAD CHURN MODEL
# =========================================

if not os.path.exists(CHURN_MODEL_PATH):

    churn_url = (
        f"https://drive.google.com/uc?id={CHURN_MODEL_ID}"
    )

    gdown.download(
        churn_url,
        CHURN_MODEL_PATH,
        quiet=False
    )

# =========================================
# DOWNLOAD REVENUE MODEL
# =========================================

if not os.path.exists(REVENUE_MODEL_PATH):

    revenue_url = (
        f"https://drive.google.com/uc?id={REVENUE_MODEL_ID}"
    )

    gdown.download(
        revenue_url,
        REVENUE_MODEL_PATH,
        quiet=False
    )

# =========================================
# LOAD MODELS
# =========================================

churn_model = joblib.load(CHURN_MODEL_PATH)

revenue_model = joblib.load(REVENUE_MODEL_PATH)

kmeans_model = joblib.load(
    os.path.join(BASE_DIR, "kmeans_model.pkl")
)

features = joblib.load(
    os.path.join(BASE_DIR, "features.pkl")
)

encoders = joblib.load(
    os.path.join(BASE_DIR, "encoders.pkl")
)

# =========================================
# TITLE
# =========================================

st.title("📡 AI-Powered Telecom Customer Churn Prediction Platform")

st.markdown(
    "### Customer Analytics & Business Intelligence Dashboard"
)

st.divider()

# =========================================
# SIDEBAR
# =========================================

st.sidebar.header("Customer Details")

input_data = {}

# =========================================
# INPUT FEATURES
# =========================================

for feature in features:

    # Numeric Fields
    if feature in [
        "tenure",
        "MonthlyCharges",
        "TotalCharges",
        "SeniorCitizen"
    ]:

        input_data[feature] = st.sidebar.number_input(
            feature,
            value=0.0
        )

    # Gender
    elif feature == "gender":

        gender = st.sidebar.selectbox(
            "Gender",
            ["Female", "Male"]
        )

        input_data[feature] = (
            1 if gender == "Male" else 0
        )

    # Yes / No Features
    elif feature in [
        "Partner",
        "Dependents",
        "PhoneService",
        "PaperlessBilling"
    ]:

        value = st.sidebar.selectbox(
            feature,
            ["No", "Yes"]
        )

        input_data[feature] = (
            1 if value == "Yes" else 0
        )

    # Internet Service
    elif feature == "InternetService":

        value = st.sidebar.selectbox(
            "Internet Service",
            ["DSL", "Fiber optic", "No"]
        )

        mapping = {
            "DSL": 0,
            "Fiber optic": 1,
            "No": 2
        }

        input_data[feature] = mapping[value]

    # Contract
    elif feature == "Contract":

        value = st.sidebar.selectbox(
            "Contract Type",
            [
                "Month-to-month",
                "One year",
                "Two year"
            ]
        )

        mapping = {
            "Month-to-month": 0,
            "One year": 1,
            "Two year": 2
        }

        input_data[feature] = mapping[value]

    # Payment Method
    elif feature == "PaymentMethod":

        value = st.sidebar.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer",
                "Credit card"
            ]
        )

        mapping = {
            "Electronic check": 0,
            "Mailed check": 1,
            "Bank transfer": 2,
            "Credit card": 3
        }

        input_data[feature] = mapping[value]

    # Default Numeric
    else:

        input_data[feature] = st.sidebar.number_input(
            feature,
            value=0.0
        )

# =========================================
# CREATE DATAFRAME
# =========================================

input_df = pd.DataFrame([input_data])

# =========================================
# ANALYZE BUTTON
# =========================================

if st.button("🔍 Analyze Customer"):

    # =====================================
    # CHURN PREDICTION
    # =====================================

    churn_prediction = (
        churn_model.predict(input_df)[0]
    )

    churn_probability = (
        churn_model.predict_proba(input_df)[0][1] * 100
    )

    # =====================================
    # REVENUE LOSS PREDICTION
    # =====================================

    revenue_loss = (
        revenue_model.predict(input_df)[0]
    )

    # =====================================
    # CUSTOMER SEGMENTATION
    # =====================================

    cluster = (
        kmeans_model.predict(input_df)[0]
    )

    # =====================================
    # RETENTION RECOMMENDATIONS
    # =====================================

    recommendations = []

    if churn_probability > 80:

        recommendations.append(
            "Provide premium retention discount."
        )

        recommendations.append(
            "Assign dedicated customer support."
        )

    elif churn_probability > 50:

        recommendations.append(
            "Offer loyalty rewards."
        )

        recommendations.append(
            "Suggest upgraded plans."
        )

    else:

        recommendations.append(
            "Customer is stable."
        )

    # =====================================
    # RESULTS
    # =====================================

    st.subheader("📊 Customer Intelligence Report")

    col1, col2, col3 = st.columns(3)

    # Churn
    with col1:

        if churn_prediction == 1:

            st.error(
                "🚨 Customer Likely to Churn"
            )

        else:

            st.success(
                "✅ Customer Likely to Stay"
            )

        st.metric(
            "Churn Probability",
            f"{churn_probability:.2f}%"
        )

    # Revenue
    with col2:

        st.metric(
            "Expected Revenue Loss",
            f"${revenue_loss:.2f}"
        )

    # Cluster
    with col3:

        st.metric(
            "Customer Segment",
            f"Cluster {cluster}"
        )

    st.progress(
        float(churn_probability / 100)
    )

    # =====================================
    # RECOMMENDATIONS
    # =====================================

    st.subheader(
        "📌 Retention Recommendations"
    )

    for rec in recommendations:

        st.info(rec)

# =========================================
# FOOTER
# =========================================

st.markdown("---")

st.markdown(
    "Built with ❤️ using Streamlit, Random Forest & K-Means"
)
