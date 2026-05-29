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
    page_title="AI Telecom Churn Platform",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================
# CUSTOM CSS
# =========================================

st.markdown("""
<style>

.main {
    background-color: #0E1117;
}

.stApp {
    background-color: #0E1117;
    color: white;
}

.metric-card {
    background: linear-gradient(135deg,#1f2937,#111827);
    padding: 25px;
    border-radius: 18px;
    border: 1px solid #374151;
    box-shadow: 0 4px 20px rgba(0,0,0,0.4);
    text-align: center;
}

.recommendation-card {
    background: #111827;
    padding: 18px;
    border-radius: 14px;
    border-left: 5px solid #3B82F6;
    margin-bottom: 10px;
}

.big-font {
    font-size:20px !important;
    font-weight:bold;
}

</style>
""", unsafe_allow_html=True)

# =========================================
# BASE DIRECTORY
# =========================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# =========================================
# GOOGLE DRIVE MODEL IDS
# =========================================

CHURN_MODEL_ID = "1M74k49wB6W_WSdYD5D6hEGxAl8TnaCur"

# =========================================
# MODEL PATH
# =========================================

CHURN_MODEL_PATH = os.path.join(
    BASE_DIR,
    "churn_model.pkl"
)

# =========================================
# DOWNLOAD MODEL
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
# LOAD MODELS
# =========================================

churn_model = joblib.load(CHURN_MODEL_PATH)

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
# HEADER
# =========================================

st.markdown("""
<h1 style='text-align:center;color:#60A5FA;'>
📡 AI-Powered Telecom Customer Churn Platform
</h1>
""", unsafe_allow_html=True)

st.markdown("""
<p style='text-align:center;font-size:20px;color:#D1D5DB;'>
Customer Analytics • Churn Prediction • Retention Intelligence
</p>
""", unsafe_allow_html=True)

st.divider()

# =========================================
# SIDEBAR
# =========================================

st.sidebar.image(
    "https://cdn-icons-png.flaticon.com/512/3135/3135715.png",
    width=120
)

st.sidebar.title("📋 Customer Details")

input_data = {}

# =========================================
# INPUTS
# =========================================

for feature in features:

    if feature == "gender":

        gender = st.sidebar.selectbox(
            "👤 Gender",
            ["Female", "Male"]
        )

        input_data[feature] = (
            1 if gender == "Male" else 0
        )

    elif feature in [
        "Partner",
        "Dependents",
        "PhoneService",
        "PaperlessBilling"
    ]:

        value = st.sidebar.selectbox(
            f"✅ {feature}",
            ["No", "Yes"]
        )

        input_data[feature] = (
            1 if value == "Yes" else 0
        )

    elif feature == "InternetService":

        value = st.sidebar.selectbox(
            "🌐 Internet Service",
            ["DSL", "Fiber optic", "No"]
        )

        mapping = {
            "DSL": 0,
            "Fiber optic": 1,
            "No": 2
        }

        input_data[feature] = mapping[value]

    elif feature == "Contract":

        value = st.sidebar.selectbox(
            "📑 Contract Type",
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

    elif feature == "PaymentMethod":

        value = st.sidebar.selectbox(
            "💳 Payment Method",
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

    elif feature == "tenure":

        input_data[feature] = st.sidebar.slider(
            "📅 Tenure (Months)",
            0,
            72,
            12
        )

    elif feature == "MonthlyCharges":

        input_data[feature] = st.sidebar.slider(
            "💰 Monthly Charges",
            0,
            500,
            75
        )

    elif feature == "TotalCharges":

        input_data[feature] = st.sidebar.number_input(
            "💵 Total Charges",
            value=1000.0
        )

    elif feature == "SeniorCitizen":

        value = st.sidebar.selectbox(
            "🧓 Senior Citizen",
            ["No", "Yes"]
        )

        input_data[feature] = (
            1 if value == "Yes" else 0
        )

    else:

        input_data[feature] = st.sidebar.number_input(
            feature,
            value=0.0
        )

# =========================================
# DATAFRAME
# =========================================

input_df = pd.DataFrame([input_data])

# =========================================
# BUTTON
# =========================================

if st.button(
    "🚀 Analyze Customer",
    use_container_width=True
):

    # =====================================
    # PREDICTIONS
    # =====================================

    churn_prediction = (
        churn_model.predict(input_df)[0]
    )

    churn_probability = (
        churn_model.predict_proba(input_df)[0][1] * 100
    )

    cluster = (
        kmeans_model.predict(input_df)[0]
    )

    # =====================================
    # REVENUE ESTIMATION
    # =====================================

    monthly_charges = input_data.get(
        "MonthlyCharges",
        0
    )

    revenue_loss = (
        monthly_charges * 0.35
    )

    if churn_probability > 80:

        revenue_loss *= 2

    elif churn_probability > 50:

        revenue_loss *= 1.5

    # =====================================
    # RISK LEVEL
    # =====================================

    if churn_probability > 80:

        risk = "🔴 CRITICAL"

    elif churn_probability > 50:

        risk = "🟠 HIGH"

    elif churn_probability > 30:

        risk = "🟡 MEDIUM"

    else:

        risk = "🟢 LOW"

    # =====================================
    # DASHBOARD CARDS
    # =====================================

    st.subheader("📊 Customer Intelligence Dashboard")

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Churn Probability",
            f"{churn_probability:.2f}%"
        )

    with col2:

        st.metric(
            "Risk Level",
            risk
        )

    with col3:

        st.metric(
            "Revenue Risk",
            f"${revenue_loss:.2f}"
        )

    with col4:

        st.metric(
            "Customer Segment",
            f"Cluster {cluster}"
        )

    st.progress(
        float(churn_probability / 100)
    )

    # =====================================
    # CUSTOMER STATUS
    # =====================================

    st.divider()

    if churn_prediction == 1:

        st.error(
            "🚨 Customer is highly likely to churn."
        )

    else:

        st.success(
            "✅ Customer is likely to stay."
        )

    # =====================================
    # RECOMMENDATIONS
    # =====================================

    st.subheader(
        "📌 AI Retention Recommendations"
    )

    recommendations = []

    if churn_probability > 80:

        recommendations = [
            "Provide premium retention discounts.",
            "Assign dedicated relationship manager.",
            "Offer long-term subscription benefits.",
            "Initiate immediate customer engagement."
        ]

    elif churn_probability > 50:

        recommendations = [
            "Provide loyalty rewards.",
            "Recommend upgraded telecom plans.",
            "Offer personalized promotions."
        ]

    else:

        recommendations = [
            "Customer relationship is stable.",
            "Continue regular engagement campaigns."
        ]

    for rec in recommendations:

        st.markdown(
            f"""
            <div class='recommendation-card'>
            ✅ {rec}
            </div>
            """,
            unsafe_allow_html=True
        )

    # =====================================
    # ANALYTICS
    # =====================================

    st.divider()

    st.subheader("📈 AI Insights")

    insight1, insight2 = st.columns(2)

    with insight1:

        st.info(
            f"""
            📊 Customers in Cluster {cluster}
            show similar telecom usage patterns.
            """
        )

    with insight2:

        st.warning(
            f"""
            💡 Estimated revenue impact:
            ${revenue_loss:.2f} per month.
            """
        )

# =========================================
# FOOTER
# =========================================

st.divider()

st.markdown("""
<div style='text-align:center;color:gray;'>

Built with ❤️ using Streamlit • Random Forest • K-Means • AI Analytics

</div>
""", unsafe_allow_html=True)
