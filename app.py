
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="WDBC Breast Cancer AI",
    page_icon="🩺",
    layout="wide"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

.stApp {
    background: #f5f8fc;
}

.hero {
    padding: 35px;
    border-radius: 20px;
    background: linear-gradient(
        135deg,
        #0f172a,
        #1e3a8a
    );
    color: white;
    margin-bottom: 30px;
}

.hero h1 {
    font-size: 42px;
    margin-bottom: 5px;
}

.hero p {
    font-size: 18px;
}

.card {
    background: white;
    padding: 25px;
    border-radius: 18px;
    box-shadow: 0px 5px 20px rgba(0,0,0,0.08);
    margin-bottom: 20px;
}

.result {
    background: white;
    padding: 30px;
    border-radius: 20px;
    text-align: center;
    box-shadow: 0px 5px 20px rgba(0,0,0,0.10);
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# LOAD MODEL
# ============================================================

model = joblib.load("wdbc_xgboost.pkl")

feature_names = joblib.load(
    "feature_names.pkl"
)

explainer = shap.TreeExplainer(model)


# ============================================================
# HEADER
# ============================================================

st.markdown("""
<div class="hero">

<h1>🩺 WDBC Breast Cancer AI</h1>

<p>
XGBoost Classification + SHAP Explainability
</p>

</div>
""", unsafe_allow_html=True)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("Navigation")

    page = st.radio(
        "Select Page",
        [
            "🔬 Prediction",
            "🧩 SHAP Explainability",
            "ℹ️ About"
        ]
    )

    st.divider()

    st.caption("Dataset: WDBC")
    st.caption("Model: XGBoost")
    st.caption("Explainability: SHAP")


# ============================================================
# PREDICTION PAGE
# ============================================================

if page == "🔬 Prediction":

    st.header("🔬 Breast Cancer Prediction")

    st.write(
        "Enter the 30 diagnostic features used by the WDBC dataset."
    )

    input_values = []

    col1, col2 = st.columns(2)

    for i, feature in enumerate(feature_names):

        if i % 2 == 0:

            with col1:

                value = st.number_input(
                    feature,
                    value=0.0,
                    format="%.5f",
                    key=f"feature_{i}"
                )

        else:

            with col2:

                value = st.number_input(
                    feature,
                    value=0.0,
                    format="%.5f",
                    key=f"feature_{i}"
                )

        input_values.append(value)


    input_df = pd.DataFrame(
        [input_values],
        columns=feature_names
    )


    if st.button(
        "🔍 Predict",
        use_container_width=True
    ):

        prediction = model.predict(
            input_df
        )[0]

        probabilities = model.predict_proba(
            input_df
        )[0]

        confidence = np.max(probabilities) * 100


        if prediction == 0:

            result = "Malignant"

        else:

            result = "Benign"


        st.markdown(
            f"""
            <div class="result">

            <h2>Prediction</h2>

            <h1>{result}</h1>

            <h3>Confidence: {confidence:.2f}%</h3>

            </div>
            """,
            unsafe_allow_html=True
        )


        # ----------------------------------------------------
        # SHAP FOR CURRENT PREDICTION
        # ----------------------------------------------------

        st.subheader("🧩 Why did the model make this prediction?")

        current_shap = explainer(
            input_df
        )

        fig = plt.figure()

        shap.plots.waterfall(
            current_shap[0],
            max_display=15,
            show=False
        )

        st.pyplot(
            fig,
            clear_figure=True
        )


# ============================================================
# SHAP PAGE
# ============================================================

elif page == "🧩 SHAP Explainability":

    st.header("🧩 SHAP Explainability")

    st.write(
        """
        SHAP explains how individual features influence
        the XGBoost prediction.
        """
    )

    uploaded_file = st.file_uploader(
        "Upload a CSV containing WDBC features",
        type=["csv"]
    )

    if uploaded_file:

        df = pd.read_csv(
            uploaded_file
        )

        missing_features = [
            feature
            for feature in feature_names
            if feature not in df.columns
        ]

        if missing_features:

            st.error(
                "Missing features in CSV: "
                + ", ".join(missing_features)
            )

        else:

            df = df[
                feature_names
            ]

            shap_values = explainer(
                df
            )


            # ------------------------------------------------
            # BAR PLOT
            # ------------------------------------------------

            st.subheader(
                "Global Feature Importance"
            )

            fig, ax = plt.subplots()

            shap.plots.bar(
                shap_values,
                max_display=15,
                show=False
            )

            st.pyplot(
                fig,
                clear_figure=True
            )


            # ------------------------------------------------
            # BEESWARM
            # ------------------------------------------------

            st.subheader(
                "SHAP Summary Plot"
            )

            fig, ax = plt.subplots()

            shap.plots.beeswarm(
                shap_values,
                max_display=15,
                show=False
            )

            st.pyplot(
                fig,
                clear_figure=True
            )


# ============================================================
# ABOUT PAGE
# ============================================================

elif page == "ℹ️ About":

    st.header("ℹ️ About the Project")

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Dataset",
            "WDBC"
        )

    with col2:

        st.metric(
            "Model",
            "XGBoost"
        )

    with col3:

        st.metric(
            "Explainability",
            "SHAP"
        )


    st.markdown("""
    ### Research Pipeline

    **WDBC Dataset**

    ↓

    **XGBoost Classification**

    ↓

    **Benign / Malignant Prediction**

    ↓

    **SHAP Explainability**

    ↓

    **Streamlit Deployment**
    """)

    st.info(
        "This application is intended for research and "
        "educational purposes and is not a medical diagnostic tool."
    )
