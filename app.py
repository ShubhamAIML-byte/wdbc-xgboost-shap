# ============================================================
# WDBC BREAST CANCER AI
# XGBoost + SHAP + Streamlit
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="WDBC Breast Cancer AI",
    page_icon="🧬",
    layout="wide"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

.main {
    padding-top: 1rem;
}

.title {
    font-size: 42px;
    font-weight: 700;
    text-align: center;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    font-size: 18px;
    color: #777;
    margin-bottom: 30px;
}

.section {
    padding: 15px;
    border-radius: 12px;
    margin-top: 20px;
    margin-bottom: 15px;
    border: 1px solid rgba(128,128,128,0.25);
}

.result-box {
    padding: 25px;
    border-radius: 15px;
    text-align: center;
    border: 1px solid rgba(128,128,128,0.25);
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# TITLE
# ============================================================

st.markdown(
    '<div class="title">🧬 WDBC Breast Cancer AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'XGBoost Classification with SHAP Explainability'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    model = joblib.load("wdbc_xgboost.pkl")

    return model


@st.cache_resource
def load_feature_names():

    names = joblib.load("feature_names.pkl")

    return list(names)


try:

    model = load_model()
    feature_names = load_feature_names()

except Exception as e:

    st.error(f"Error loading model or feature names: {e}")
    st.stop()


# ============================================================
# STANDARD WDBC FEATURE NAMES
# ============================================================

standard_features = [
    "mean radius",
    "mean texture",
    "mean perimeter",
    "mean area",
    "mean smoothness",
    "mean compactness",
    "mean concavity",
    "mean concave points",
    "mean symmetry",
    "mean fractal dimension",

    "radius error",
    "texture error",
    "perimeter error",
    "area error",
    "smoothness error",
    "compactness error",
    "concavity error",
    "concave points error",
    "symmetry error",
    "fractal dimension error",

    "worst radius",
    "worst texture",
    "worst perimeter",
    "worst area",
    "worst smoothness",
    "worst compactness",
    "worst concavity",
    "worst concave points",
    "worst symmetry",
    "worst fractal dimension"
]


# ============================================================
# DISPLAY-FRIENDLY NAMES
# ============================================================

display_names = [
    "Mean Radius",
    "Mean Texture",
    "Mean Perimeter",
    "Mean Area",
    "Mean Smoothness",
    "Mean Compactness",
    "Mean Concavity",
    "Mean Concave Points",
    "Mean Symmetry",
    "Mean Fractal Dimension",

    "Radius SE",
    "Texture SE",
    "Perimeter SE",
    "Area SE",
    "Smoothness SE",
    "Compactness SE",
    "Concavity SE",
    "Concave Points SE",
    "Symmetry SE",
    "Fractal Dimension SE",

    "Worst Radius",
    "Worst Texture",
    "Worst Perimeter",
    "Worst Area",
    "Worst Smoothness",
    "Worst Compactness",
    "Worst Concavity",
    "Worst Concave Points",
    "Worst Symmetry",
    "Worst Fractal Dimension"
]


# ============================================================
# FEATURE GROUPS
# ============================================================

mean_features = display_names[0:10]

se_features = display_names[10:20]

worst_features = display_names[20:30]


# ============================================================
# FEATURE INPUT SECTION
# ============================================================

st.markdown("---")

st.header("🔬 Patient Feature Parameters")

st.caption(
    "Enter the 30 diagnostic measurements used by the WDBC XGBoost model."
)


# Store inputs in exact model order
inputs = []


# ------------------------------------------------------------
# MEAN FEATURES
# ------------------------------------------------------------

with st.container():

    st.markdown(
        '<div class="section">',
        unsafe_allow_html=True
    )

    st.subheader("📊 Mean Measurements")

    cols = st.columns(3)

    for i, feature in enumerate(mean_features):

        with cols[i % 3]:

            value = st.number_input(
                label=feature,
                min_value=0.0,
                value=0.0,
                step=0.01,
                format="%.4f",
                key=f"mean_{i}"
            )

            inputs.append(value)

    st.markdown("</div>", unsafe_allow_html=True)


# ------------------------------------------------------------
# STANDARD ERROR FEATURES
# ------------------------------------------------------------

with st.container():

    st.markdown(
        '<div class="section">',
        unsafe_allow_html=True
    )

    st.subheader("📈 Standard Error (SE) Measurements")

    cols = st.columns(3)

    for i, feature in enumerate(se_features):

        with cols[i % 3]:

            value = st.number_input(
                label=feature,
                min_value=0.0,
                value=0.0,
                step=0.01,
                format="%.4f",
                key=f"se_{i}"
            )

            inputs.append(value)

    st.markdown("</div>", unsafe_allow_html=True)


# ------------------------------------------------------------
# WORST FEATURES
# ------------------------------------------------------------

with st.container():

    st.markdown(
        '<div class="section">',
        unsafe_allow_html=True
    )

    st.subheader("🔬 Worst Measurements")

    cols = st.columns(3)

    for i, feature in enumerate(worst_features):

        with cols[i % 3]:

            value = st.number_input(
                label=feature,
                min_value=0.0,
                value=0.0,
                step=0.01,
                format="%.4f",
                key=f"worst_{i}"
            )

            inputs.append(value)

    st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# CHECK FEATURE COUNT
# ============================================================

if len(inputs) != 30:

    st.error(
        f"Expected 30 features, but received {len(inputs)}."
    )

    st.stop()


# ============================================================
# INPUT DATAFRAME
# ============================================================

input_array = np.array(inputs).reshape(1, -1)


# Use the model's original feature names when possible
if len(feature_names) == 30:

    model_columns = feature_names

else:

    model_columns = standard_features


input_df = pd.DataFrame(
    input_array,
    columns=model_columns
)


# ============================================================
# PREDICTION BUTTON
# ============================================================

st.markdown("---")

predict_button = st.button(
    "🔍 Predict Breast Cancer",
    type="primary",
    use_container_width=True
)


# ============================================================
# PREDICTION
# ============================================================

if predict_button:

    try:

        # ----------------------------------------------------
        # MODEL PREDICTION
        # ----------------------------------------------------

        prediction = model.predict(input_df)[0]

        probabilities = model.predict_proba(input_df)[0]

        predicted_probability = float(
            np.max(probabilities)
        )


        # ----------------------------------------------------
        # HANDLE LABEL
        # ----------------------------------------------------

        if prediction in [0, "0"]:

            result = "Benign"

        elif prediction in [1, "1"]:

            result = "Malignant"

        else:

            result = str(prediction)


        # ----------------------------------------------------
        # RESULT
        # ----------------------------------------------------

        st.markdown("---")

        st.header("🩺 Prediction Result")


        col1, col2 = st.columns(2)


        with col1:

            st.metric(
                "Prediction",
                result
            )


        with col2:

            st.metric(
                "Confidence",
                f"{predicted_probability * 100:.2f}%"
            )


        # ----------------------------------------------------
        # RESULT MESSAGE
        # ----------------------------------------------------

        if result.lower() == "malignant":

            st.error(
                "⚠️ Model prediction: Malignant"
            )

        elif result.lower() == "benign":

            st.success(
                "✅ Model prediction: Benign"
            )

        else:

            st.info(
                f"Model prediction: {result}"
            )


        # ====================================================
        # PROBABILITY TABLE
        # ====================================================

        st.subheader("📊 Prediction Probability")

        probability_df = pd.DataFrame(
            {
                "Class": ["Class 0", "Class 1"],
                "Probability": probabilities
            }
        )

        probability_df["Probability"] = (
            probability_df["Probability"] * 100
        ).round(2)

        probability_df["Probability"] = (
            probability_df["Probability"].astype(str) + "%"
        )

        st.dataframe(
            probability_df,
            use_container_width=True,
            hide_index=True
        )


        # ====================================================
        # SHAP EXPLAINABILITY
        # ====================================================

        st.markdown("---")

        st.header("🧠 SHAP Explainability")

        st.caption(
            "SHAP shows which features contributed most "
            "to the model's prediction."
        )


        try:

            # ------------------------------------------------
            # CREATE SHAP EXPLAINER
            # ------------------------------------------------

            explainer = shap.TreeExplainer(model)

            shap_values = explainer.shap_values(input_df)


            # ------------------------------------------------
            # HANDLE DIFFERENT SHAP OUTPUT FORMATS
            # ------------------------------------------------

            if isinstance(shap_values, list):

                # Binary classification
                if len(shap_values) > 1:

                    shap_for_plot = shap_values[1][0]

                else:

                    shap_for_plot = shap_values[0][0]

            else:

                shap_array = np.asarray(shap_values)

                if shap_array.ndim == 3:

                    shap_for_plot = shap_array[0, :, 1]

                elif shap_array.ndim == 2:

                    shap_for_plot = shap_array[0]

                else:

                    shap_for_plot = shap_array


            # ------------------------------------------------
            # SHAP DATAFRAME
            # ------------------------------------------------

            shap_df = pd.DataFrame(
                {
                    "Feature": display_names,
                    "Value": inputs,
                    "SHAP Value": shap_for_plot
                }
            )


            shap_df["Absolute SHAP"] = (
                shap_df["SHAP Value"].abs()
            )


            shap_df = shap_df.sort_values(
                "Absolute SHAP",
                ascending=False
            )


            # ------------------------------------------------
            # TOP FEATURES TABLE
            # ------------------------------------------------

            st.subheader("🔎 Top Influential Features")

            top_features = shap_df.head(10).copy()

            top_features = top_features[
                [
                    "Feature",
                    "Value",
                    "SHAP Value"
                ]
            ]

            top_features["Value"] = (
                top_features["Value"].round(4)
            )

            top_features["SHAP Value"] = (
                top_features["SHAP Value"].round(5)
            )

            st.dataframe(
                top_features,
                use_container_width=True,
                hide_index=True
            )


            # =================================================
            # SHAP BAR CHART
            # =================================================

            st.subheader("📈 Feature Contribution")

            plot_df = shap_df.head(10).sort_values(
                "SHAP Value"
            )


            fig, ax = plt.subplots(
                figsize=(10, 6)
            )

            ax.barh(
                plot_df["Feature"],
                plot_df["SHAP Value"]
            )

            ax.axvline(
                0,
                linewidth=1
            )

            ax.set_xlabel(
                "SHAP Value"
            )

            ax.set_ylabel(
                "Feature"
            )

            ax.set_title(
                "Top 10 SHAP Feature Contributions"
            )

            plt.tight_layout()

            st.pyplot(fig)


            # ------------------------------------------------
            # SHAP INTERPRETATION
            # ------------------------------------------------

            st.info(
                """
                **How to interpret SHAP:**

                • Positive SHAP value → pushes the prediction toward Class 1.

                • Negative SHAP value → pushes the prediction toward Class 0.

                • Larger absolute SHAP value → stronger influence on the prediction.

                **Note:** SHAP explains the machine-learning model's decision;
                it is not a medical diagnosis.
                """
            )


        except Exception as shap_error:

            st.warning(
                f"SHAP analysis could not be generated: {shap_error}"
            )


        # ====================================================
        # INPUT SUMMARY
        # ====================================================

        with st.expander("📋 View All 30 Input Parameters"):

            summary_df = pd.DataFrame(
                {
                    "Feature": display_names,
                    "Value": inputs
                }
            )

            st.dataframe(
                summary_df,
                use_container_width=True,
                hide_index=True
            )


    except Exception as prediction_error:

        st.error(
            f"Prediction error: {prediction_error}"
        )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("🧬 WDBC AI")

    st.markdown("---")

    st.subheader("Model")

    st.write("**XGBoost Classifier**")

    st.subheader("Explainability")

    st.write("**SHAP TreeExplainer**")

    st.subheader("Dataset")

    st.write(
        "Wisconsin Diagnostic Breast Cancer (WDBC)"
    )

    st.markdown("---")

    st.caption(
        "For educational and research purposes only. "
        "This application is not a substitute for professional medical diagnosis."
    )
