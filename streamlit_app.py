import streamlit as st
import pandas as pd

from snowflake.ml.registry import Registry


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Bank Marketing Predictor",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# HEADER
# =========================================================

st.title("🏦 Bank Marketing Response Predictor")

st.markdown(
    """
    ### Predict customer subscription

    This application uses a machine learning model trained on
    the UCI Bank Marketing dataset and registered in Snowflake.

    Enter the customer's information in the sidebar and click
    **Predict Subscription**.
    """
)


# =========================================================
# SNOWFLAKE CONNECTION
# =========================================================

try:
    # Container Runtime recommended connection method
    conn = st.connection("snowflake")
    session = conn.session()

    # -----------------------------------------------------
    # IMPORTANT:
    # Explicitly set the Snowflake session context.
    # This fixes the "current database" error that occurred
    # when model.run() tried to create a temporary stage.
    # -----------------------------------------------------

    session.sql(
        "USE DATABASE BANK_ML_DB"
    ).collect()

    session.sql(
        "USE SCHEMA BANK_ML_DB.ML"
    ).collect()

    session.sql(
        "USE WAREHOUSE BANK_ML_WH"
    ).collect()

except Exception as e:

    st.error(
        "❌ Could not connect to Snowflake."
    )

    st.exception(e)

    st.stop()


# =========================================================
# MODEL REGISTRY
# =========================================================

try:

    registry = Registry(
        session=session,
        database_name="BANK_ML_DB",
        schema_name="ML"
    )

    # V2 was explicitly registered for Warehouse inference
    model = (
        registry
        .get_model("BANK_RESPONSE_MODEL")
        .version("V2")
    )

except Exception as e:

    st.error(
        "❌ Could not load BANK_RESPONSE_MODEL V2."
    )

    st.exception(e)

    st.stop()


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.header("👤 Customer Information")

st.sidebar.markdown(
    "Enter the customer details below."
)


# ---------------------------------------------------------
# AGE
# ---------------------------------------------------------

age = st.sidebar.slider(
    "Age",
    min_value=18,
    max_value=95,
    value=35
)


# ---------------------------------------------------------
# JOB
# ---------------------------------------------------------

job = st.sidebar.selectbox(
    "Job",
    [
        "admin.",
        "blue-collar",
        "entrepreneur",
        "housemaid",
        "management",
        "retired",
        "self-employed",
        "services",
        "student",
        "technician",
        "unemployed",
        "unknown"
    ]
)


# ---------------------------------------------------------
# MARITAL STATUS
# ---------------------------------------------------------

marital = st.sidebar.selectbox(
    "Marital Status",
    [
        "divorced",
        "married",
        "single"
    ]
)


# ---------------------------------------------------------
# EDUCATION
# ---------------------------------------------------------

education = st.sidebar.selectbox(
    "Education",
    [
        "primary",
        "secondary",
        "tertiary",
        "unknown"
    ]
)


# ---------------------------------------------------------
# CREDIT DEFAULT
# ---------------------------------------------------------

default = st.sidebar.selectbox(
    "Credit Default",
    [
        "no",
        "yes"
    ]
)


# ---------------------------------------------------------
# BALANCE
# ---------------------------------------------------------

balance = st.sidebar.number_input(
    "Account Balance",
    min_value=-100000,
    max_value=1000000,
    value=2500,
    step=100
)


# ---------------------------------------------------------
# HOUSING LOAN
# ---------------------------------------------------------

housing = st.sidebar.selectbox(
    "Housing Loan",
    [
        "no",
        "yes"
    ]
)


# ---------------------------------------------------------
# PERSONAL LOAN
# ---------------------------------------------------------

loan = st.sidebar.selectbox(
    "Personal Loan",
    [
        "no",
        "yes"
    ]
)


# ---------------------------------------------------------
# CONTACT TYPE
# ---------------------------------------------------------

contact = st.sidebar.selectbox(
    "Contact Type",
    [
        "cellular",
        "telephone",
        "unknown"
    ]
)


# ---------------------------------------------------------
# DAY OF CONTACT
# ---------------------------------------------------------

day = st.sidebar.slider(
    "Contact Day",
    min_value=1,
    max_value=31,
    value=15
)


# ---------------------------------------------------------
# MONTH
# ---------------------------------------------------------

month = st.sidebar.selectbox(
    "Contact Month",
    [
        "jan",
        "feb",
        "mar",
        "apr",
        "may",
        "jun",
        "jul",
        "aug",
        "sep",
        "oct",
        "nov",
        "dec"
    ]
)


# ---------------------------------------------------------
# CURRENT CAMPAIGN CONTACTS
# ---------------------------------------------------------

campaign = st.sidebar.number_input(
    "Contacts in Current Campaign",
    min_value=1,
    max_value=100,
    value=1,
    step=1
)


# ---------------------------------------------------------
# DAYS SINCE PREVIOUS CONTACT
# ---------------------------------------------------------

pdays = st.sidebar.number_input(
    "Days Since Previous Contact",
    min_value=-1,
    max_value=1000,
    value=-1,
    step=1
)


# ---------------------------------------------------------
# PREVIOUS CAMPAIGN CONTACTS
# ---------------------------------------------------------

previous = st.sidebar.number_input(
    "Previous Campaign Contacts",
    min_value=0,
    max_value=100,
    value=0,
    step=1
)


# ---------------------------------------------------------
# PREVIOUS CAMPAIGN OUTCOME
# ---------------------------------------------------------

poutcome = st.sidebar.selectbox(
    "Previous Campaign Outcome",
    [
        "unknown",
        "failure",
        "success",
        "other"
    ]
)


# =========================================================
# FEATURE ENGINEERING
# =========================================================
#
# These MUST match the feature engineering used during
# model training.
#
# duration is intentionally NOT included because it was
# excluded from the training features.
# =========================================================

total_contacts = campaign + previous


if age < 30:

    age_group = "18-29"

elif age < 40:

    age_group = "30-39"

elif age < 50:

    age_group = "40-49"

elif age < 60:

    age_group = "50-59"

else:

    age_group = "60+"


# =========================================================
# CREATE INPUT DATAFRAME
# =========================================================

input_data = pd.DataFrame(
    [
        {
            "age": age,
            "job": job,
            "marital": marital,
            "education": education,
            "default": default,
            "balance": balance,
            "housing": housing,
            "loan": loan,
            "contact": contact,
            "day": day,
            "month": month,
            "campaign": campaign,
            "pdays": pdays,
            "previous": previous,
            "poutcome": poutcome,
            "total_contacts": total_contacts,
            "age_group": age_group
        }
    ]
)


# =========================================================
# MAIN INPUT PREVIEW
# =========================================================

st.subheader("📋 Customer Information")

st.dataframe(
    input_data,
    use_container_width=True,
    hide_index=True
)


# =========================================================
# MODEL INFORMATION
# =========================================================

with st.expander("🤖 Model Information"):

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Model",
            "BANK_RESPONSE_MODEL"
        )

    with col2:
        st.metric(
            "Version",
            "V2"
        )

    with col3:
        st.metric(
            "Inference",
            "Snowflake Warehouse"
        )


# =========================================================
# PREDICTION BUTTON
# =========================================================

st.divider()

predict_button = st.button(
    "🔮 Predict Subscription",
    type="primary",
    use_container_width=True
)


# =========================================================
# PREDICTION
# =========================================================

if predict_button:

    try:

        with st.spinner(
            "Running the Snowflake ML model..."
        ):

            # -------------------------------------------------
            # PREDICTION
            # -------------------------------------------------

            prediction_result = model.run(
                input_data,
                function_name='"predict"'
            )

            # -------------------------------------------------
            # PROBABILITY
            # -------------------------------------------------

            probability_result = model.run(
                input_data,
                function_name='"predict_proba"'
            )


        # =====================================================
        # EXTRACT RESULTS
        # =====================================================

        predicted_class = int(
            prediction_result.iloc[0, 0]
        )

        subscribe_probability = float(
            probability_result.iloc[0, 1]
        )

        not_subscribe_probability = float(
            probability_result.iloc[0, 0]
        )


        # =====================================================
        # RESULT HEADER
        # =====================================================

        st.subheader("🎯 Prediction Result")


        # =====================================================
        # RESULT COLUMNS
        # =====================================================

        col1, col2 = st.columns(2)


        # -----------------------------------------------------
        # PREDICTION
        # -----------------------------------------------------

        with col1:

            if predicted_class == 1:

                st.success(
                    "✅ Customer is predicted to SUBSCRIBE"
                )

            else:

                st.error(
                    "❌ Customer is predicted NOT TO SUBSCRIBE"
                )


        # -----------------------------------------------------
        # SUBSCRIPTION PROBABILITY
        # -----------------------------------------------------

        with col2:

            st.metric(
                "Subscription Probability",
                f"{subscribe_probability:.2%}"
            )


        # =====================================================
        # PROBABILITY DETAILS
        # =====================================================

        st.markdown("### 📊 Probability Breakdown")


        col1, col2 = st.columns(2)


        with col1:

            st.metric(
                "Not Subscribe",
                f"{not_subscribe_probability:.2%}"
            )


        with col2:

            st.metric(
                "Subscribe",
                f"{subscribe_probability:.2%}"
            )


        # =====================================================
        # PROGRESS BAR
        # =====================================================

        st.markdown(
            "#### Subscription Probability"
        )

        st.progress(
            min(
                max(subscribe_probability, 0.0),
                1.0
            )
        )


        # =====================================================
        # INTERPRETATION
        # =====================================================

        if predicted_class == 1:

            st.info(
                f"""
                The model predicts **SUBSCRIBE** with a
                probability of **{subscribe_probability:.2%}**.
                """
            )

        else:

            st.info(
                f"""
                The model predicts **NOT SUBSCRIBE** with a
                subscription probability of
                **{subscribe_probability:.2%}**.
                """
            )


        # =====================================================
        # TECHNICAL OUTPUT
        # =====================================================

        with st.expander("🔍 Technical Prediction Output"):

            st.write(
                "Predicted class:",
                predicted_class
            )

            st.write(
                "Class 0 probability:",
                f"{not_subscribe_probability:.6f}"
            )

            st.write(
                "Class 1 probability:",
                f"{subscribe_probability:.6f}"
            )

            st.write(
                "Model:",
                "BANK_RESPONSE_MODEL"
            )

            st.write(
                "Version:",
                "V2"
            )


    except Exception as e:

        st.error(
            "❌ Prediction failed."
        )

        st.markdown(
            """
            **Possible causes:**

            1. The Snowflake warehouse is unavailable.
            2. The role does not have permission to use the model.
            3. The model input columns do not match the registered signature.
            4. The registered model version is not available for warehouse inference.
            """
        )

        st.exception(e)


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "Built with Snowflake ML Model Registry + Streamlit"
)