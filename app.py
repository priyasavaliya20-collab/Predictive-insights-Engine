import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from model import (
    model,
    data,
    features,
    mae,
    rmse,
    r2
)

from sklearn.model_selection import train_test_split


# -----------------------------------------
# PAGE
# -----------------------------------------

st.set_page_config(
    page_title="Predictive Insight Engine",
    page_icon="🏠",
    layout="wide"
)


# -----------------------------------------
# TITLE
# -----------------------------------------

st.title("🏠 Predictive Insight Engine")

st.write(
    "House Price Prediction using Multiple Linear Regression"
)


# -----------------------------------------
# SIDEBAR
# -----------------------------------------

st.sidebar.header("Enter Property Details")


area = st.sidebar.number_input(
    "House Area (sq.ft)",
    min_value=300,
    max_value=10000,
    value=1500
)


bedrooms = st.sidebar.number_input(
    "Number of Bedrooms",
    min_value=1,
    max_value=10,
    value=3
)


bathrooms = st.sidebar.number_input(
    "Number of Bathrooms",
    min_value=1,
    max_value=10,
    value=2
)


location_score = st.sidebar.number_input(
    "Location Score",
    min_value=0.0,
    max_value=10.0,
    value=5.0,
    step=0.1
)


age_years = st.sidebar.number_input(
    "Age of Property",
    min_value=0,
    max_value=100,
    value=10
)


# -----------------------------------------
# PREDICT
# -----------------------------------------

if st.sidebar.button("🔮 Predict House Price"):

    input_data = pd.DataFrame(
        [[
            area,
            bedrooms,
            bathrooms,
            location_score,
            age_years
        ]],
        columns=features
    )

    prediction = model.predict(input_data)[0]

    st.success("Prediction generated successfully!")

    st.metric(
        "Predicted House Price",
        f"₹ {prediction:,.0f}"
    )


# -----------------------------------------
# MODEL PERFORMANCE
# -----------------------------------------

st.header("📊 Model Performance")


col1, col2, col3 = st.columns(3)


with col1:
    st.metric(
        "R² Score",
        f"{r2:.4f}"
    )


with col2:
    st.metric(
        "MAE",
        f"₹ {mae:,.0f}"
    )


with col3:
    st.metric(
        "RMSE",
        f"₹ {rmse:,.0f}"
    )


# -----------------------------------------
# DATASET
# -----------------------------------------

st.header("📂 Dataset")

st.write(
    f"Total Records: {len(data):,}"
)

st.dataframe(
    data.head(20),
    use_container_width=True
)


# -----------------------------------------
# ACTUAL VS PREDICTED
# -----------------------------------------

st.header("📈 Actual vs Predicted")


X = data[features]
y = data[target] if "target" in globals() else data["house_price_inr"]


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


predictions = model.predict(X_test)


fig, ax = plt.subplots(figsize=(10, 6))


ax.scatter(
    y_test,
    predictions,
    alpha=0.5
)


min_value = min(
    y_test.min(),
    predictions.min()
)

max_value = max(
    y_test.max(),
    predictions.max()
)


ax.plot(
    [min_value, max_value],
    [min_value, max_value],
    linestyle="--"
)


ax.set_xlabel("Actual House Price")

ax.set_ylabel("Predicted House Price")

ax.set_title("Actual vs Predicted House Prices")


st.pyplot(fig)


# -----------------------------------------
# COEFFICIENTS
# -----------------------------------------

st.header("📌 Model Coefficients")


coefficient_data = pd.DataFrame({
    "Feature": features,
    "Coefficient": model.coef_
})


st.dataframe(
    coefficient_data,
    use_container_width=True,
    hide_index=True
)


# -----------------------------------------
# FOOTER
# -----------------------------------------

st.markdown("---")

st.caption(
    "Predictive Insight Engine | Multiple Linear Regression"
)