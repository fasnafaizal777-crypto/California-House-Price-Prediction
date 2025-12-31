import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import base64

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# ------------------------------------------
# BACKGROUND IMAGE FUNCTION
# ------------------------------------------
def set_bg(image_file):
    with open(image_file, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()

    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )


# ------------------------------------------
# PAGE STATE
# ------------------------------------------
if "page" not in st.session_state:
    st.session_state.page = "home"


# ------------------------------------------
# FRONT PAGE (WITH BACKGROUND IMAGE)
# ------------------------------------------
if st.session_state.page == "home":

    set_bg("bg.jpg")   # ✅ Background ONLY here

    st.markdown("<br><br><br>", unsafe_allow_html=True)
    st.markdown(
        "<h1 style='text-align:center; color:white;white-space:nowrap;'>🏡 California House Price Prediction System</h1>",
        unsafe_allow_html=True
    )

    st.markdown(
        "<p style='text-align:center; color:White;'>"
        "A Machine Learning project using Linear Regression<br>"
        "Built with Python, Scikit-learn & Streamlit"
        "</p>",
        unsafe_allow_html=True
    )

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns([3,3,2])
    with col2:
        if st.button("🚀 Open Prediction App"):
            st.session_state.page = "predict"
            st.rerun()


# ------------------------------------------
# PREDICTION PAGE (NO BACKGROUND)
# ------------------------------------------
if st.session_state.page == "predict":
    st.image("ss.JPG", 
             use_container_width=True, 
             
            )
   
    st.markdown(
        "<h1 style='text-align:center; color:black;white-space:nowrap;'>🏡 California House Price Prediction </h1>",
        unsafe_allow_html=True
    )

    # Load dataset
    data = fetch_california_housing()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df["MedHouseValue"] = data.target

    # Remove outliers
    df = df[df["MedHouseValue"] < 5]

    # Prepare data
    X = df.drop("MedHouseValue", axis=1)
    y = df["MedHouseValue"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)

    st.subheader("Enter the values to predict house price")

    MedInc = st.number_input("Median Income", min_value=0.0)
    HouseAge = st.number_input("House Age", min_value=0.0)
    AveRooms = st.number_input("Average Rooms", min_value=0.0)
    AveBedrms = st.number_input("Average Bedrooms", min_value=0.0)
    Population = st.number_input("Population", min_value=0.0)
    AveOccup = st.number_input("Average Occupancy", min_value=0.0)
    Latitude = st.number_input("Latitude", min_value=0.0)
    Longitude = st.number_input("Longitude", min_value=0.0)

    if st.button("Predict Price"):
        input_data = np.array([[MedInc, HouseAge, AveRooms, AveBedrms,
                                Population, AveOccup, Latitude, Longitude]])
        prediction = model.predict(input_data)[0] * 100000
        st.success(f"Predicted House Price: ${prediction:,.2f}")

    # ------------------------------------------
    # HEATMAP
    # ------------------------------------------
    st.subheader("📊 Heatmap of Features")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(df.corr(), cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    # ------------------------------------------
    # SCATTER PLOT
    # ------------------------------------------
    st.subheader("📈 Median Income vs House Price")
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    ax2.scatter(df["MedInc"], df["MedHouseValue"], alpha=0.4)
    ax2.set_xlabel("Median Income")
    ax2.set_ylabel("Median House Value")
    st.pyplot(fig2)