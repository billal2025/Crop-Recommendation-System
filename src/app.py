import streamlit as st
import joblib
import pandas as pd

# Load model
model = joblib.load(r"C:\Users\Administrator\PycharmProjects\crop-recommendation-system\models\crop_recommendation_model.pkl")

# UI Layout
st.title("Crop Treatment Recommendation System 🌾")
st.write("Enter the following information to get a recommendation:")

# Input fields
problem = st.selectbox("Problem", ["Pest infestation", "Nutrient deficiency", "Drought", "Fungal disease", "Weed overgrowth"])
crop = st.selectbox("Crop", ["Corn", "Wheat", "Rice", "Potato", "Tomato"])
season = st.selectbox("Season", ["Spring", "Summer", "Autumn", "Winter"])
weather = st.selectbox("Weather", ["Hot", "Humid", "Dry", "Rainy"])
soil = st.selectbox("Soil Type", ["Clay", "Sandy", "Loamy", "Silty", "Peaty", "Chalky", "Saline", "Laterite", "Black"])
pest = st.selectbox("Pest Type", ["None", "Aphids", "Cutworms", "Whiteflies", "Spider Mites",
                                   "Root-Knot Nematodes", "Armyworms", "Stink Bugs", "Thrips",
                                   "Cabbage Loopers", "Fruit Flies"])

# Button to trigger prediction
if st.button("Get Recommendation"):
    # Prepare input for prediction
    user_data = pd.DataFrame([[problem, crop, season, weather, soil, pest]],
                              columns=["Problem", "Crop", "Season", "Weather", "Soil Type", "Pest Type"])
    prediction = model.predict(user_data)
    st.success(f"Recommended Treatment: {prediction[0]}")
