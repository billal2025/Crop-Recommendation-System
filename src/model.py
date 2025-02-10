import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report, accuracy_score
import shap
import joblib
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv(r"C:\Users\Administrator\PycharmProjects\crop-recommendation-system\data\enhanced_synthetic_crop_data.csv")

# Define features and target
X = df[["Problem", "Crop", "Season", "Weather", "Soil Type", "Pest Type"]]
y = df["Recommendation"]

# Split the data: 70% train, 15% validation, 15% test
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("problem_tfidf", TfidfVectorizer(), "Problem"),
        ("onehot", OneHotEncoder(handle_unknown='ignore'), ["Crop", "Season", "Weather", "Soil Type", "Pest Type"])
    ]
)

# Build pipeline
model = make_pipeline(preprocessor, RandomForestClassifier(random_state=42))

# Train the model
model.fit(X_train, y_train)

# Save the trained model
model_path = r"C:\Users\Administrator\PycharmProjects\crop-recommendation-system\models\crop_recommendation_model.pkl"
joblib.dump(model, model_path)
print(f"Model saved successfully at: {model_path}")

# Validate the model
val_predictions = model.predict(X_val)
print("Validation Accuracy:", accuracy_score(y_val, val_predictions))

# Test the model on unseen data
test_predictions = model.predict(X_test)
print("Test Accuracy:", accuracy_score(y_test, test_predictions))

# Detailed Performance Report
print("\nClassification Report on Test Data:")
print(classification_report(y_test, test_predictions))

# Sample test cases
sample_data = pd.DataFrame([
    {"Problem": "Pest infestation", "Crop": "Tomato", "Season": "Summer", "Weather": "Humid",
     "Soil Type": "Loamy", "Pest Type": "Aphids"},
    {"Problem": "Drought", "Crop": "Rice", "Season": "Summer", "Weather": "Dry",
     "Soil Type": "Clay", "Pest Type": "None"},
    {"Problem": "Fungal disease", "Crop": "Potato", "Season": "Spring", "Weather": "Rainy",
     "Soil Type": "Black", "Pest Type": "None"}
])

# Get model predictions
sample_predictions = model.predict(sample_data)
for i, rec in enumerate(sample_predictions):
    print(f"Sample {i + 1} Recommendation: {rec}")

