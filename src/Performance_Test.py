import shap
import joblib
import pandas as pd
import matplotlib.pyplot as plt

# Load the model and sample data
model = joblib.load("../models/crop_recommendation_model.pkl")
df = pd.read_csv("../data/enhanced_synthetic_crop_data.csv")

# Extract features used for training
X = df[["Problem", "Crop", "Season", "Weather", "Soil Type", "Pest Type"]]

# Preprocess features using the model's internal preprocessor
preprocessed_X = model.named_steps['columntransformer'].transform(X)

# Convert sparse matrix to dense array
preprocessed_X_dense = preprocessed_X.toarray() if hasattr(preprocessed_X, "toarray") else preprocessed_X

# Extract feature names from the transformer
feature_names = model.named_steps['columntransformer'].get_feature_names_out()

# Initialize SHAP explainer
explainer = shap.Explainer(model.named_steps['randomforestclassifier'], preprocessed_X_dense, feature_names=feature_names)

# Explain the model's predictions for a subset
shap_values = explainer(preprocessed_X_dense[:50])

# Visualize feature importance
shap.summary_plot(shap_values, preprocessed_X_dense[:50], feature_names=feature_names)
plt.show()
